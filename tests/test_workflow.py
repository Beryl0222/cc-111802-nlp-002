import json
import tempfile
import unittest
from pathlib import Path

from nlp_homework_authorship_review.annotations import Annotation
from nlp_homework_authorship_review.repository import RecordRepository
from nlp_homework_authorship_review.text import locate_quote, normalize_text, tokenize
from nlp_homework_authorship_review.workflow import TextWorkflow


FIXTURE = json.loads(Path("fixtures/records.json").read_text(encoding="utf-8"))
REVISION2 = json.loads(Path("fixtures/records_revision2.json").read_text(encoding="utf-8"))


def make_workflow(directory):
    return TextWorkflow(RecordRepository(Path(directory) / "records.json"))


class TextWorkflowTests(unittest.TestCase):
    def test_normalization_and_tokenization(self):
        self.assertEqual(normalize_text("  Ａ  B\nC  "), "A B C")
        self.assertEqual(tokenize("语文课 A2。"), ["语", "文", "课", "A", "2", "。"])

    def test_locate_quote_keeps_original_offsets_under_nfkc(self):
        # 全角字符、合字展开与空白压缩都不能让原文偏移错位。
        text = "  ＡＢ　教材：观察是写作的基础。\n收尾"
        location = locate_quote(text, "观察是写作的基础。")
        candidate = location["candidates"][0]
        self.assertEqual(
            text[candidate["original_start"]:candidate["original_end"]],
            "观察是写作的基础。",
        )
        ligature = locate_quote("前缀 ﬁle 后缀", "file")["candidates"][0]
        self.assertEqual(
            "前缀 ﬁle 后缀"[ligature["original_start"]:ligature["original_end"]], "ﬁle",
        )
        # 标点变化时退化为词元匹配，仍能切回包含原标点的原文区间。
        fallback_text = "细致观察，能够积累素材。"
        fallback = locate_quote(fallback_text, "细致观察能够积累素材")
        self.assertEqual(fallback["match_kind"], "token")
        candidate = fallback["candidates"][0]
        self.assertEqual(
            fallback_text[candidate["original_start"]:candidate["original_end"]],
            "细致观察，能够积累素材",
        )

    def test_ingest_persists_analysis_and_uses_newer_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = make_workflow(directory)
            first = {"record_id": "lesson-1", "source": "synthetic", "revision": 1,
                     "observed_at": "2026-11-18T09:00:00Z", "text": "原稿 甲乙", "context": "课堂"}
            second = {**first, "revision": 2, "text": "修订稿 乙丙"}
            self.assertEqual(workflow.ingest([first]), {"received": 1, "stored": 1, "annotations": 0})
            workflow.ingest([second, first])
            result = workflow.list_records()[0]
            self.assertEqual(result["revision"], 2)
            self.assertEqual(result["text"], "修订稿 乙丙")
            self.assertEqual(result["analysis"]["token_count"], 5)

    def test_invalid_records_are_rejected_without_partial_write(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = RecordRepository(Path(directory) / "records.json")
            workflow = TextWorkflow(repository)
            valid = {"record_id": "one", "source": "synthetic", "revision": 1,
                     "observed_at": "2026-11-18T09:00:00Z", "text": "有效文本"}
            workflow.ingest([valid])
            invalid = {**valid, "record_id": "two", "observed_at": "bad-time"}
            with self.assertRaises(ValueError):
                workflow.ingest([valid, invalid])
            self.assertEqual([row["record_id"] for row in repository.read()], ["one"])

    def test_source_filter_and_fixture_are_local(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = make_workflow(directory)
            workflow.ingest(FIXTURE)
            self.assertEqual(len(workflow.list_records("school-synthetic")), 2)
            self.assertEqual(workflow.list_records("missing"), [])

    def test_original_text_and_original_positions_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = make_workflow(directory)
            record = {
                "record_id": "r1", "source": "school-synthetic", "revision": 1,
                "observed_at": "2026-11-18T09:00:00Z",
                "text": "  教材指出：“观察是写作的基础。”\n学生补充了记录。 ",
                "annotation_proposals": [{
                    "category": "citation", "quote": "观察是写作的基础。",
                    "source_ref": "教材第三单元", "rationale": "直接引用", "confidence": 0.9,
                }],
            }
            workflow.ingest([record])
            stored = workflow.get_record("r1")
            self.assertEqual(stored["original_text"], record["text"])
            annotation = stored["annotations"][0]
            start, end = annotation["anchor"]["original_start"], annotation["anchor"]["original_end"]
            # 标注位置必须能在原始未规范化文本中切回原句，而不是只对应规范化后的位置。
            self.assertEqual(stored["original_text"][start:end], "观察是写作的基础。")
            self.assertEqual(annotation["state"], "pending_manual")

    def test_repeated_citations_get_independent_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = make_workflow(directory)
            workflow.ingest(FIXTURE)
            record = workflow.get_record("homework-authorship-review-lesson-a")
            citations = [a for a in record["annotations"] if a["category"] == "citation"]
            self.assertEqual(len(citations), 2)
            self.assertNotEqual(citations[0]["annotation_id"], citations[1]["annotation_id"])
            # 同一句原文被两次引用时，依据不同则各自独立，不得互相覆盖。
            self.assertIn("第二次引用", citations[1]["rationale"])
            # 第二处引用在原文中对应同一来源文字的另一次出现，独立进入待人工状态而非并入第一条。
            self.assertEqual(citations[1]["state"], "pending_manual")
            self.assertNotEqual(citations[0]["annotation_id"], citations[1]["annotation_id"])
            # 两句引用在文本中各有位置候选，系统不替教师选择具体位置。
            self.assertEqual(citations[0]["anchor"]["candidate_count"], 2)
            self.assertEqual(citations[1]["anchor"]["candidate_count"], 2)
            self.assertIsNone(citations[0]["anchor"]["start"])

    def test_insufficient_evidence_never_produces_violation_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = make_workflow(directory)
            workflow.ingest(FIXTURE)
            record = workflow.get_record("homework-authorship-review-lesson-a")
            tool_annotation = next(
                a for a in record["annotations"] if a["category"] == "tool_suggestion"
            )
            self.assertEqual(tool_annotation["state"], "pending_manual")
            self.assertIn("未找到", tool_annotation["status_reason"])
            missing_source = {
                "record_id": "r2", "source": "school-synthetic", "revision": 1,
                "observed_at": "2026-11-18T09:00:00Z", "text": "这是一段改写文字。",
                "annotation_proposals": [{
                    "category": "paraphrase", "quote": "改写文字",
                    "rationale": "疑似改写", "confidence": 0.4,
                }],
            }
            workflow.ingest([missing_source])
            annotation = workflow.get_record("r2")["annotations"][0]
            self.assertEqual(annotation["state"], "pending_manual")
            self.assertIn("来源依据不足", annotation["status_reason"])
            self.assertIn("低于自动确认门槛", annotation["status_reason"])
            for summary in workflow.summaries():
                # 系统只陈述待复核状态，绝不输出确定性的代写/违规结论。
                self.assertIn("系统不作", summary["verdict"])

    def test_ambiguous_match_is_not_silently_chosen(self):
        location = locate_quote("同一处 同一处 结尾", "同一处")
        self.assertEqual(location["match_kind"], "exact")
        self.assertEqual(len(location["candidates"]), 2)
        with tempfile.TemporaryDirectory() as directory:
            workflow = make_workflow(directory)
            record = {
                "record_id": "r3", "source": "school-synthetic", "revision": 1,
                "observed_at": "2026-11-18T09:00:00Z", "text": "同一处 同一处 结尾",
                "annotation_proposals": [{
                    "category": "citation", "quote": "同一处",
                    "source_ref": "教材", "rationale": "重复出现的引用", "confidence": 0.9,
                }],
            }
            workflow.ingest([record])
            annotation = workflow.get_record("r3")["annotations"][0]
            self.assertEqual(annotation["state"], "pending_manual")
            self.assertIsNone(annotation["anchor"]["start"])
            self.assertIn("需人工确认具体引用位置", annotation["status_reason"])

    def test_revision_reanchors_and_never_silently_reuses_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = make_workflow(directory)
            record = {
                "record_id": "r4", "source": "school-synthetic", "revision": 1,
                "observed_at": "2026-11-18T09:00:00Z",
                "text": "开头铺垫。观察是写作的基础。细致观察能够积累素材。",
                "annotation_proposals": [
                    {
                        "category": "citation", "quote": "观察是写作的基础。",
                        "source_ref": "教材第三单元", "rationale": "直接引用",
                        "confidence": 0.95, "client_key": "r4-cite",
                    },
                    {
                        "category": "paraphrase", "quote": "细致观察能够积累素材",
                        "source_ref": "教材第三单元", "rationale": "改写",
                        "confidence": 0.8, "client_key": "r4-para",
                    },
                    {
                        "category": "tool_suggestion", "quote": "已删除的工具建议片段",
                        "rationale": "工具记录缺失", "confidence": 0.3,
                        "client_key": "r4-tool",
                    },
                ],
            }
            workflow.ingest([record])
            citation = next(a for a in workflow.get_record("r4")["annotations"] if a["client_key"] == "r4-cite")
            # 教师先确认一条标注，验证修订后结论会被挂起并回到队列。
            workflow.decide("r4", citation["annotation_id"], "confirm", "teacher-wang")

            revision = [{
                "record_id": "r4", "source": "school-synthetic", "revision": 2,
                "observed_at": "2026-11-18T10:00:00Z",
                # 前面插入文字使唯一匹配片段的偏移发生变化；工具片段被删除；引用句额外重复一次。
                "text": "新增的开头段落。观察是写作的基础。细致观察能够积累素材。另附：观察是写作的基础。",
                "annotation_proposals": [],
            }]
            workflow.ingest(revision)
            after = workflow.get_record("r4")
            self.assertEqual(after["revision"], 2)

            # 唯一匹配的改写片段：偏移必须随新文本迁移，不能沿用旧位置。
            moved = next(a for a in after["annotations"] if a["client_key"] == "r4-para")
            new_span = (moved["anchor"]["original_start"], moved["anchor"]["original_end"])
            self.assertEqual(after["original_text"][new_span[0]:new_span[1]], "细致观察能够积累素材")
            self.assertEqual(moved["state"], "reanchored")
            self.assertIn("唯一匹配重新锚定", moved["status_reason"])
            self.assertTrue(any("重新锚定" in event["message"] for event in moved["events"]))

            # 出现两处候选的引用：标记歧义失效，位置留空，教师原结论挂起。
            ambiguous = next(a for a in after["annotations"] if a["client_key"] == "r4-cite")
            self.assertEqual(ambiguous["state"], "stale_ambiguous")
            self.assertIsNone(ambiguous["anchor"].get("original_start"))
            self.assertIsNone(ambiguous["anchor"].get("start"))

            # 修订中找不到的片段标记失效，旧偏移被清空而不是沿用。
            stale = next(a for a in after["annotations"] if a["client_key"] == "r4-tool")
            self.assertEqual(stale["state"], "stale_not_found")
            self.assertIsNone(stale["anchor"].get("original_start"))

    def test_teacher_decisions_drive_queue_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = make_workflow(directory)
            workflow.ingest(FIXTURE)
            queue_before = workflow.review_queue()
            self.assertTrue(queue_before)
            self.assertTrue(all(item["record_id"] and item["annotation_id"] for item in queue_before))
            first = queue_before[0]
            # 队列项可回溯：span 能切回原文，并携带处理依据与事件。
            self.assertTrue(first["events"])
            if first["span"]:
                record = workflow.get_record(first["record_id"])
                self.assertEqual(
                    record["original_text"][first["span"]["start"]:first["span"]["end"]],
                    first["located_quote"],
                )

            workflow.decide(first["record_id"], first["annotation_id"], "confirm", "teacher-wang", "引用成立")
            queue_after = workflow.review_queue()
            self.assertNotIn(
                (first["record_id"], first["annotation_id"]),
                {(item["record_id"], item["annotation_id"]) for item in queue_after},
            )
            summary = next(
                row for row in workflow.summaries() if row["record_id"] == first["record_id"]
            )
            self.assertEqual(summary["teacher_confirmed"], 1)
            self.assertIn("引用", summary["confirmed_categories"])
            self.assertEqual(
                summary["manual_review_pending"],
                sum(1 for a in workflow.get_record(first["record_id"])["annotations"]
                    if a["state"] not in ("confirmed", "rejected")),
            )

            # 重开后标注回到队列。
            workflow.decide(first["record_id"], first["annotation_id"], "reopen", "teacher-wang")
            self.assertIn(
                (first["record_id"], first["annotation_id"]),
                {(item["record_id"], item["annotation_id"]) for item in workflow.review_queue()},
            )

    def test_audit_trail_returns_original_text_and_basis(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = make_workflow(directory)
            workflow.ingest(FIXTURE)
            record_id = "homework-authorship-review-lesson-b"
            annotation_id = workflow.get_record(record_id)["annotations"][0]["annotation_id"]
            workflow.decide(record_id, annotation_id, "reject", "teacher-li", "依据不充分")
            trail = workflow.audit_trail(record_id, annotation_id)
            self.assertEqual(trail["record_id"], record_id)
            self.assertIn("教师复核作业片段", trail["original_text"])
            self.assertEqual(trail["annotation"]["state"], "rejected")
            self.assertTrue(any("teacher-li" in event["message"] for event in trail["annotation"]["events"]))

    def test_same_revision_ingest_is_idempotent_for_proposals(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = make_workflow(directory)
            workflow.ingest(FIXTURE)
            count_first = len(workflow.get_record("homework-authorship-review-lesson-a")["annotations"])
            workflow.ingest(FIXTURE)
            count_second = len(workflow.get_record("homework-authorship-review-lesson-a")["annotations"])
            self.assertEqual(count_first, count_second)
            # 历史教师决策在同修订重新导入后仍然保留。
            annotation_id = workflow.get_record("homework-authorship-review-lesson-b")["annotations"][0]["annotation_id"]
            workflow.decide("homework-authorship-review-lesson-b", annotation_id, "confirm", "teacher-li")
            workflow.ingest(FIXTURE)
            self.assertEqual(
                workflow.get_record("homework-authorship-review-lesson-b")["annotations"][0]["state"],
                "confirmed",
            )

    def test_stale_annotation_cannot_be_confirmed_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = make_workflow(directory)
            workflow.ingest(FIXTURE)
            workflow.ingest(REVISION2)
            stale = next(
                a for a in workflow.get_record("homework-authorship-review-lesson-a")["annotations"]
                if a["state"] == "stale_not_found"
            )
            with self.assertRaises(ValueError):
                workflow.decide(
                    "homework-authorship-review-lesson-a",
                    stale["annotation_id"], "confirm", "teacher-wang",
                )

    def test_annotation_round_trip_through_repository(self):
        annotation = Annotation(
            annotation_id="x#ann-0001", record_id="x", category="citation",
            quote="原文", rationale="依据", state="pending_manual",
        )
        restored = Annotation.from_dict(annotation.to_dict())
        self.assertEqual(restored.annotation_id, "x#ann-0001")


if __name__ == "__main__":
    unittest.main()
