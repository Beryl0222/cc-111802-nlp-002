from .annotations import (
    Annotation,
    apply_decision,
    build_annotation,
    reanchor_annotation,
    record_summary,
    review_queue,
)
from .records import TextRecord
from .text import annotate, normalize_text


class TextWorkflow:
    def __init__(self, repository):
        self.repository = repository

    def ingest(self, values):
        if not isinstance(values, list):
            raise TypeError("导入内容必须是记录列表")
        incoming = [TextRecord.from_dict(value).to_dict() for value in values]
        raw_values = values
        existing = self.repository.read()
        current = {record["record_id"]: record for record in existing}

        for index, record in enumerate(incoming):
            record_id = record["record_id"]
            previous = current.get(record_id)
            if previous is not None and record["revision"] < previous["revision"]:
                # 旧修订号一律忽略，避免覆盖新文本与新标注。
                continue

            record["original_text"] = record["text"]
            record["normalized_text"] = normalize_text(record["text"])
            record["text"] = record["normalized_text"]
            record["analysis"] = annotate(record)

            if previous is None or record["revision"] > previous["revision"]:
                # 新修订：保留全部历史标注，但对每条标注重新锚定；
                # 旧偏移只有在找到唯一匹配时才迁移，并留下事件记录。
                carried = [Annotation.from_dict(item).to_dict() for item in previous.get("annotations", [])] if previous else []
                for item in carried:
                    annotation = Annotation.from_dict(item)
                    reanchor_annotation(annotation, record["original_text"])
                    item.clear()
                    item.update(annotation.to_dict())
                record["annotations"] = carried
            else:
                # 同一修订重新导入：标注与处理记录保持稳定，只追加新建议。
                record["annotations"] = [
                    Annotation.from_dict(item).to_dict() for item in previous.get("annotations", [])
                ]

            self._add_proposals(record, raw_values[index].get("annotation_proposals", []))

            current[record_id] = record

        result = sorted(current.values(), key=lambda row: row["record_id"])
        self.repository.replace(result)
        return {
            "received": len(incoming),
            "stored": len(result),
            "annotations": sum(len(record.get("annotations", [])) for record in result),
        }

    def _add_proposals(self, record, proposals):
        if proposals is None:
            return
        if not isinstance(proposals, list):
            raise TypeError("annotation_proposals 必须是列表")
        existing = record.setdefault("annotations", [])
        existing_keys = {
            (item.get("client_key") or "")
            for item in existing
            if item.get("client_key")
        }
        existing_fingerprints = {
            (item["category"], item["quote"], item["source_ref"], item["rationale"])
            for item in existing
        }
        for proposal in proposals:
            if not isinstance(proposal, dict):
                raise TypeError("标注建议必须是对象")
            client_key = str(proposal.get("client_key", "")).strip()
            fingerprint = (
                str(proposal.get("category", "")).strip(),
                str(proposal.get("quote", "")),
                str(proposal.get("source_ref", "")).strip(),
                str(proposal.get("rationale", "")).strip(),
            )
            # 同修订重复导入同一建议时保持幂等；不同依据或不同位置的引用各自独立。
            if (client_key and client_key in existing_keys) or fingerprint in existing_fingerprints:
                continue
            annotation_id = f"{record['record_id']}#ann-{len(existing) + 1:04d}"
            proposal_with_text = dict(proposal)
            proposal_with_text["text"] = record["original_text"]
            annotation = build_annotation(proposal_with_text, record["record_id"], annotation_id)
            existing.append(annotation.to_dict())
            if client_key:
                existing_keys.add(client_key)
            existing_fingerprints.add(fingerprint)

    def list_records(self, source=None):
        records = self.repository.read()
        if source:
            records = [record for record in records if record["source"] == source]
        return records

    def get_record(self, record_id):
        for record in self.repository.read():
            if record["record_id"] == record_id:
                return record
        raise KeyError(f"未找到记录：{record_id}")

    def decide(self, record_id, annotation_id, decision, reviewer, note=""):
        records = self.repository.read()
        target_record = None
        target_annotation = None
        for record in records:
            if record["record_id"] != record_id:
                continue
            for item in record.get("annotations", []):
                if item["annotation_id"] == annotation_id:
                    target_record = record
                    target_annotation = item
                    break
        if target_record is None:
            raise KeyError(f"未找到标注：{record_id} / {annotation_id}")
        annotation = Annotation.from_dict(target_annotation)
        apply_decision(annotation, decision, reviewer, note)
        target_annotation.clear()
        target_annotation.update(annotation.to_dict())
        self.repository.replace(sorted(records, key=lambda row: row["record_id"]))
        return target_annotation

    def review_queue(self):
        return review_queue(self.repository.read())

    def summaries(self):
        return [record_summary(record) for record in sorted(self.repository.read(), key=lambda row: row["record_id"])]

    def audit_trail(self, record_id, annotation_id):
        """返回某条标注的完整处理依据，便于从队列回到原文与决策过程。"""
        record = self.get_record(record_id)
        for item in record.get("annotations", []):
            if item["annotation_id"] == annotation_id:
                anchor = item.get("anchor", {})
                start = anchor.get("original_start")
                end = anchor.get("original_end")
                return {
                    "record_id": record_id,
                    "revision": record["revision"],
                    "original_text": record["original_text"],
                    "annotation": item,
                    "located_quote": record["original_text"][start:end] if start is not None and end is not None else "",
                }
        raise KeyError(f"未找到标注：{record_id} / {annotation_id}")
