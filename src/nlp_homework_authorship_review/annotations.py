"""作业表达来源标注：标注模型、位置锚定、状态流转与复核摘要。

设计原则：
- 标注只陈述来源依据与定位结果，系统不产出“代写/违规”结论；
- 资料不足（缺出处、低置信度、位置歧义、锚定失效等）一律进入人工判断状态；
- 每条标注独立编号并保留完整事件日志，同一原文被多次引用时互不覆盖；
- 文本修订后标注必须重新锚定，旧偏移绝不静默迁移到新字符位置。
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .text import locate_quote, normalize_text, tokenize


CATEGORIES = ("citation", "paraphrase", "student_note", "tool_suggestion")
CATEGORY_LABELS = {
    "citation": "引用",
    "paraphrase": "改写",
    "student_note": "学生补充说明",
    "tool_suggestion": "辅助工具建议",
}

# 待人工处理的状态集合；任何证据或锚定问题都回到该队列。
MANUAL_STATES = (
    "pending_manual",
    "reanchored",
    "stale_ambiguous",
    "stale_not_found",
    "reopened",
)
# 教师已做出确认/驳回，不再出现在默认队列中。
RESOLVED_STATES = ("confirmed", "rejected")
VALID_STATES = MANUAL_STATES + RESOLVED_STATES

# 低于该置信度的标注必须留待人工判断。
MIN_AUTOMATIC_CONFIDENCE = 0.6


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _event(message):
    return {"at": now_iso(), "message": message}


def _require_choice(value, choices, name):
    value = str(value or "").strip()
    if value not in choices:
        raise ValueError(f"{name} 必须是以下之一：{', '.join(choices)}")
    return value


@dataclass
class Annotation:
    annotation_id: str
    record_id: str
    category: str
    quote: str
    rationale: str
    source_ref: str = ""
    confidence: float = 0.0
    state: str = "pending_manual"
    status_reason: str = ""
    anchor: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    decided_by: str = ""
    decided_at: str = ""
    client_key: str = ""

    @classmethod
    def from_dict(cls, value):
        annotation = cls(
            annotation_id=str(value.get("annotation_id", "")).strip(),
            record_id=str(value.get("record_id", "")).strip(),
            category=str(value.get("category", "")).strip(),
            quote=str(value.get("quote", "")),
            rationale=str(value.get("rationale", "")).strip(),
            source_ref=str(value.get("source_ref", "")).strip(),
            confidence=float(value.get("confidence", 0.0) or 0.0),
            state=str(value.get("state", "")).strip(),
            status_reason=str(value.get("status_reason", "")).strip(),
            anchor=dict(value.get("anchor") or {}),
            events=list(value.get("events") or []),
            decided_by=str(value.get("decided_by", "")).strip(),
            decided_at=str(value.get("decided_at", "")).strip(),
            client_key=str(value.get("client_key", "")).strip(),
        )
        if not annotation.annotation_id or not annotation.record_id:
            raise ValueError("标注缺少编号或所属记录")
        if annotation.category not in CATEGORIES:
            raise ValueError("标注类别无效")
        if not annotation.quote.strip():
            raise ValueError("标注缺少引用原文")
        if not annotation.rationale:
            raise ValueError("标注缺少处理依据")
        if not 0.0 <= annotation.confidence <= 1.0:
            raise ValueError("置信度必须位于 0 到 1 之间")
        if annotation.state not in VALID_STATES:
            raise ValueError("标注状态无效")
        return annotation

    def to_dict(self):
        return asdict(self)


def _evidence_reasons(category, source_ref, confidence):
    reasons = []
    if category in ("citation", "paraphrase", "tool_suggestion") and not source_ref:
        reasons.append("未提供出处或工具名称，来源依据不足")
    if confidence < MIN_AUTOMATIC_CONFIDENCE:
        reasons.append(f"置信度 {confidence:.2f} 低于自动确认门槛 {MIN_AUTOMATIC_CONFIDENCE:.1f}")
    return reasons


def build_annotation(proposal, record_id, annotation_id):
    """根据导入的标注建议生成标注，并完成首次位置锚定。"""
    if not isinstance(proposal, dict):
        raise TypeError("标注建议必须是对象")
    category = _require_choice(proposal.get("category"), CATEGORIES, "标注类别")
    quote = str(proposal.get("quote", ""))
    rationale = str(proposal.get("rationale", "")).strip()
    if not quote.strip():
        raise ValueError(f"标注 {annotation_id} 缺少引用原文")
    if not rationale:
        raise ValueError(f"标注 {annotation_id} 缺少处理依据")
    source_ref = str(proposal.get("source_ref", "")).strip()
    try:
        confidence = float(proposal.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError) as error:
        raise ValueError("置信度必须是 0 到 1 的数字") from error
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("置信度必须位于 0 到 1 之间")
    client_key = str(proposal.get("client_key", "")).strip()

    location = locate_quote(proposal.get("text", ""), quote) if str(proposal.get("text", "")) else {"match_kind": "none", "candidates": []}
    anchor = {
        "normalized_quote": normalize_text(quote),
        "token_signature": tokenize(quote),
        "context_quote": str(proposal.get("context_quote", "")).strip(),
        "match_kind": location["match_kind"],
        "start": None,
        "end": None,
        "token_start": None,
        "token_end": None,
        "original_start": None,
        "original_end": None,
    }

    reasons = _evidence_reasons(category, source_ref, confidence)
    declared_span = proposal.get("span")
    if declared_span is not None and (
        not isinstance(declared_span, dict)
        or not isinstance(declared_span.get("start"), int)
        or not isinstance(declared_span.get("end"), int)
        or declared_span["end"] <= declared_span["start"]
    ):
        raise ValueError("span 必须包含整数 start 且 start < end")

    if location["match_kind"] == "none":
        reasons.append("当前文本中未找到该片段，无法建立位置锚定")
    elif len(location["candidates"]) > 1:
        anchor["candidate_count"] = len(location["candidates"])
        reasons.append(f"片段在文本中出现 {len(location['candidates'])} 处，需人工确认具体引用位置")
    else:
        candidate = location["candidates"][0]
        if declared_span is not None and (
            declared_span["start"] != candidate["original_start"]
            or declared_span["end"] != candidate["original_end"]
        ):
            reasons.append("申报的原文区间与实际匹配位置不一致，以匹配位置为准并留待人工核对")
        anchor.update(candidate)

    if not reasons:
        state = "pending_manual"
        status_reason = "定位与依据齐全，等待教师复核"
    else:
        state = "pending_manual"
        status_reason = "；".join(reasons)

    return Annotation(
        annotation_id=annotation_id,
        record_id=record_id,
        category=category,
        quote=quote,
        rationale=rationale,
        source_ref=source_ref,
        confidence=confidence,
        state=state,
        status_reason=status_reason,
        anchor=anchor,
        events=[_event(f"标注创建：{CATEGORY_LABELS[category]}，{status_reason}")],
        client_key=client_key,
    )


def reanchor_annotation(annotation, text):
    """文本修订后重新锚定既有标注，返回 (是否更新, 事件消息或 None)。

    - 唯一匹配：把偏移迁移到新位置并记录“重新锚定”，原决策保留但要求教师复核；
    - 多处匹配：标记歧义失效，不选择位置；
    - 找不到：标记失效，保留历史依据；
    - 无原文片段的标注：同样置为失效，等待人工补录。
    旧偏移永远不会在没有匹配证据的情况下直接照搬到新文本。
    """
    location = locate_quote(text, annotation.quote)
    old_anchor = dict(annotation.anchor)
    annotation.anchor["match_kind"] = location["match_kind"]
    for key in ("start", "end", "token_start", "token_end", "original_start", "original_end", "candidate_count"):
        annotation.anchor.pop(key, None)

    if location["match_kind"] == "none":
        annotation.state = "stale_not_found"
        annotation.status_reason = "文本修订后未找到该片段，旧位置已失效，需人工处理"
        annotation.events.append(_event(annotation.status_reason))
        return True, annotation.status_reason
    if len(location["candidates"]) > 1:
        annotation.anchor["candidate_count"] = len(location["candidates"])
        annotation.state = "stale_ambiguous"
        annotation.status_reason = f"文本修订后该片段出现 {len(location['candidates'])} 处，位置歧义，需人工选择"
        annotation.events.append(_event(annotation.status_reason))
        return True, annotation.status_reason

    candidate = location["candidates"][0]
    annotation.anchor.update(candidate)
    same_position = (
        old_anchor.get("original_start") == candidate["original_start"]
        and old_anchor.get("original_end") == candidate["original_end"]
    )
    if same_position and annotation.state != "reanchored":
        return False, None
    previous = annotation.state
    annotation.state = "reanchored"
    annotation.status_reason = "文本修订后已按唯一匹配重新锚定，请教师重新确认"
    if previous in RESOLVED_STATES:
        annotation.status_reason += f"（原结论 {previous} 已挂起）"
        annotation.decided_by = ""
        annotation.decided_at = ""
    annotation.events.append(_event(annotation.status_reason))
    return True, annotation.status_reason


def apply_decision(annotation, decision, reviewer, note=""):
    """应用教师确认/驳回/重开，并写入处理依据。"""
    decision = _require_choice(decision, ("confirm", "reject", "reopen"), "复核动作")
    reviewer = str(reviewer or "").strip()
    if not reviewer:
        raise ValueError("复核动作必须记录教师标识")
    if decision == "confirm":
        if annotation.state in ("stale_ambiguous", "stale_not_found"):
            raise ValueError("位置失效的标注需先重新定位，不能直接确认")
        annotation.state = "confirmed"
        annotation.status_reason = "教师确认该标注的来源与位置"
    elif decision == "reject":
        annotation.state = "rejected"
        annotation.status_reason = "教师驳回该标注"
    else:
        annotation.state = "reopened"
        annotation.status_reason = "教师要求重新人工判断"
    annotation.decided_by = reviewer
    annotation.decided_at = now_iso()
    message = f"教师 {reviewer} 执行 {decision}：{annotation.status_reason}"
    if note.strip():
        message += f"；备注：{note.strip()}"
    annotation.events.append(_event(message))
    return annotation


def review_queue(records):
    """组装人工复核队列，按记录号与标注号稳定排序，保留回溯所需全部线索。"""
    queue = []
    for record in sorted(records, key=lambda row: row["record_id"]):
        for annotation in record.get("annotations", []):
            if annotation["state"] not in MANUAL_STATES:
                continue
            anchor = annotation.get("anchor", {})
            start = anchor.get("original_start")
            end = anchor.get("original_end")
            if start is not None and end is not None:
                original_text = record.get("original_text", record["text"])
                located_quote = original_text[start:end]
            else:
                located_quote = ""
            queue.append({
                "record_id": record["record_id"],
                "revision": record["revision"],
                "annotation_id": annotation["annotation_id"],
                "category": annotation["category"],
                "category_label": CATEGORY_LABELS[annotation["category"]],
                "state": annotation["state"],
                "status_reason": annotation["status_reason"],
                "quote": annotation["quote"],
                "located_quote": located_quote,
                "source_ref": annotation.get("source_ref", ""),
                "confidence": annotation.get("confidence", 0.0),
                "rationale": annotation["rationale"],
                "span": None if start is None else {"start": start, "end": end},
                "events": list(annotation.get("events", [])),
            })
    return queue


def record_summary(record):
    """生成单条作业摘要；教师确认结果直接影响各类计数与待办数量。"""
    annotations = record.get("annotations", [])
    counts = {label: 0 for label in CATEGORIES}
    state_counts = {state: 0 for state in VALID_STATES}
    for annotation in annotations:
        counts[annotation["category"]] += 1
        state_counts[annotation["state"]] += 1
    manual_count = sum(state_counts[state] for state in MANUAL_STATES)
    confirmed = [annotation for annotation in annotations if annotation["state"] == "confirmed"]
    unresolved_quotes = [
        annotation["quote"]
        for annotation in annotations
        if annotation["state"] in ("stale_ambiguous", "stale_not_found", "reopened", "pending_manual", "reanchored")
    ]
    return {
        "record_id": record["record_id"],
        "source": record["source"],
        "revision": record["revision"],
        "annotation_total": len(annotations),
        "category_counts": {CATEGORY_LABELS[key]: value for key, value in counts.items()},
        "state_counts": state_counts,
        "manual_review_pending": manual_count,
        "teacher_confirmed": len(confirmed),
        "confirmed_categories": [CATEGORY_LABELS[annotation["category"]] for annotation in confirmed],
        "unresolved_quotes": unresolved_quotes,
        "verdict": "资料不足或待复核项未处理完，系统不作代写/违规结论" if manual_count else "教师已完成全部标注复核（确认或驳回），系统不作自动违规结论",
    }
