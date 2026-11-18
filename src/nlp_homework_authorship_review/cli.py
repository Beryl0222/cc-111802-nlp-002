import argparse
import json
from pathlib import Path

from .repository import RecordRepository
from .workflow import TextWorkflow


def build_parser():
    parser = argparse.ArgumentParser(description="本地教研文本处理工具")
    parser.add_argument("action", choices=("ingest", "list", "queue", "decide", "summary", "audit"))
    parser.add_argument("path", nargs="?", default="fixtures/records.json")
    parser.add_argument("--state", default="var/records.json")
    parser.add_argument("--record", help="decide/audit 使用的作业记录号")
    parser.add_argument("--annotation", help="decide/audit 使用的标注号")
    parser.add_argument("--decision", choices=("confirm", "reject", "reopen"), help="教师复核动作")
    parser.add_argument("--reviewer", help="执行复核的教师标识")
    parser.add_argument("--note", default="", help="复核备注")
    return parser


def main():
    args = build_parser().parse_args()
    workflow = TextWorkflow(RecordRepository(args.state))
    if args.action == "ingest":
        result = workflow.ingest(json.loads(Path(args.path).read_text(encoding="utf-8")))
    elif args.action == "list":
        result = workflow.list_records()
    elif args.action == "queue":
        result = workflow.review_queue()
    elif args.action == "summary":
        result = workflow.summaries()
    elif args.action == "decide":
        if not (args.record and args.annotation and args.decision and args.reviewer):
            raise SystemExit("decide 需要 --record、--annotation、--decision、--reviewer 参数")
        result = workflow.decide(args.record, args.annotation, args.decision, args.reviewer, args.note)
    else:
        if not (args.record and args.annotation):
            raise SystemExit("audit 需要 --record 与 --annotation 参数")
        result = workflow.audit_trail(args.record, args.annotation)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
