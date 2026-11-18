# 作业表达来源复核台

作业表达来源复核台用于处理本地合成教研文本，保存来源版本、文本片段、来源标注和教师处理记录，供校内复核使用。系统只整理可复核的证据与位置，不自动给出代写或违规结论；资料不足时一律留待人工判断。

## 环境

需要 Python 3.11 或更新版本。应用运行不依赖第三方包；运行测试和构建前，安装固定版本的开发工具：`python -m pip install -r requirements-dev.txt`。

## 标注与锚定规则

- 每条记录同时保留 `original_text`（原始排版）与 `normalized_text`（规范化文本）；标注锚点同时记录两套字符偏移，可从任何队列结果切回原文片段。
- 标注分四类：`citation`（引用）、`paraphrase`（改写）、`student_note`（学生补充说明）、`tool_suggestion`（辅助工具建议）。
- 每条标注有独立编号和事件日志。同一来源文字被多次引用时，只要处理依据不同就各自独立，不会互相覆盖；同修订重复导入同一建议（相同 `client_key` 或相同内容指纹）保持幂等。
- 定位先做规范化整串匹配，再退化为词元序列匹配。片段出现多处候选时不自动选择，标注进入 `pending_manual` 等待教师确认位置。
- 新修订导入时，历史标注全部重新锚定：唯一匹配才迁移偏移并记为 `reanchored`（原教师结论挂起）；多处候选记为 `stale_ambiguous`；找不到记为 `stale_not_found`。旧偏移永远不会静默落到新字符位置，失效标注在重新定位前不能被确认。
- 缺少出处、置信度低于 0.6、位置歧义或锚定失效的标注都进入人工队列，摘要只陈述待复核状态，不作违规判定。

## 运行

数据保存在 `var/records.json`。

```bash
# 导入演练记录（含 annotation_proposals 标注建议）
PYTHONPATH=src python -m nlp_homework_authorship_review.cli ingest fixtures/records.json
# 导入修订版，观察标注重新锚定
PYTHONPATH=src python -m nlp_homework_authorship_review.cli ingest fixtures/records_revision2.json
# 查看记录与标注
PYTHONPATH=src python -m nlp_homework_authorship_review.cli list
# 人工复核队列（含原文位置、依据和事件日志）
PYTHONPATH=src python -m nlp_homework_authorship_review.cli queue
# 教师决策：confirm / reject / reopen
PYTHONPATH=src python -m nlp_homework_authorship_review.cli decide \
  --record <记录号> --annotation <标注号> \
  --decision confirm --reviewer teacher-li --note "引用成立"
# 作业摘要，计数随教师确认/驳回更新
PYTHONPATH=src python -m nlp_homework_authorship_review.cli summary
# 回到某条标注对应的原文与完整处理依据
PYTHONPATH=src python -m nlp_homework_authorship_review.cli audit \
  --record <记录号> --annotation <标注号>
```

## 测试

执行 `python -m pytest -q` 检查文本规范化、片段定位、标注锚定、修订重锚、教师决策、复核队列、摘要与本地持久化。
