# 回答与引用可靠性

## 当前行为

检索结果为空（包括未达到 `--min-score`）时不调用生成模型，返回“未检索到符合条件的资料”。有证据时，提示中列明允许的证据编号。生成后检查数字引用，包括 `[1]`、`[1, 2]`；不存在的编号和数字开头但格式无法解析的方括号引用会标记为异常。

默认 `--citation-policy repair`：回答没有有效编号、含无效编号或格式异常时，带原问题、历史、证据和初稿最多请求一次修复。修复后仍不通过或修复调用失败，返回明确的无可靠引用答复，状态分别为 `abstained_after_invalid_repair` 或 `abstained_after_repair_error`，不会将原始无效回答写入会话历史。修复可能增加一次模型调用及费用。

`--citation-policy warn` 保留原始回答，仅报告检查结果，适合研究对照。HTTP `/v1/ask` 也可传入 `citation_policy`；默认是 `repair`。`citation_check` 是最终编号检查，`initial_citation_check` 为修复前检查，`citation_repair_usage` 记录服务返回的修复用量；`answer_status` 区分 `answered`、`no_evidence` 和 `citation_unresolved`。

**编号检查不判断事实是否正确、句子是否被证据支持、是否每个论断都有引用。** `evidence_support_status` 固定为 `not_verified`，不能解读为通过。即使给出的编号存在，修复模型也可能保留不受支持的内容；遇到模型自身的无引用拒答，默认策略可能仍尝试一次修复并返回通用提示。规则仅覆盖约定的方括号数字引用，不保证理解所有自然语言引用格式。

## 人工核查示例

`examples/evaluation/citation_review.json` 有六个手工构造的例子，覆盖有效编号、无效编号、事实不受支持、部分支持和适当拒答。人工核查时按以下顺序记录：

1. 问题和回答的每个事实性陈述是什么？
2. 引用编号是否存在，引用的实际片段是否足以支持对应陈述？
3. 有缺失信息时是否明确说明？输出 `supported`、`partially_supported`、`unsupported` 或 `appropriate_abstention`，并记录理由。

例子是验证流程的合成数据，不能用作系统正确率。代码测试特意验证“编号有效但事实不受支持”仍会通过语法检查，以防把两者混淆。

## 阈值示例

`examples/evaluation/fruit_threshold.json` 是针对 `examples/knowledge/fruit.md` 人工标注的 8 条开发题和 4 条验证题。`scripts/calibrate_threshold.py` 使用 Top-1 余弦分数在开发集选择平衡准确率最高的阈值，平分时选择更高阈值，再报告验证集混淆矩阵。一次真实 `text-embedding-v4` 调用得到阈值约 0.6469，开发集 8/8、验证集 4/4；结果保存在 `experiments/results/fruit-threshold-calibration.json`。

该阈值只对示例文件、模型与样本成立；样本极小，正负问题也较简单，不能推广为 QReCC 或其他知识库的默认阈值。相似度衡量相关性，不能证明答案受到证据支持；更换模型、索引或语料需要重新收集独立标注并校准。现有默认 `min_score=0` 没有经过通用校准。

## 对历史实验的影响

本次修改了生成提示并新增默认修复调用，输出内容和成本可能与历史实验不同。旧实验结果保留，不能与新策略生成的分数直接混为同一设置；新实验应记录 `citation_policy` 和代码版本，并将修复 Token 计入成本。
