# 阶段四：配对评估与数据审计

新增 `scripts/evaluate_pair.py`，读取同一数据集和两份单策略评估结果，核对数据哈希、记录 ID 与会话 ID，报告数据覆盖、未配对记录、错误和指标缺失。对 Hit、MRR、上下文来源命中、Token F1、引用格式通过率及耗时，按记录配对计算差值、变化方向和按会话重采样的 95% percentile bootstrap 区间；每个指标独立排除失败或缺失的配对。固定种子使重采样可复现。少于两个会话时不输出区间。

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/evaluate_pair.py data/qrecc/retrieval-stage3-answer-8.json experiments/results/retrieval-stage3-answer-dense-8.json experiments/results/retrieval-stage3-answer-hybrid-8.json experiments/results/my-comparison.json
```

输出必须为新文件。比较前应手动检查报告中的两个 manifest，确认除研究因素外其余设置相同；工具不自动证明配置兼容或检查数据泄漏。对于原始数据相同但被修改过的文件，脚本拒绝哈希不符的运行结果。报告保留输入文件与脚本哈希、记录级变化和排除数；包含已有结果的原始 manifest，请勿在公开前忽略数据隐私。

## 示例

以阶段三 8 条问答的混合组减向量组计算，MRR 差值 -0.0167，按会话 bootstrap 区间 [-0.0500, 0]；Token F1 差值 +0.0255，区间约 [-0.0047, 0.0703]。数据仅 8 个会话，且按覆盖选样；这些区间不能消除选择偏差或反映远程模型变化。两组单独生成，F1 差异也不能直接归因于检索配置。完整机器可读结果在 `experiments/results/stage4-paired-analysis.json`。

## 未完成事项

尚未构建新的独立保留集，现有候选池依赖测试标注筛选；不能将上述样本称为未见测试集。`scripts/run_rewrite_study.py` 支持指定策略、混合检索和重排，并按记录保存断点；使用相同参数、数据、索引和源码加 `--resume` 恢复中断运行。失败记录视为已完成，不自动重试；通用 `evaluate` 和阶段三脚本尚不能续跑。尚无模型证据支持性评分、自动配置兼容性审计和新实验的完整用量/费用汇总。引用格式、来源命中和词面 F1 均不能证明回答被证据支持；没有人工标注时不报告该真实性指标。阶段四目前完成的是离线配对分析与覆盖审计，以上部分仍需继续开发。

可续跑的实验示例（输出路径必须是新路径）：

```powershell
.\.venv\Scripts\python.exe scripts/run_rewrite_study.py data/qrecc/rewrite-eval-50.json experiments/results/my-study.json --index 832ca220f6955497f36abdbf --strategies history rewrite_checked --retrieval-mode hybrid --rerank --candidate-k 10
# 中断后追加相同参数并加 --resume；最终 JSON 已存在时不可续跑。
```

该脚本的断点固定数据与源码哈希、模型及 ANN 索引版本，但当前未验证 FTS 索引内容哈希；运行期间不要重建词法索引。
