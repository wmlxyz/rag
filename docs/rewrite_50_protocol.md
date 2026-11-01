# 50 会话 Query Rewrite 对照实验

## 固定方案

从先前预先随机选取的 200 个 QReCC 测试会话中，保留有历史、非空参考答案、且在当前候选池中至少匹配到一个参考 passage 的会话；排除前两次 Query Rewrite 实验看过的会话，按固定种子 `20261028` 从 68 个可用会话中随机选 50 个，每个会话选文件顺序中首个合格追问。记录保存在 `data/qrecc/rewrite-eval-50.json`，并保存来源及排除数据的哈希。

知识库为原先的 1149 个 passage、3009 个 800/120 字符 chunk 的候选池：测试标注匹配的 149 个片段和前 10 万行中抽出的 1000 个非目标干扰片段。固定索引 `832ca220f6955497f36abdbf`、HNSW `ef_search=64`、Top-K=5、相似度阈值 0、证据预算 4000 字符；不使用 Reranker、去重或向量缓存。HNSW 在此前 50 个查询的 Top-20 对照中，平均与精确结果集合重合率为 99.6%；该数据是 Top-20 的统计，不能直接解释为本实验 Top-5 的准确率。

三组仅改变 Query 构造：`current`、`history`、`rewrite`。历史来自 QReCC 参考数据，包括此前人工改写问题和参考答案，最多最近 8 条消息。`rewrite` 在有历史时调用生成模型生成独立问题；所有方法回答时使用同一参考历史，避免用不同策略生成的历史干扰对照。生成模型 `qwen-plus`，Embedding `text-embedding-v4`，温度 0；API 服务不保证完全确定。三组依次运行，耗时受网络和服务时间波动影响。

每题分别记录召回、实际证据、回答、引用编号、用量、耗时和错误。有效标注样本上统计 Hit@5 和 MRR@5；参考答案非空样本上计算词面 Token F1 和完全匹配。检索相关性为来源级代理，不证明召回的 chunk 包含答案；编号有效率不证明引用支持回答。报告显式列出成功和失败及各指标分母。按相同问题配对计算策略差异，并以会话为单位 bootstrap 给出 95% 区间；该区间只能描述此受限候选池下的抽样波动，不能修复数据选择偏差。

本实验是研究用诊断，不是完整 QReCC 开放域的无偏评测：正例依赖测试标签进入候选池、干扰文档从语料前缀抽取，且仅选择了可映射证据的会话。部分候选会话可能出现在此前 Reranker 检索诊断中，已明确排除的是此前 Query Rewrite 实验的会话。不能据此声称可推广到全部对话或证明因果效果。

## 运行与恢复

实验需配置 README 中的 Qwen 生成和 Embedding 环境变量，并使用已安装 FAISS 的项目虚拟环境。`run_rewrite_study.py` 每完成一题写入并同步 JSONL 检查点；异常中断后，对相同命令追加 `--resume` 可以继续。恢复前会核对数据、配置、索引版本、ANN 文件摘要、模型地址及源码哈希；发生变化则拒绝继续。失败的单题会记录异常类型，不会自动重复付费调用，汇总中应单独检查失败率。

完成后的结果保存在 `experiments/results/rewrite-50-v1.json`，逐题检查点是其 `.jsonl` 文件。统计命令：

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/analyze_rewrite_study.py experiments/results/rewrite-50-v1.json experiments/results/rewrite-50-analysis.json
```

统计脚本计算会话级配对 bootstrap、报告模型返回的生成及改写 Token、失败和延迟；Embedding API 用量及价格没有记录，不能推算完整费用。不要把含对话与回答的结果直接公开，需先核对数据许可与隐私。
