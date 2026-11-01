# 基线评测使用说明

## 已准备数据

`data/qrecc/qrecc-test.json` 来自 [QReCC 官方 Zenodo 版本](https://zenodo.org/records/5760304)，下载文件 MD5 为 `7ad29296a1dfd2f75dafd653a8822269`，与官方发布值一致。文件为 16,451 轮测试数据。`data/qrecc/test-seed42-10.json` 使用种子 42 均匀抽取 10 个完整会话，共 62 轮；其中 54 轮有非空参考答案与 URL。该小样本用于联调，不足以支持正式统计结论。数据保存在被忽略的 `data/` 目录；遵循 QReCC CC BY-SA 3.0 许可和原始语料许可。

官方 Context 包含**此前的人工改写问题**和非空参考答案。导入器逐轮重建并校验它，保留缺失答案导致的非成对历史。所有方法共享这份参考历史，避免不同方法的生成误差混入比较，但这不是完全自回归真实对话，且历史中包含人工改写信息，报告时必须说明。

## 准备其他固定子集

在项目根目录执行：

```powershell
$env:PYTHONPATH = 'src'
python -m rag.cli prepare-qrecc data/qrecc/qrecc-test.json data/qrecc/another-subset.json --split test --conversations 10 --seed 123
```

准备阶段按会话抽样并保留所有轮次，输出包含原文件 SHA-256 和种子。开发集应从训练集按会话选取；测试集不能用于调参。不要把参考答案作为索引文档。

新版准备文件包含 `reference_passages`；旧版文件需要重新运行 `prepare-qrecc`。

## 从 QReCC passages 构建受限候选池

安装流式读取依赖（仅此功能需要）：`python -m pip install -e '.[qrecc]'`。镜像来自第三方 [slupart/qrecc-passages](https://huggingface.co/datasets/slupart/qrecc-passages)，默认固定到提交 `99706cf40ae845ba9730e6a49b6ed67337bd6166`；其与官方压缩包的完整等价性尚未独立验证。执行：

```powershell
python -m rag.cli qrecc-subset data/qrecc/test-seed42-10-v2.json data/qrecc/my-pool --max-scan 10000 --distractors 100 --seed 42
```

该命令依次扫描固定版本的前 `max-scan` 行，保存其中与 QReCC `Passages` 标注 ID 精确匹配的片段，并对其余可用片段使用固定种子的蓄水池抽样选取干扰项。它不把参考答案当成文档。输出目录包含 `documents/`、片段 ID 和文本哈希清单 `manifest.json`、带精确标注映射的 `mapped_dataset.json`。该池使用测试标签选取相关片段，且干扰项来自前缀而非全语料均匀抽样，属于**受限、带选择偏差的候选池**。改变扫描范围会改变实验条件；网络读取可能传输比最终文件大得多的数据。

覆盖率以清单为准。标注不一定存在、前缀中不一定能找到标注，也可能因长度上限被排除；未匹配的样本不计入检索指标。若覆盖率为零，不可做检索效果评估。为了避免浪费费用，应先看 `manifest.json` 的 `coverage`，确认非零后再调用 Embedding 建索引：

```powershell
python -m rag.cli index data/qrecc/my-pool/documents --db data/rag.sqlite3
python -m rag.cli evaluate data/qrecc/my-pool/mapped_dataset.json experiments/results/qrecc-run.json --index '<index_id>' --strategies current history rewrite human
```

也可以通过 `--jsonl` 输入本地逐行 JSON，要求每行含原始 `id` 和 `contents`；来源版本用该文件 SHA-256 记录。命令拒绝覆盖已有输出；中途流式读取失败不会发布完整输出。当前方案适合小规模实验，不能用于高效索引完整语料。

实际联调时，固定随机 10 个会话在前 1 万条中没有找到参考片段；因此进一步扫描前 10 万条并与整个官方测试集的 passage ID 匹配。按覆盖情况筛选 5 个完整会话，再从已下载池中二次抽样，获得 `data/qrecc/qrecc-smoke-pool`：34 轮、106 个片段，其中 6 个匹配参考 ID、100 个为干扰项，11 轮有匹配标注。这是**先看覆盖再选会话**的冒烟数据，不能当作无偏测试集或正式基线。中间池 `data/qrecc/passage-fulltest-prefix-100000` 的清单记录来源和覆盖率。复现二次选择：

```powershell
python -m rag.cli qrecc-covered-sample data/qrecc/passage-fulltest-prefix-100000/mapped_dataset.json data/qrecc/covered-new.json --conversations 5 --seed 42
python -m rag.cli qrecc-subset data/qrecc/covered-new.json data/qrecc/pool-new --pool data/qrecc/passage-fulltest-prefix-100000 --max-scan 10000 --distractors 100 --seed 42
```

二次抽样池的来源标识是上游 manifest 的 SHA-256；上游 manifest 保存了第三方镜像固定版本及各文本哈希。`--pool` 会检查所有输入文本哈希。要获得有代表性的实验，需要预先确定问题样本、扩展或完整扫描语料并提高覆盖率；不要用上述覆盖筛选样本声称总体检索性能。

## 证据语料及标签

QReCC 正式检索语料为大规模 passages，压缩包约 27.5 GB，当前还没有下载或索引。可以先准备来源明确的受限候选池，必须记录选择规则、文档版本、非相关文档、URL 和覆盖率；若使用测试问题的答案来源选文档，应明确标注为答案来源构成的受限候选池，不能声称等同完整开放域检索。

先将候选文档构建为索引，并创建 UTF-8 `mapping.json`，将索引中相对文件路径映射到对应的原始参考 URL。单个文档可对应多个精确 URL：

```json
{"example.txt": ["https://example.org/page", "https://example.org/page#section"]}
```

```powershell
python -m rag.cli map-qrecc-sources data/qrecc/test-seed42-10.json mapping.json data/qrecc/mapped.json --index '<index_id>'
```

映射按 URL 字符串精确匹配，不自动合并锚点、重定向或 URL 规范化。映射工具验证文件存在于该索引并记录映射文件哈希和映射覆盖率。参考 URL 对应文档仅是**文档级代理标签**，不保证检索片段含答案；映射不到的记录不参与检索指标分母。对缺失、版本差异和匹配方式需要人工核查并记录。仓库提供的 Wikipedia 语料准备脚本仅供尝试，访问失败或抓取页面变化时不能视为已经完成；页面版本可能与 QReCC 原始语料不同。

## 运行实验

先配置 README 中的生成与 Embedding 环境变量。以下命令运行的是小型中文演示，**不能用于比较方法优劣**：

```powershell
python -m rag.cli evaluate examples/evaluation/fruit-demo.json experiments/results/demo-new.json --index '<index_id>' --top-k 1 --strategies current history rewrite human
```

`human` 使用该轮人工改写作为检索输入，仅是参考组。QReCC 实验将数据路径换成完成证据映射的文件；各策略使用相同记录、参考历史、索引和生成设置。输出路径不能覆盖已有结果；输出包括数据与配置哈希、索引清单、模型名、部分源码哈希、每条 Query、召回列表、回答、引用编号检查、Token 和步骤耗时，以及按策略汇总。结果包含原始对话与回答，请保护文件并在分享前确认许可。

文档级 Hit@K 与 MRR@K 对有映射标签的样本计算；未标注样本不计入分母。答案 Exact Match 和 token F1 是简单词面匹配，英文按字母数字词切分、中文按字符切分并移除编号引用；它不是官方 QReCC 指标，无法代表语义正确性。引用指标只检查编号存在，不检查证据是否支持回答。没有价格和 Embedding 用量时不估算成本；改写与生成 Token 按服务实际返回记录。延迟受网络与服务负载影响。本阶段尚未实现置信区间、人工评审界面和正式报告。
