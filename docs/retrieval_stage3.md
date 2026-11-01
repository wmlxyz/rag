# 阶段三：混合检索与 Reranker 对照

## 实现与使用

在已有不可变向量索引的基础上，`build-lexical` 为每个索引版本建立独立的 SQLite FTS5 倒排索引；重建在同一事务中完成。查询时分别取向量与关键词排名，使用 RRF（k=60，按 chunk ID 去重）融合；可在融合后继续使用 Reranker。默认仍为 `dense`，需显式选择 `hybrid`：

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m rag.cli build-lexical --index 832ca220f6955497f36abdbf
.\.venv\Scripts\python.exe -m rag.cli ask 'Your question' --index 832ca220f6955497f36abdbf --backend ann --retrieval-mode hybrid --rerank --candidate-k 10 --top-k 5
```

`evaluate` 和 `serve` 也支持 `--retrieval-mode hybrid`；服务启动后该选项固定。Qwen 生成、Embedding、Reranker 环境变量配置见 [Reranker 使用说明](reranker_ann_study.md)。使用 ANN 前仍需先执行 `build-ann`；每个新索引需单独执行 `build-lexical`。FTS5 需由本机 SQLite 支持。

关键词使用 SQLite `unicode61` 分词，支持安全转义用户输入的 FTS 操作符；查询只保留按首次出现顺序去重的前 64 个词。较长的历史可能挤掉后面的当前问题词，英文停用词也可能引入噪声；中文不做专门分词，不能期待与中文专用搜索引擎同等效果。`min_score` 只过滤向量分支，关键词分支不受余弦阈值限制；RRF 分数不能与余弦或重排分数直接比较。混合检索是可选能力，尚未验证大规模容量和并发。

## 检索对照：12 条追问

沿用 1149 passage、3009 chunk 的受限候选池与已有 50 条覆盖筛选追问，按文件顺序取前 12 条。每条使用最近 8 条参考历史消息与当前问题拼接，ANN 召回，关键词检索后分别取前 10、20 个进行融合，最终 Top-5。每条查询的向量/关键词结果在同一次配对中复用，然后对四组候选分别调用 `qwen3-rerank`，共 48 次，均成功。来源命中指 Top-K 中至少一个 chunk 的来源匹配标注，不保证 chunk 包含答案；MRR 使用首次匹配来源的倒数排名。

| 候选 | 候选来源命中 | 重排前 Hit@5 | 重排前 MRR@5 | 重排后 Hit@5 | 重排后 MRR@5 | 平均重排秒数 | 排序 Token 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 向量 10 | 12/12 | 11/12 | 0.746 | 12/12 | 0.799 | 0.766 | 37,939 |
| 向量 20 | 12/12 | 11/12 | 0.746 | 12/12 | 0.799 | 0.814 | 75,308 |
| 混合 10 | 12/12 | 12/12 | 0.778 | 12/12 | 0.788 | 0.764 | 37,662 |
| 混合 20 | 12/12 | 12/12 | 0.778 | 12/12 | 0.799 | 0.849 | 76,002 |

本次 20 候选的排序 Token 约为 10 候选的两倍，而重排后指标相同或接近；不意味着 10 在其他数据上足够。向量检索平均 0.884 秒，包含 Embedding 网络请求和首次 ANN 加载；词法检索平均 0.035 秒，含 SQLite 查询，测量顺序固定，不能视为生产延迟基准。未记录 Embedding Token，无法计算总费用。

`test:1052:8` 的标注来源在向量前 5 中未命中，但在混合前 5 中排名第 3；混合 10 重排后排名第 5，混合 20 重排后排名第 3。排序分数是模型相关性判断，并非证据正确性的保证。`test:1002:2` 四种重排后首次标注来源均排第 4，说明“命中”仍掩盖了排名靠后的情况。

原始记录和源码哈希见 [retrieval-stage3-12.json](../experiments/results/retrieval-stage3-12.json)。重跑时需使用新输出路径：

```powershell
.\.venv\Scripts\python.exe scripts/retrieval_stage3.py data/qrecc/rewrite-eval-50.json experiments/results/new-retrieval-stage3.json --index 832ca220f6955497f36abdbf --limit 12
```

脚本逐条保存已完成记录；异常只记录类型。中断后文件保留部分结果，但目前不支持自动续跑。实验顺序固定，未做重复抽样或置信区间分析。

## 回答对照：8 条追问

从同一批中取前 8 条，history、ANN、Top-5、候选 10、Reranker、4000 字符预算，分别使用向量和混合检索；均独立调用生成模型，`citation_policy=warn`，故不做引用修复。每组 8 次均成功：

| 指标 | 向量 + 重排 | 混合 + 重排 |
|---|---:|---:|
| 来源 Hit@5 | 8/8 | 8/8 |
| 来源 MRR@5 | 0.698 | 0.681 |
| 答案 Token F1 均值 | 0.251 | 0.277 |
| 引用格式检查通过 | 7/8 | 6/8 |
| 平均总耗时 | 5.14 秒 | 5.12 秒 |

Token F1 是简单词面重合，不是语义正确率；引用格式检查不验证证据是否支持回答。两组单独生成，即使相同证据也可能有随机差异，因此 F1 差值不能直接归因于检索策略。本次使用 `warn` 是为了观察原始引用，常规使用仍默认 `repair`。

完整结果分别见 [向量问答](../experiments/results/retrieval-stage3-answer-dense-8.json) 和 [混合问答](../experiments/results/retrieval-stage3-answer-hybrid-8.json)，数据见 [8 条追问](../data/qrecc/retrieval-stage3-answer-8.json)。

## 结论边界与下一步

这些题目因已知标注来源可在候选池中找到而被筛选；候选池本身也按测试标注选取正例，且样本曾用于研究，不是未见测试集，不代表完整 QReCC。12 和 8 条的观察仅供调试，不能用于宣布混合检索或 Reranker 的一般效果。没有在同一大候选池重建其他 Chunk Size/Embedding 索引，本阶段只沿用[此前的小样本切片和上下文预算实验](retrieval_efficiency_study.md)。下一步应使用独立样本、盲审证据支持关系，比较中文分词和当前问题优先的词法查询，并测试多次运行、吞吐及更大规模索引。
