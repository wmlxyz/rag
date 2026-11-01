# Reranker、近似检索与较大 QReCC 候选池

## 已实现

- `--rerank` 通过百炼原生文本排序接口调用 `qwen3-rerank`：先召回 `--candidate-k` 个切片，再重排并保留 `--top-k`。返回 `candidates`、`retrieved`、`rerank_usage`、`rerank_error` 和重排耗时。默认关闭；故障默认报错，`ask` 和 `evaluate` 可显式使用 `--rerank-fallback` 回退并记录状态；本地服务目前不提供该开关。它不是 OpenAI Chat 兼容协议，使用独立配置。
- `--backend ann` 使用 CPU FAISS HNSW；必须先对已有的索引执行 `build-ann`，默认仍使用精确检索。使用归一化向量内积近似余弦，支持 `--ef-search`。文件和 SQLite 标签映射按版本发布，加载前检查 SHA-256、向量维度与数量；首次加载后在进程内复用。近似结果可能遗漏精确 Top-K，需要针对数据集评测。HNSW 和反序列化索引占用内存，构建与序列化还会有额外内存峰值；不等于支持无限大或分布式语料。
- 后续建库按批将向量写入临时 SQLite，完成后在单个事务中发布，不再将全部向量常驻 Python 列表。仍需两遍读取文档，保留文档清单，且尚无失败断点续跑或分布式构建。进程意外终止可能留下暂存文件。

## 使用

FAISS 为可选依赖，安装项目时运行 `python -m pip install -e '.[ann]'`。当前工作区 `.venv` 已安装 FAISS；示例使用该环境并在项目根目录执行。除生成和向量配置外，还需要设置百炼原生排序接口：

```powershell
$env:PYTHONPATH = 'src'
$env:LLM_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
$env:LLM_MODEL = 'qwen-plus'
$env:LLM_API_KEY = $env:QWEN_API_KEY
$env:EMBEDDING_BASE_URL = $env:LLM_BASE_URL
$env:EMBEDDING_MODEL = 'text-embedding-v4'
$env:EMBEDDING_API_KEY = $env:QWEN_API_KEY
$env:RERANK_BASE_URL = 'https://dashscope.aliyuncs.com/api/v1'
$env:RERANK_MODEL = 'qwen3-rerank'
$env:RERANK_API_KEY = $env:QWEN_API_KEY

# 已有较大 QReCC 索引；更换索引需要重新 build-ann。
.\.venv\Scripts\python.exe -m rag.cli build-ann --index '<index_id>'
.\.venv\Scripts\python.exe -m rag.cli ask 'Your question' --index '<index_id>' --backend ann --rerank --candidate-k 20 --top-k 5
```

地区、服务权限、模型名称及单次请求限制以 [百炼文本排序文档](https://help.aliyun.com/zh/model-studio/text-rerank-api) 为准。FAQ 中的 Reranker 分数与召回余弦分数不可比较：`score_type` 标注最终分数类型，`candidates` 保存重排前分数；相似度阈值先用于召回。

本地服务和实验也支持 `--backend ann --rerank --candidate-k 20`。API 服务的配置由服务启动选项固定；默认本机认证、单线程，不可直接用于公网生产。切换 Embedding 模型时需先构建新文本索引，再为其构建 ANN 文件。重建 ANN 会切换元数据到新版本，旧版本文件不会自动删除。

## 较大候选池

先在官方 QReCC 测试集中，以种子 `20261028` 随机选择 200 个完整会话（1244 轮）；再扫描第三方镜像固定版本的前 100,000 行，保留与这些问题标注 passage ID 精确匹配的片段，并从非目标片段中以相同种子选取 1000 个干扰片段。最终为 1149 个 passage、3009 个 800/120 字符 chunk。1244 轮中 644 轮有 passage ID，199 轮至少有一个 ID 被匹配；部分轮次有多个参考 ID，可能只有部分匹配。未映射记录不计算检索指标。

这仍是**依据测试标注选择正例、在语料前缀选择干扰项的受限候选池**，不是官方完整 5400 万片段语料，不可与完整开放域基准直接比较。镜像版本与来源、缺失 ID、选择方式和文件哈希见 `data/qrecc/test200-pool-1000neg/manifest.json`。SQLite 文本索引 ID 为 `832ca220f6955497f36abdbf`；其 ANN 索引包含 3009 个向量、1024 维，文件约 13.1 MB。

## 50 个会话的检索诊断

在 200 个会话中按固定文件顺序选择前 50 个具有历史和已匹配参考片段的不同会话，每会话取首个合格追问。因为依赖可用标注和前缀覆盖，这些样本不是无偏的总体样本。使用参考历史（最近 8 条消息）拼接当前问题，精确召回 20 个候选，比较原始前 5 与百炼重排前 5；不运行生成模型。两者使用相同候选，50 次均成功。

| 指标 | 重排前 | 重排后 |
|---|---:|---:|
| 标注来源 Hit@5 | 49/50 | 49/50 |
| MRR@5 | 0.836 | 0.892 |
| 标注来源首次出现在第 1 位 | 38/50 | 41/50 |

按首个标注来源的倒数排名比较，9 条提高、1 条降低、40 条不变；有一条从命中变成未命中，另一条则从未命中变为命中。Reranker 平均耗时 0.859 秒，50 次返回共 288,276 排序 Token；不含 Embedding 和生成模型用量，不足以计算完整费用。该测试仅比较检索，没有评测回答质量或引用是否支持回答。

相同查询向量、相同候选数的 FAISS HNSW Top-20 与精确 Top-20 的平均集合重合率为 99.6%，最低单条 90%。排序算法不同且存在浮点误差或并列分数，该数字不是标注证据召回率。每次先预热该查询的 Embedding 缓存，按顺序交替运行精确和 ANN；排除第一个 ANN 冷加载后，49 次平均：精确约 1.403 秒、ANN 约 0.0041 秒。该时间包含本地 SQLite 取数，排除 Embedding 网络调用、Reranker 与生成；只适用于此机器和 3009 条向量的小型测试，不是生产容量指标。首次 ANN 加载含文件读取、校验与反序列化，约 0.31 秒。

详细记录见 `experiments/results/retrieval-large-50.json`；通过 `scripts/compare_retrieval.py` 重跑（该脚本后续已补充记录数据和源码哈希，此次原始文件没有这些字段）。真实 ANN→重排→生成的单轮测试也已运行，且正确识别到模型生成了无效引用编号；引用支持性校验和自动修复尚未完成。
