# 多轮对话检索增强生成系统

本项目面向课程研究与可扩展部署，构建支持多轮对话、证据引用与可复现实验的 RAG 系统。当前实现了单机基础 RAG、持久化索引与会话、命令行和本地 HTTP 接口，并完成百炼 API 联调及小规模研究实验；完整语料评测、自部署开放权重模型和生产部署尚未完成。

## 文档

- [系统架构与接口设计](docs/architecture.md)
- [数据与实验设计](docs/evaluation.md)
- [基线评测使用说明](docs/experiment_guide.md)
- [Query Rewrite 初步研究报告](docs/rewrite_study.md)
- [50 会话 Query Rewrite 对照结果](docs/rewrite_50_results.md)
- [Query Rewrite 检查与回退](docs/rewrite_stage2.md)
- [检索与效率优化实验](docs/retrieval_efficiency_study.md)
- [回答与引用可靠性说明](docs/answer_reliability.md)
- [Reranker、近似检索与较大 QReCC 测试](docs/reranker_ann_study.md)
- [开发计划与验收标准](docs/roadmap.md)

## 默认方向

- 开发阶段使用 Qwen API；服务器阶段通过配置连接自部署的开放权重模型。
- 研究重点为对话 Query 构造与重写，比较当前问题、历史拼接、模型改写及人工改写参考。
- QReCC 用于英文研究评测；自建中文知识库用于补充验证和演示。两者结果分别报告。
- 先实现可靠的单机系统，再依据实际部署需求扩展；设计不等同于已经达到生产环境要求。

模型密钥不会写入仓库。涉及数据集、模型和文档时，使用前需核对对应许可与使用条件。

回答默认检查数字引用；编号有误或没有有效引用时最多尝试修复一次，仍不合格时返回明确状态。`--citation-policy warn` 保留原回答供研究分析；编号有效不代表内容已由证据支持，相关限制和校准示例见可靠性说明。

## 本地运行和模型连接

需要 Python 3.10 或更高版本。运行时模型适配器仅依赖 Python 标准库；测试需要 pytest。以下命令在项目根目录的 PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m pytest -q
```

按服务商文档获取对应地域的 OpenAI 兼容 Base URL、密钥和可用模型名，并在当前终端设置环境变量。将占位符替换为自己的值；不要将真实密钥写入文件、聊天记录或版本库。`.env.example` 仅展示变量名，程序不会自动加载 `.env`。

```powershell
$env:LLM_BASE_URL = 'https://your-provider-compatible-endpoint/v1'
$env:LLM_API_KEY = 'your-key'
$env:LLM_MODEL = 'your-model'
rag chat-check '你好，请简短介绍自己。'

$env:EMBEDDING_BASE_URL = 'https://your-embedding-compatible-endpoint/v1'
$env:EMBEDDING_API_KEY = 'your-key'
$env:EMBEDDING_MODEL = 'your-embedding-model'
rag embed-check '用于验证的文本'
```

未安装项目时可用 `$env:PYTHONPATH='src'` 后执行 `python -m rag.cli chat-check '你好'`。查询配置路径可通过 `rag --config 路径 chat-check '你好'` 指定。生成和 Embedding 各自使用独立配置，可指向不同服务。切换服务器时修改对应地址、模型和凭据；服务需实现相应的 OpenAI 兼容端点。自部署服务若不校验密钥，可按其要求设置一个占位值。模型行为和服务参数仍需实际验证；Embedding 模型变化需要重建未来的向量索引。

真实调用可能产生费用。当前重试仅覆盖网络错误、408、429 和 5xx，且调用超时按单次请求计算；重试可能产生额外费用。不要将非本地服务配置为明文 HTTP，测试服务器可以使用本机 HTTP。

## 构建索引与多轮问答

先如上配置生成模型与 Embedding，再执行：

```powershell
rag index examples/knowledge --db data/rag.sqlite3
# 将输出中的 index_id 填入下面的命令。
rag ask '苹果有哪些颜色？' --index '<index_id>' --session demo --strategy history
rag ask '它是什么科？' --index '<index_id>' --session demo --strategy rewrite
```

索引支持递归读取 UTF-8 `.txt` 和 `.md`；切片大小及重叠是字符数，可通过 `--chunk-size` 和 `--overlap` 设置。Embedding 默认每批 10 条，可以通过 `--batch-size` 调整，需遵守所用模型的输入限制。构建完成后一次性提交 SQLite 事务，输出索引 ID 和文档清单；相同输入和配置复用索引。SQLite 保存在 `--db` 指定的文件，保存索引和会话。检索采用归一化向量的精确余弦扫描，适合小型知识库，随语料增长会变慢；不要将它当作大规模生产向量数据库。

`--strategy` 可选 `current`、`history`、`rewrite`。输出 JSON 包含检索 Query、回答、证据来源及片段、得分和引用编号检查。使用同一 `--session` 连续提问；不指定时自动生成会话 ID。`--min-score` 为余弦相似度阈值，默认 0 并非通用的相关性判定标准，应依据模型和数据校准。没有达到阈值的证据时系统拒答；有证据时，引用 ID 检查只能验证编号存在，不能保证回答被证据支持。`--context-chars` 限制证据文本字符数，不是模型 token 上下文限制；还需根据模型窗口调整历史、问题和输出长度。

研究性策略还可选 `rewrite_guarded`（强化改写约束）和 `rewrite_fusion`（原问题与强化改写分别检索并以 RRF 融合）。融合后分数为 RRF 而非余弦相似度，返回的 `score_type` 及 `retrieval_branches` 用于区分。实验尚未证明这些策略在一般场景优于默认策略。

还可选择 `rewrite_checked`、`rewrite_checked_fusion` 和 `rewrite_verified`，分别提供规则检查、检查后双路检索，以及额外模型复核；触发检查时回退到历史拼接。规则或复核通过不保证改写的语义忠实，详细结果及限制见研究说明。

可使用 `rag ask ... --context-mode deduplicate` 去掉同一文档中重复的字符区间；默认仍为 `standard`。本地 HTTP 服务可通过 `rag serve --embedding-cache-size 128` 开启进程内 LRU 向量缓存，默认关闭。缓存按完整文本精确匹配，不缓存回答；切换模型后应重启清空。实验可通过 `rag evaluate ... --embedding-cache-size 128` 观察缓存统计，但跨策略复用缓存会影响时延比较，应记录运行顺序和预热状态。

## 本地 HTTP API

在本机终端设置独立随机令牌后运行（需先设置模型相关环境变量）：

```powershell
$env:RAG_API_TOKEN = 'replace-with-a-long-random-secret'
rag serve --port 8000
```

服务仅绑定 `127.0.0.1`，使用单线程处理，适合本地开发；停止服务使用 Ctrl+C。另开终端，设置相同的 `RAG_API_TOKEN` 后：

```powershell
$headers = @{ Authorization = "Bearer $env:RAG_API_TOKEN" }
$body = @{ index_id = '<index_id>'; question = '苹果有哪些颜色？'; session_id = 'demo'; strategy = 'history' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/v1/ask' -Headers $headers -ContentType 'application/json' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

`GET /health` 同样需要令牌。索引通过 CLI 构建；API 不提供文件系统写入接口。该开发服务没有 TLS、细粒度授权、限流或多用户隔离，不能直接暴露到公网。会话内容明文存储在本地 SQLite；不要用它保存无法安全存储的敏感数据。同一会话的请求应串行处理。模型身份由服务地址和模型名组成；服务端在相同名称下替换模型时，仍须主动重建索引。

阶段三混合检索和 Reranker 对照实验见 [实验报告](docs/retrieval_stage3.md)。

阶段四配对评估与数据审计见 [使用说明](docs/evaluation_stage4.md)。

## 简便对话启动

配置用户或系统环境变量 `QWEN_API_KEY` 后，在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts/chat.py
```

输入问题并回车，输入 `/quit` 退出。每次启动自动创建新会话；可使用 `--session <会话ID>` 恢复会话、`--strategy rewrite` 启用改写、`--hybrid` 启用 ANN、混合检索和 Reranker。默认连接百炼及现有 QReCC 索引；会产生 API 调用。
