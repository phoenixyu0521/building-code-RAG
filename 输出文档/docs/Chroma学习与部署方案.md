# Chroma 向量库 学习与部署方案

> 面向项目：**建筑规范 AI 助手**（RAG，须带条文号 + 规范名 + 版本号）
> 制定日期：2026-09-12
> 对应计划：W4（9/28–10/4）「学 Chroma 向量库」，W6（10/12–10/18）「向量化入库 + 基础问答 + 引用出处」

> ### ⚠️ 2026-09-13 21:5x 实施后更正（本文 §3.2 / §4.3 的库路径已作废）
> 本文原定库位置 `输出文档/chroma_db/` **不可用** —— 实测确认 chromadb 1.5.9 在**含中文的路径**下
> **不写 HNSW 索引文件（4 个 `.bin`）且不报错**，必然做出打不开的库。
> **现行库位置：`D:\WorkBuddy\chroma_db`（纯 ASCII，仓库外）。**
> 落地细节见《本地部署操作指引.md》，本方案其余部分（选型、两段式、幂等重建思路）依然成立。

---

## 0. 一句话结论

**用 Chroma 嵌入式模式（`PersistentClient`）+ 本地中文 embedding 模型 `bge-small-zh-v1.5`，先做纯本地单机版；不要装 Docker、不要用 Chroma 内置默认模型。**

原因（均为本机实测约束，非通用建议）：

| 探测项 | 实测结果 | 对方案的影响 |
|---|---|---|
| Docker | **未安装** | 排除 Chroma Server 模式；嵌入式模式无需任何基础设施 |
| `huggingface.co` 直连 | **不通**（000） | 模型必须先配 `HF_ENDPOINT=https://hf-mirror.com` |
| `modelscope.cn` | **不通**（000） | 不走 ModelScope 下载 |
| PyPI 清华镜像 | 通（200，0.49s） | 依赖安装走 `-i` 清华源 |
| 已有环境 | 托管 Python 3.13.12；venv `default`、`paddleocr` | 新建独立 venv `chroma`，勿污染现有环境 |
| API Key | **无任何 embedding API 配置** | 走本地模型，零成本、且符合「规范原文不出本机」的合规红线 |

**合规理由（这条是硬约束）**：规范原文与 chunks 全部内容禁止上传第三方云。若用 OpenAI / 百炼等云 embedding，等于把 **1083 条规范全文**送出去。本地模型是唯一合规路径。

---

## 1. 学习计划（约 6 小时，拆 3 次）

### 阶段 A：概念建立（1.5 h）—— 只读不写代码

先搞清 4 个概念，**必须能用自己的话回答**，否则不进入下一阶段：

| 概念 | 必须能回答的问题 |
|---|---|
| Collection | 它和关系数据库的「表」像在哪、不像在哪？ |
| Embedding | 为什么「图书馆」和「阅览室」向量距离近，而和「图书馆」字面完全不同的「书库」也近？ |
| 距离函数 | `l2` / `cosine` / `ip` 三者区别？**为什么文本必须用 `cosine`？** |
| ANN / HNSW | 为什么 1083 条也要用近似搜索，而不是暴力算？ |

**推荐资料**（按顺序，不必全看）：
1. Chroma 官方文档「Concepts」+「Getting Started」——最权威，半小时
2. Chroma Cookbook（cookbook.chromadb.dev）——官方实战配方，看「Deployment Patterns」一节
3. B 站搜「RAG 原理」「向量数据库」，**优先系统讲解，不急着抄代码**（你计划里已写）

> ⚠️ 学习阶段的核心判断：**看懂 `distances` 是距离不是相似度分数，越小越近**。这是新手第一大坑。

### 阶段 B：最小可运行实验（2 h）—— 用你的真实数据

**不要用教程里的水果、动物示例。** 直接用你自己的规范条文。理由：教程示例让你感觉学会了，但学不到「中文长条文检索效果如何」这个真问题。

实验脚本要做 5 件事（写在 `输出文档/scripts/` 下）：
1. 读 `chunks_v5/chunks.jsonl` 的 1083 条
2. 建 Collection（指定 `cosine` + 中文模型）
3. 全部入库
4. 跑 5 个**真实建筑问题**，看 Top-5 召回
5. 打印距离值，人工判断排序合不合理

**验收标准**：问「图书馆书库的防火分区面积限值」，Top-5 里**必须**出现 GB 55037 或 JGJ 38 的相关条文。

### 阶段 C：参数与调优认知（2.5 h）

| 主题 | 要摸清的事 |
|---|---|
| 元数据过滤 | `where={"standard": "GB 55037-2022"}` 怎么用——**这是你项目的核心能力**（用户问「防火」就该只在防火规范里搜） |
| HNSW 参数 | `hnsw:space` / `M` / `search_ef` 各自影响什么；**1083 条这个量级几乎不用调**，但要懂原理 |
| 混合检索 | Chroma 自带全文检索能力，了解 BM25 + 向量混合检索，为后续 rerank 做铺垫 |
| 持久化 | 搞清楚 `chroma.sqlite3` + `data_level0.bin` 各自存什么 |

---

## 2. 部署方案

### 2.1 目录规划

```
D:\WorkBuddy\项目\
├── 建筑规范PDF\              # 素材（PDF，已 gitignore）
├── 输出文档\
│   ├── chunks_v5\            # 输入：chunks.jsonl（1083 条，现行版本）
│   ├── scripts\              # 脚本
│   │   ├── build_chroma.py       # 入库脚本（新建）
│   │   ├── query_chroma.py       # 检索测试（新建）
│   │   └── _chroma_lab.py        # 学习实验（新建，_ 前缀=草稿）
│   └── chroma_db\            # 向量库数据（新建，必须 gitignore）
└── .workbuddy\               # 工作区（已 gitignore）
```

**Chroma 数据库位置放 `输出文档/chroma_db/`**，与 `chunks_v5` 同级，便于对照与重建。

### 2.2 安装方式选型（本机实测，先读这个）

Chroma 有 4 种存在形态。**本项目选第 1 种。**

| # | 方式 | 安装动作 | 适用 | 本项目 |
|---|---|---|---|---|
| **1** | **嵌入式（`pip install chromadb`）** | 装进 venv，Python 进程内运行 | 单机、单应用、原型→小规模 | ✅ **选它** |
| 2 | `pip install chromadb[full]` | 同上，外加 OpenAI/Cohere 等 extras | 要用云 embedding | ❌ 多余，且云 embedding 违反合规 |
| 3 | Server 模式（`chroma run` / Docker） | 起独立服务，`HttpClient` 连 | 多应用共享一份库 | ❌ **本机未装 Docker** |
| 4 | Chroma Cloud | 无需安装，托管 | 生产多租户 | ❌ **规范全文上第三方云，破红线** |

**为什么选嵌入式（三条理由）**：

1. **零基础设施**——本机实测 `docker: command not found`，且嵌入式不需要任何服务进程。
2. **合规**——数据不出本机。方式 4 直接违反「规范原文不上第三方云」；方式 2 的 extras 是为云 embedding 准备的，用不上。
3. **API 完全一致**——`PersistentClient` 与 `HttpClient` 的 collection 用法一模一样。将来真要升 Server 模式，**查询代码一行不用改**，只换取客户端那两行。所以现在选嵌入式不锁死未来。

**一个必须避开的错误**：`chromadb.Client()` 是**纯内存**，进程退出数据即丢。**必须用 `chromadb.PersistentClient(path=...)`**。

#### 依赖可用性核验（2026-09-13 实测）

Python 3.13 是较新的运行时，装前逐项核实了 Windows wheel 是否存在：

| 包 | 目标版本 | requires_python | win_amd64 + cp313 wheel | 结论 |
|---|---|---|---|---|
| chromadb | 1.5.9 | `>=3.9` | 纯 Python（`py3-none-any`），无需编译 | ✅ |
| torch | 2.14.0 | `>=3.10` | 有（2.9.0/2.9.1 等 20 个） | ✅ |
| onnxruntime | 最新 | — | 有（1.29.0 / 1.30.0） | ✅ |
| grpcio | 最新 | — | 有（45 个） | ✅ |
| tokenizers | 最新 | — | 有（0.20.2 / 0.20.3） | ✅ |
| sentence-transformers | 6.0.1 | `>=3.10` | 纯 Python | ✅ |
| pypika / overrides | — | — | 纯 Python | ✅ |

**结论：托管的 Python 3.13.12 可直接使用，不必降到 3.12。**

#### 镜像源实测（2026-09-13）

| 源 | simple 索引（pip 实际用这个） | JSON 接口 |
|---|---|---|
| 清华 tuna | ✅ 有 chromadb **1.5.9**（130 个版本 / 880 个文件） | ⚠️ **返回陈旧的 0.5.5**（缓存，勿据此判断） |
| pypi.org | ✅ 同上有 1.5.9 | ✅ 准确 |

> ⚠️ **坑**：`https://pypi.tuna.tsinghua.edu.cn/pypi/chromadb/json` 这个 JSON 接口返回的是 **2024 年的旧版本号**（chromadb 0.5.5、torch 2.4.0）。**pip 不走这个接口，走 `simple/` 索引，拿到的是正确的 1.5.9**。若手工查版本号，请用 `https://pypi.org/pypi/chromadb/json` 或直接看 `simple/` 页面。

**版本选择**：Chroma **0.x → 1.x 磁盘格式不兼容**（1.x 是 Rust 重写版，写/查快 3–5 倍）。本项目从零开始，直接用 **1.x**，无历史包袱、无需迁移。

### 2.3 环境搭建（一次性）

```bash
# 1) 新建独立 venv（勿用 paddleocr 那个，依赖会冲突）
PY313="C:/Users/A/.workbuddy/binaries/python/versions/3.13.12/python.exe"
"$PY313" -m venv "C:/Users/A/.workbuddy/binaries/python/envs/chroma"

# 2) 安装（清华源；chromadb 会拉 torch，体积约 2-3 GB，耐心等）
PIP="C:/Users/A/.workbuddy/binaries/python/envs/chroma/Scripts/pip.exe"
"$PIP" install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    chromadb sentence-transformers
```

**版本策略**：Chroma 1.x 是 Rust 重写版（写/查快 3–5 倍），**0.x → 1.x 磁盘格式不兼容**。本机从零开始，直接用 1.x，无需迁移。

### 2.4 中文 Embedding 模型选型

| 模型 | 维度 | 体积 | 中文检索分（C-MTEB Retrieval） | 建议 |
|---|---|---|---|---|
| **bge-small-zh-v1.5** | 512 | ~100 MB | 61.77 | ✅ **首选**——CPU 友好，1083 条规模足够 |
| bge-base-zh-v1.5 | 768 | ~400 MB | 69.49 | 若精度不够再升 |
| bge-large-zh-v1.5 | 1024 | ~1.2 GB | 70.46 | CPU 上太慢，本项目不需要 |
| text2vec-base-chinese | 768 | ~400 MB | 38.79 | ❌ 老项目兼容用，别选 |

**决策**：先用 `bge-small-zh-v1.5`。理由——① 你的数据只有 1083 条，小模型足够；② 本机无 GPU，CPU 推理速度是关键；③ 后续要对比检索效果，小模型换大模型只需重建库（1083 条重建约 1 分钟）。

**下载必须先配镜像**（本机 HF 直连不通）：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 2.5 入库脚本要点

```python
import chromadb, json
from chromadb.utils import embedding_functions

# 关键 1：余弦距离（文本必须用 cosine，不能用默认的 l2）
client = chromadb.PersistentClient(path=r"D:\WorkBuddy\项目\输出文档\chroma_db")

# 关键 2：显式指定本地中文模型，绝不能让 Chroma 用它的默认英文模型
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5"
)

collection = client.get_or_create_collection(
    name="building_codes",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},     # 关键 3
)

# 关键 4：把 chunks 的 12 个字段全塞进 metadata，检索时可过滤
# 关键 5：用 upsert 而非 add —— 脚本可重复运行不报错
```

**5 个关键点说明**：

| # | 要点 | 不这么做的后果 |
|---|---|---|
| 1 | `PersistentClient` | 用 `Client()` 是纯内存，**重启即丢**——新手第一大坑 |
| 2 | 显式指定中文模型 | 默认 `all-MiniLM-L6-v2` 是**英文模型**，中文检索效果灾难 |
| 3 | `hnsw:space="cosine"` | 默认是 `l2`，对文本 embedding 不合适 |
| 4 | 元数据完整落库 | 过滤检索是项目核心能力，metadata 缺失就废了一半 |
| 5 | `upsert` | `create_collection` 二次运行报错；`add` 重复 ID 报错 |

**metadata 的落库字段**（直接映射 chunks.jsonl 的 12 个字段）：

| 字段 | 用途 | 是否可过滤 |
|---|---|---|
| `standard` | 规范代号 | ✅ 核心过滤维度 |
| `standard_name` | 规范中文名 | ✅ |
| `version` | 版本年 | ✅ |
| `force_status` | 强制状态（强条/已废止） | ✅ **行业价值核心** |
| `status` | 现行/废止 | ✅ |
| `article_no` | 条文号 | ✅ 精确匹配 |
| `chapter` | 章号 | ✅ |
| `page_start` / `page_end` | 页码 | 展示用 |
| `kind` | article/section/table | ✅ 可按类型过滤 |
| `text` | 条文原文 | 作为 document |

> Chroma metadata **只支持 str/int/float/bool**，`None` 会报错。注意那 1 条 `chapter` 缺失的记录要填 0 或空串。

### 2.6 检索验证脚本要点

```python
res = collection.query(
    query_texts=["图书馆书库的防火分区面积限值是多少"],
    n_results=5,
    where={"standard": {"$in": ["GB 55037-2022", "JGJ 38-2015"]}},  # 元数据过滤
    include=["documents", "metadatas", "distances"],
)
# distances 越小越近——不是相似度分数！
```

**必测的 5 类查询**（覆盖项目真实场景）：

| 类型 | 示例 | 验证点 |
|---|---|---|
| 精确条文号 | 「GB 50352-2019 第 6.7.1 条」 | 能否一字不差命中 |
| 自然语言 | 「图书馆疏散楼梯的净宽度要求」 | 语义召回能力 |
| 跨规范 | 「防火分区怎么划分」 | 能否同时召回 GB 55037 与 JGJ 38 |
| 废止条文 | 「GB 50016 的强条还有效吗」 | `force_status` 过滤能否生效 |
| 表格类 | 「设计使用年限分类表」 | `kind=table` 的召回质量 |

---

## 3. 与 Dify 版的关系（避免重复劳动）

你现在有两条链路，**不要混用**：

| | Dify 链路（已完成） | Chroma 链路（本方案） |
|---|---|---|
| 用途 | 快速验证、演示 | **最终交付形态**（自研代码版） |
| Embedding | Dify 云端（百炼） | **本地 bge-small-zh** |
| 切分 | Dify 按 `@@@` 切 | **直接用 chunks.jsonl**，无需再切 |
| 合规 | ⚠️ 规范原文上云 | ✅ **全本地** |

**关键收益**：`chunks_v5` 是一次性切好的，Chroma 直接读 JSONL 入库，**不需要任何切分逻辑**——这是你前期投入的回报。**注意用 v5（现行版），不要用 v4**——v5 修掉了 `NOISE_RE` 吞数字的缺陷，找回了 16 条内容与 9 条条文内部枚举序号。

**对照价值**：同 5 个问题分别问 Dify 版和 Chroma 版，**哪个引用更准**，就是你的评测数据，也是作品集里最有说服力的对比。

---

## 4. 执行顺序与时间安排

### 第一次（W4，约 3 h）
1. 环境搭建：建 venv + 装依赖（30 min，含 2–3 GB 下载）
2. 模型下载：配 `HF_ENDPOINT` 后下 `bge-small-zh-v1.5`（10 min）
3. 阶段 A 概念学习（1.5 h）
4. 跑通「5 条数据入库 + 1 次查询」最小闭环（1 h）

**里程碑：能在终端里看到一条规范条文的检索结果和距离值。**

### 第二次（W4 末，约 3 h）
1. 阶段 B：1083 条全量入库（约 10–30 min，CPU 推理）
2. 阶段 C：元数据过滤实验（1 h）
3. 5 类查询验证 + 记录结果表（1 h）
4. 调整参数重跑对比（1 h）

**里程碑：5 类查询各有一个明确结论（命中/未命中/需调优）。**

### 第三次（W6，进入 MVP）
按计划 W6 做「向量化入库 + 基础问答 + 引用出处」：接 LLM API，把 Top-K 检索结果拼进 Prompt，**强制要求输出「条文号 + 规范名 + 版本号」**。

---

## 5. 已知风险与对策

| 风险 | 症状 | 对策 |
|---|---|---|
| **torch 下载慢/失败** | pip 卡在 `Downloading torch` | 用清华源；torch CPU 版约 200 MB，若失败改 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple` 单独装 |
| **HF 模型下载失败** | `ConnectionError` / 超时 | 必须 `export HF_ENDPOINT=https://hf-mirror.com`；若仍失败，用 `huggingface-cli download` 先拉模型 |
| **安全删除守卫** | 进程中途猝死，日志有 `[safe-delete][SAFE_DELETE_BULK_*]` | 已知本机问题；批量操作拆小，或用 `%TEMP%` 目录 |
| **中文输出乱码** | 终端字符错乱 | 已知问题：Paddle 的 GBK 日志混流所致。Chroma 无此问题；若出现，输出重定向到文件再看 |
| **`None` 值 metadata** | `ValueError` 入库失败 | 那 1 条 `chapter` 缺失的记录要预处理成 `0` |
| **重启后查不到数据** | collection 为空 | 检查是否误用 `chromadb.Client()` 而非 `PersistentClient` |
| **多进程写入** | 数据损坏 | **Chroma 线程安全但不是进程安全**；永远不要让两个脚本同时写同一路径 |

**回退方案**：`chroma_db/` 是纯目录，删掉重跑入库脚本即可（1083 条约 1 分钟）。**入库脚本必须写成幂等的**（`upsert` + `get_or_create_collection`），这样任何实验失败都能一键重来。

---

## 6. 本阶段结束时应交付的东西

- [ ] `build_chroma.py`：可重复运行的入库脚本
- [ ] `query_chroma.py`：5 类查询的验证脚本
- [ ] `chroma_db/`：已持久化的 1083 条向量库
- [ ] **检索结果记录表**：5 类查询 × 命中情况 × 距离值（这是作品集素材）
- [ ] 一段能写进简历的话：「基于 Chroma + bge-small-zh 搭建本地向量检索，1083 条规范条文，支持元数据过滤，纯本地部署满足规范版权合规要求」

---

## 7. 待拍板的点

**已定（无需再选）**：
- 安装方式 → **嵌入式 `pip install chromadb`**（§2.2 已论证，4 种方式逐一排除）
- Python 运行时 → **托管 3.13.12**（依赖 wheel 已逐项核实可用）
- Server 模式 / Docker → **不做**（本机未装 Docker，且嵌入式 API 完全一致，将来可无痛升级）

**待你决定**：

1. **embedding 模型**：
   - `bge-small-zh-v1.5`（512 维 / 100 MB / 检索分 61.77）——快，1083 条够用
   - `bge-base-zh-v1.5`（768 维 / 400 MB / 检索分 69.49）——准，CPU 慢约一倍
   > 建议先 small。换模型只需重建库（1083 条约 1 分钟），不锁死。
2. **是否现在动手**：执行 §2.3 会下载 `chromadb + torch` 约 2–3 GB 到 C 盘（当前余量 124 GB，充足）。说一声我就开始。
