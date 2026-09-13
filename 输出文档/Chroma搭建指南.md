# Chroma 本地知识库搭建指南

> 面向本项目（建筑规范 RAG），从零到「能检索」的完整步骤。
> 上游产物：`输出文档\chunks_v5\chunks.jsonl`（1083 条，已通过校验）
> 本文所有体积/耗时数字均为 2026-09-13 在本机实测。

> ### ⚠️ 2026-09-13 21:5x 重要更正 —— 库路径必须纯 ASCII
> 本文中出现的 `D:\WorkBuddy\项目\输出文档\chroma_db` **已作废**。
> 实测确认：**chromadb 1.5.9 在含中文的路径下不写 HNSW 索引文件（4 个 `.bin`）且不报错** →
> 必然得到打不开的「半成品库」。**库现位于 `D:\WorkBuddy\chroma_db`（纯 ASCII，仓库外）。**
> 环境搭建（venv / 依赖 / 模型缓存）部分不受影响，仍然有效；
> 建库与验收请以《**本地部署操作指引.md**》为准。

---

## 0. 先回答：「千问的 embedding 模型」能不能用？

**要分两种形态，结论完全相反。**

| 形态 | 能不能用 | 原因 |
|---|---|---|
| **通义千问云端 API**（阿里云百炼 DashScope，`text-embedding-v3` / `v4`） | ❌ **不要用** | 每嵌一段文本，规范原文就发一次阿里云。**这直接破你自己定的合规红线**（规范全文不上第三方云） |
| **开源本地模型** `Qwen/Qwen3-Embedding-0.6B`（另有 4B / 8B） | ✅ **推荐** | 权重下载到本机、推理在本机 CPU 跑完，**数据一步都不出电脑** |

一句话：**「千问的模型」可以用，「千问的接口」不行。**

云端 API 唯一的诱惑是省事（不用下模型、不用装 torch），但它把 1083 条规范正文全部上传到阿里云服务器 —— 这是本项目从第一天就否掉的路，别再回头。

> 顺带说清楚：这不代表 API 有问题，而是你的**场景**不允许。如果是个人笔记、公开博客，用 API 完全没问题。

---

## 1. 现状盘点

| 环节 | 状态 |
|---|---|
| 专用 venv `envs\chroma` | ✅ 已建（Python 3.13.14） |
| `chromadb 1.5.9` | ✅ 已装，冒烟测试通过 |
| 数据库目录 `输出文档\chroma_db\` | ✅ 已初始化（`chroma.sqlite3` 184 KB） |
| 切分产物 `chunks_v5` | ✅ 1083 条，校验全绿 |
| **embedding 依赖** | ❌ 待装（第 3 步） |
| **embedding 模型权重** | ❌ 待下（第 4 步） |
| **建索引脚本** | ❌ 待写（第 5 步） |

本文的 3 → 4 → 5 → 6 步就是把最后四行做完。

---

## 2. 选型：用哪个模型？

| 模型 | 权重体积 | 向量维度 | 中文检索 | 建议 |
|---|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | **1.11 GB** | 1024（可自定义降到 32～1024） | 第一梯队，MTEB 多语榜前列 | **首选** |
| `BAAI/bge-small-zh-v1.5` | **91 MB** | 512 | 够用，明显弱于千问 | 想省流量选它 |
| `BAAI/bge-base-zh-v1.5` | 392 MB | 768 | 介于两者之间 | 折中 |

**为什么推荐千问 0.6B：**
- 你的库只有 1083 条、正文中位长度才 56 字，语料极小 → 模型大一点**完全跑得动**
- 建筑规范里同义表述多（"净宽"/"最小宽度"/"不应小于"），语义模型越强，召回越准
- 1.11 GB 的下载量，按实测 ~5 MB/s 大约 **4 分钟**
- Apache-2.0 许可，可商用

> 两个模型都不用改代码 —— 只换第 5 步脚本顶部那一行 `MODEL` 变量。

---

## 3. 步骤一：装依赖（约 207 MB）

**必须在 chroma 那个 venv 里装**，不要用系统的 `python`。

先把这一行设好 —— **后面每一步都要用到它**（PowerShell 里设一次，当前窗口一直有效）：

```powershell
$PY = "C:\Users\A\.workbuddy\binaries\python\envs\chroma\Scripts\python.exe"
```

然后装依赖：

```powershell
& $PY -m pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> ⚠️ **别用 `activate.bat` 去"激活"venv —— 在 PowerShell 里它不生效。**
> 原因是 `.bat` 属于 cmd 的脚本，在 PowerShell 里执行会开一个子进程，环境变量传不回当前窗口。
> 想在 PowerShell 里真激活得用 `Activate.ps1`，但本机执行策略可能拦截。
> **绝对路径（`$PY`）永远有效，也最省心，本文全程用这种写法。**
>
> （如果你习惯用 **cmd** 而不是 PowerShell，那 `...\Scripts\activate.bat` 是正常的，激活后提示符前会出现 `(chroma)`。）

**实测下载清单（42 个包，合计 ≈ 207 MB）：**

| 包 | 体积 |
|---|---|
| `torch 2.14.0`（Windows **CPU 版**） | 118.4 MB |
| `scipy 1.18.1` | 34.9 MB |
| `numpy 2.5.3` | 12.0 MB |
| `transformers 5.17.0` | 11.7 MB |
| `scikit-learn 1.9.1` | 7.9 MB |
| `sentence-transformers 6.0.1` | 0.7 MB |
| 其余 36 个包 | ≈ 21 MB |

> ⚠️ **纠正一个旧说法**：之前提过「torch 要 2 GB+」—— 那是 Linux 上带 CUDA 的版本。
> **Windows 从 PyPI 装到的 torch 是纯 CPU 版，只有 118 MB。**所以这一整套比想象中轻得多。

---

## 4. 步骤二：下模型（约 1.11 GB）

### 4.1 为什么要设 `HF_ENDPOINT`

本机**连不上 `huggingface.co`**（境外），必须走国内镜像。实测 `hf-mirror.com` 通、`modelscope.cn`（魔搭）也通。

```powershell
# PowerShell 里这样设（当前窗口有效）
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

> 这一步必须在**同一个窗口**里、在下模型之前执行。关掉窗口就失效，下次要重设。

### 4.2 下载

```powershell
& $PY -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B')"
```

`huggingface_hub` 随 `sentence-transformers` 一起装好了，不用另外装。

> **你的窗口里现在应该有这两个变量**：`$PY`（第 3 步设的）和 `$env:HF_ENDPOINT`（4.1 设的）。
> 换窗口就要重设 —— 这是最容易忘、也最容易报「连不上 huggingface」的原因。
>
> **其实这一步可以跳过**：直接跑第 5 步的 `build_index.py`，它到 `SentenceTransformer(MODEL)` 那行会**自动下载**同一个模型。
> 单独先下只是为了能看清进度、失败了方便重试。

**模型落到哪：**

```
C:\Users\A\.cache\huggingface\hub\models--Qwen--Qwen3-Embedding-0.6B\
```

（不是工作区、不是桌面 —— huggingface 的标准缓存位置。想换位置就设 `HF_HOME`，见 4.3。）

**下载慢/中断了怎么办：** 重跑同一条命令即可，它会续传，已下的分片不重复下。

### 4.3 不一定要放 C 盘 —— 想挪到 D 盘怎么做

本方案一共有**两样东西**占空间，默认都落在 C 盘：

| 东西 | 默认位置 | 体积 |
|---|---|---|
| venv（Python 环境 + 依赖） | `C:\Users\A\.workbuddy\binaries\python\envs\chroma` | **1219 MB** |
| 模型权重缓存 | `C:\Users\A\.cache\huggingface\hub\` | **1.11 GB** |

**① 模型缓存 → 一条环境变量就能改（推荐）**

`HF_HOME` 是 huggingface 缓存的总目录，模型会落到 **`$HF_HOME\hub\`**：

```powershell
$env:HF_HOME     = "D:\WorkBuddy\models\hf"
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

改完就走这个位置，**下载前设就行，不用重新装依赖**：

```
D:\WorkBuddy\models\hf\hub\models--Qwen--Qwen3-Embedding-0.6B\
```

> 想让它在**每个新窗口都自动生效**（不用每次重设），在 PowerShell 里跑一次：
>
> ```powershell
> [Environment]::SetEnvironmentVariable("HF_HOME", "D:\WorkBuddy\models\hf", "User")
> [Environment]::SetEnvironmentVariable("HF_ENDPOINT", "https://hf-mirror.com", "User")
> ```
>
> 设完**要新开窗口**才生效。想撤掉就把第三个参数值改成 `$null` 再跑一次。

**② venv → 不建议搬，真要挪只能重建**

venv 里的 `Scripts\pip.exe`、`chroma.exe` 等启动器把**绝对路径写死在自己文件里**，直接剪切文件夹会让它们全部失效（`python.exe` 本身可能还能跑，但 pip / CLI 会坏）。

如果确实想搬到 D 盘，正确做法是**新建一个再删旧的**：

```powershell
$PY = "C:\Users\A\.workbuddy\binaries\python\versions\3.13.12\python.exe"
& $PY -m venv "D:\WorkBuddy\models\envs\chroma"
& "D:\WorkBuddy\models\envs\chroma\Scripts\python.exe" -m pip install chromadb sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

装完记得把后面所有命令里的 `$PY` 换成新路径，旧的删掉。

**要不要挪？** 本机实测 C 盘剩 **119.8 GB**，D 盘剩 **110.9 GB** —— 两样加起来才 2.3 GB，**放着不动完全没问题**。真正值得挪的理由只有「C 盘是系统盘、不想让它越来越满」。**模型缓存改 `HF_HOME` 是性价比最高的那一步，venv 留着就行。**

**备选源（hf-mirror 挂了再用）—— 魔搭 ModelScope：**

```powershell
& $PY -m pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
& $PY -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B')"
```

魔搭是**阿里自家的模型社区**（Qwen 也是阿里的），属于第一方官方源，比镜像更正规。

> ⚠️ **走魔搭下载的话，第 5 步脚本里的 `MODEL` 必须改。**
> 别再写 `"Qwen/Qwen3-Embedding-0.6B"`（那是 HuggingFace 的仓库名，会让它去 HF 找），
> 要改成**下载时输出的那个本地目录绝对路径**，例如：
>
> ```python
> MODEL = r"C:\Users\A\.cache\modelscope\hub\models\Qwen\Qwen3-Embedding-0.6B"
> ```
>
> 下载完把终端打印的**实际路径**抄下来，别照抄这里的示例。

**❌ GitHub 上没有权重**：`QwenLM/Qwen3-Embedding` 仓库里只有代码和文档，模型权重（那 1.11 GB）从来不放 GitHub。

---

## 5. 步骤三：建索引

脚本**已经生成好了**，位置：

```
D:\WorkBuddy\项目\输出文档\scripts\build_index.py
```

> ⚠️ **先确认这个文件存在再运行。** 如果直接跑报
> `can't open file '...build_index.py': [Errno 2] No such file or directory`
> —— 就是文件还没建。下面代码贴在这里，纯属备份参考，**正常情况下不用手抄**。

<details>
<summary>点开看脚本内容（备份）</summary>

```python
# -*- coding: utf-8 -*-
"""把 chunks_v5 的 1083 条规范灌进本地 Chroma 向量库。"""
import json, os, sys, time

import chromadb
from sentence_transformers import SentenceTransformer

# --- Windows 控制台默认 GBK，打印中文会炸，强制 UTF-8 ---
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ============ 配置区 ============
PROJ      = r"D:\WorkBuddy\项目"
CHUNKS    = os.path.join(PROJ, r"输出文档\chunks_v5\chunks.jsonl")
DB_DIR    = os.path.join(PROJ, r"输出文档\chroma_db")
MODEL     = "Qwen/Qwen3-Embedding-0.6B"   # 省流量可换 "BAAI/bge-small-zh-v1.5"
COLL      = "standards"
USE_HEADER = False   # True = 把「规范名 + 条文号」拼进待向量化文本（可提升规范名命中）
# ================================

META_KEYS = ("standard", "standard_name", "version", "force_status", "status",
             "article_no", "chapter", "kind", "page_start", "page_end")


def clean_meta(r):
    """Chroma 不接受 None，统一转成空串。"""
    return {k: ("" if r.get(k) is None else r.get(k)) for k in META_KEYS}


def main():
    # 1) 读数据
    with open(CHUNKS, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print(f"[1/4] 读入 {len(rows)} 条")

    # 2) 载模型
    t = time.time()
    model = SentenceTransformer(MODEL)
    print(f"[2/4] 模型就绪 {time.time() - t:.1f}s，"
          f"向量维度 {model.get_sentence_embedding_dimension()}")

    # 3) 向量化（文档侧不加 query 指令）
    docs = [(f"{r['standard_name']} {r['article_no']}：{r['text']}"
             if USE_HEADER else r["text"]) for r in rows]
    t = time.time()
    embs = model.encode(docs, batch_size=32, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True)
    print(f"[3/4] 向量化完成 {time.time() - t:.1f}s，shape={embs.shape}")

    # 4) 写入 Chroma
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLL)      # 重跑时先清空，避免重复累积
        print("      （已清除同名旧集合）")
    except Exception:
        pass
    col = client.get_or_create_collection(
        COLL, metadata={"hnsw:space": "cosine", "description": "建筑规范条文库"})

    for i in range(0, len(rows), 256):
        j = min(i + 256, len(rows))
        col.add(
            ids=[r["id"] for r in rows[i:j]],
            documents=docs[i:j],
            embeddings=embs[i:j].tolist(),
            metadatas=[clean_meta(r) for r in rows[i:j]],
        )
    print(f"[4/4] 入库完成，集合 `{COLL}` 内共 {col.count()} 条")


if __name__ == "__main__":
    main()
```

</details>

**运行：**

```powershell
& $PY "D:\WorkBuddy\项目\输出文档\scripts\build_index.py"
```

**预期输出**（1083 条、CPU 推理，约 1～4 分钟）：

```
[1/4] 读入 1083 条
[2/4] 模型就绪 8.3s，向量维度 1024
[3/4] 向量化完成 95.2s，shape=(1083, 1024)
[4/4] 入库完成，集合 `standards` 内共 1083 条
```

### 关键设计说明（别改错）

| 点 | 为什么 |
|---|---|
| `hnsw:space="cosine"` | 模型自带 L2 归一化，必须配余弦距离；用默认的 L2 排名会不准 |
| `documents` 存原始正文 | 供显示，也供 `where_document={"$contains": "3.2.1"}` 做精确条文号匹配 |
| `ids` 用 `chunk["id"]` | 形如 `GB 50352-2019_3.2.1`，天然唯一，重跑不会重复 |
| `normalize_embeddings=True` | 与 cosine 配套 |
| 元数据 10 个字段 | 后面能按规范/类型/强条状态过滤，这是 Dify 给不了的 |

---

## 6. 步骤四：检索验证

脚本**已经生成好了**：`D:\WorkBuddy\项目\输出文档\scripts\query_index.py`

<details>
<summary>点开看脚本内容（备份）</summary>

```python
# -*- coding: utf-8 -*-
"""本地规范库检索测试。"""
import os, sys
import chromadb
from sentence_transformers import SentenceTransformer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DB_DIR = r"D:\WorkBuddy\项目\输出文档\chroma_db"
MODEL  = "Qwen/Qwen3-Embedding-0.6B"   # 必须和 build_index.py 一致！
COLL   = "standards"
TOP_K  = 5


def fmt(meta, doc, dist, rank):
    print(f"\n[{rank}] 距离 {dist:.4f}")
    print(f"     《{meta['standard_name']}》{meta['standard']}  第 {meta['article_no']} 条"
          f"  ［{meta['force_status']}］")
    print(f"     {doc[:200]}")


def main():
    model  = SentenceTransformer(MODEL)
    client = chromadb.PersistentClient(path=DB_DIR)
    col    = client.get_collection(COLL)
    print(f"库内 {col.count()} 条，模型 {MODEL}")

    while True:
        q = input("\n提问（直接回车退出）> ").strip()
        if not q:
            break
        # 查询侧必须带指令前缀 —— 模型自带的 prompt_name="query"
        qv = model.encode([q], prompt_name="query",
                          normalize_embeddings=True).tolist()
        res = col.query(query_embeddings=qv, n_results=TOP_K,
                        include=["documents", "metadatas", "distances"])
        for rank, (doc, meta, dist) in enumerate(
                zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), 1):
            fmt(meta, doc, dist, rank)


if __name__ == "__main__":
    main()
```

</details>

**运行并测试：**

```powershell
& $PY "D:\WorkBuddy\项目\输出文档\scripts\query_index.py"
```

试这三个问题（分别验证正文、表格、术语）：

```
书库通道的最小净宽是多少
阅览室的照度标准值
什么是半地下室
```

**怎么判断好不好用：**

- ✅ 距离 **< 0.40** 且条文号对得上 → 检索正常
- ⚠️ 距离 **0.45～0.60** → 勉强召回，需要调优
- ❌ 距离 **> 0.65** 或答非所问 → 检查是不是忘了 `prompt_name="query"`

---

## 7. 目录与文件约定

| 路径 | 用途 |
|---|---|
| `输出文档\scripts\build_index.py` | 建索引（可重跑，幂等） |
| `输出文档\scripts\query_index.py` | 检索测试 |
| `输出文档\chroma_db\` | 向量库落盘位置（已加 `.gitignore`，**不进 Git**） |
| `C:\Users\A\.cache\huggingface\`（或 `$HF_HOME\hub`） | 模型缓存（不在工作区；可用 `HF_HOME` 改到 D 盘，见 4.3） |

**重跑规则**：改了模型或切分产物 → 重跑 `build_index.py` 即可，它会先 `delete_collection` 再重建，不会脏。

---

## 8. 常见报错对照表

| 报错 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: No module named 'chromadb'` | 用到了系统 Python，不是 venv 的 | 一律用 `& $PY xxx.py` 调用，别裸敲 `python` |
| `ModuleNotFoundError: No module named 'sentence_transformers'` | 同上，或第 3 步没装 | `& $PY -m pip install sentence-transformers` |
| 在 PowerShell 里跑了 `activate.bat`，看着没报错但 `python` 还是系统的 | `.bat` 在 PowerShell 里环境变量传不回来 | 别激活，直接用 `$PY` 绝对路径 |
| `OSError: We couldn't connect to 'https://huggingface.co'` | 没设镜像 | `$env:HF_ENDPOINT = "https://hf-mirror.com"` |
| `HTTP Error 403: Forbidden`（手动 urlopen 时） | hf-mirror 校验 User-Agent | 用 `huggingface_hub`，别自己拼 URL |
| `Expected metadata value to be a str, int, float or bool` | 元数据里有 `None` | 用脚本里的 `clean_meta()` |
| `UnicodeEncodeError: 'gbk' codec...` | Windows 控制台默认 GBK | 脚本开头的 `sys.stdout.reconfigure` |
| 集合里条数翻倍 | 没清空就重跑 | 脚本已含 `delete_collection`，别手工 `add` |
| 距离算出来怪怪的 | 建库时不是 cosine | 集合 metadata 必须 `{"hnsw:space": "cosine"}` |

---

## 9. 合规提醒

- 本文全流程 **100% 本地**：模型在本机、向量在本机、数据一步不上网。
- **唯二联网的两个动作**：① 从 PyPI 装依赖；② 从 hf-mirror 下模型权重。两者传的都是**公开的软件包/模型**，不含规范原文。
- 绝对不要为了"省事"换成 `text-embedding-v3` 之类的**云端 embedding API** —— 那等于把 1083 条规范正文完整上传。
- 向量库文件 `chroma_db\` 已在 `.gitignore` 里，**不要提交到 GitHub**。

---

## 10. 做完之后

你手上会有一个能按语义检索、带完整元数据（规范名 / 条文号 / 版本 / 强条状态 / 页码）的本地规范库。

**下一步的两个方向（本文不展开）：**
1. **接大模型出答案** —— 检索到片段后拼进 prompt 调 LLM。本地可接 Ollama，云端要注意合规。
2. **混合检索加召回** —— 现在只有向量一路；加一路 BM25 或 `where_document={"$contains": ...}` 做精确条文号匹配，效果会更稳。
