# -*- coding: utf-8 -*-
"""把 chunks_v5 的 1083 条规范灌进本地 Chroma 向量库（两段式）。

⚠️⚠️ 中文路径陷阱（2026-09-13 实测确认，这是之前反复做出「半成品库」的真正原因）
--------------------------------------------------------------------------
chromadb 1.5.9 的 HNSW 索引由 **Rust 绑定**负责落盘，而它在**含非 ASCII 字符的
路径**下会**静默失败**：`chroma.sqlite3` 正常写入、segment 子目录也建出来了，
但那 4 个 HNSW 索引文件（header.bin / data_level0.bin / length.bin / link_lists.bin）
**一个都不写**，也不报错。结果就是一个「sqlite 里 1083 条齐全、但索引缺失」的库，
打开即 `Error loading hnsw index`，只能整个删掉重建。

实测对照（同一进程、同一段代码，只换父目录的字符）：

    D:\\WorkBuddy\\aaa\\ct      → 4 个 .bin  ✅
    D:\\WorkBuddy\\中中中\\ct    → 0 个 .bin  ❌
    D:\\WorkBuddy\\ct_ascii     → 4 个 .bin  ✅
    D:\\WorkBuddy\\项目\\ct_a    → 0 个 .bin  ❌
    %TEMP%\\ct1                → 4 个 .bin  ✅

所以库目录定在 `D:\\WorkBuddy\\chroma_db`（**全 ASCII**），
**不要**放回 `D:\\WorkBuddy\\项目\\输出文档\\` —— 那条路径里「项目」「输出文档」都是中文。
脚本开头有 `_assert_ascii_path()` 兜底，路径不合格会直接拒绝启动。

为什么拆成两段
--------------
CPU 上向量化 1083 条要 30~60 分钟。若「向量化 + 入库」连在一个流程里跑，
中途 Ctrl+C 或关窗口会让这 30~60 分钟白费；更糟的是 Chroma 会留下一个
半成品库 —— 向量写进了 sqlite，但 HNSW 索引文件（*.bin）没生成。这种库
**连打开都会报错**（Error loading hnsw index），只能整个删掉重建。

拆开之后：
  [vec] 读 chunks → 向量化 → 存 .npy               慢，但只做一次
  [db ] 读 .npy   → add() → 落盘 HNSW 索引         秒级，可无限次重试

用法（$PY 的定义见《本地部署操作指引.md》第 0 步）
  & $PY build_index.py                       # 两段一起跑（首次用这个）
  & $PY build_index.py --stage vec           # 只做向量化
  & $PY build_index.py --stage db            # 只做入库（缓存已存在时）
  & $PY build_index.py --stage vec --force   # 忽略缓存，强制重算
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# 模型缓存位置：必须在 import sentence_transformers 之前定下来
# （huggingface_hub 在 import 时就锁定了 HF_HOME）
# ---------------------------------------------------------------------------
MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _has_weights(hub_root):
    """hub 目录下是否已有 Qwen3-Embedding 的真实权重（>100 MB）。"""
    d = os.path.join(hub_root, "models--Qwen--Qwen3-Embedding-0.6B", "snapshots")
    if not os.path.isdir(d):
        return False
    for r, _, fs in os.walk(d):
        for f in fs:
            if f.endswith((".safetensors", ".bin")):
                try:
                    if os.path.getsize(os.path.join(r, f)) > 100 * 1024 * 1024:
                        return True
                except OSError:
                    pass
    return False


def _resolve_hf_home():
    """模型缓存根目录：环境变量 > 自动探测已有权重 > 默认 D 盘。"""
    if os.environ.get("HF_HOME"):
        return os.environ["HF_HOME"], "来自环境变量 HF_HOME"
    for c in (r"D:\WorkBuddy\models\hf",
              os.path.join(os.path.expanduser("~"), ".cache", "huggingface")):
        if _has_weights(os.path.join(c, "hub")):
            return c, "自动探测到已有权重"
    return r"D:\WorkBuddy\models\hf", "默认值（本机尚无权重，将尝试联网下载）"


HF_HOME, HF_HOME_SRC = _resolve_hf_home()
os.environ["HF_HOME"] = HF_HOME
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np                                     # noqa: E402
import torch                                           # noqa: E402
import chromadb                                        # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")           # Windows 控制台默认 GBK
except Exception:
    pass

# ============================ 配置区 ============================
PROJ      = r"D:\WorkBuddy\项目"
CHUNKS    = os.path.join(PROJ, r"输出文档\chunks_v5\chunks.jsonl")
CACHE_DIR = os.path.join(PROJ, r"输出文档\cache")
NPY       = os.path.join(CACHE_DIR, "emb_qwen3-embedding-0.6b.npy")
META      = os.path.join(CACHE_DIR, "emb_meta.json")
# ⚠️ 库目录必须是**纯 ASCII 路径**！见文件头「中文路径陷阱」。
DB_DIR    = r"D:\WorkBuddy\chroma_db"
COLL      = "standards"
ENC_BATCH = 32      # 向量化的批大小
DB_BATCH  = 256     # 写入 Chroma 的批大小
USE_HEADER = False  # True = 把「规范名 + 条文号」拼进待向量化文本
# ================================================================

META_KEYS = ("standard", "standard_name", "version", "force_status", "status",
             "article_no", "chapter", "kind", "page_start", "page_end")


def _has_non_ascii(s):
    return any(ord(c) > 127 for c in s)


def _assert_ascii_path(p):
    """库目录必须是纯 ASCII —— 否则 HNSW 索引不会落盘（见文件头「中文路径陷阱」）。"""
    bad = [c for c in p if ord(c) > 127]
    if bad:
        raise SystemExit(
            "\n[!] 库目录含非 ASCII 字符，直接拒绝启动：\n"
            f"      {p}\n"
            f"    问题字符：{''.join(sorted(set(bad)))}\n\n"
            "    chromadb 的 HNSW 索引在中文路径下**不会写 .bin**（静默失败），\n"
            "    最后必定得到一个打不开的半成品库。\n"
            "    请把 DB_DIR 改到纯 ASCII 路径，例如 D:\\WorkBuddy\\chroma_db\n")


def clean_meta(r):
    """Chroma 不接受 None，统一转成空串。"""
    return {k: ("" if r.get(k) is None else r.get(k)) for k in META_KEYS}


def read_chunks():
    raw = open(CHUNKS, "rb").read()
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    return rows, hashlib.sha1(raw).hexdigest()


def cache_state(rows, sha1):
    """返回 (能否复用, 原因)。"""
    if not (os.path.exists(NPY) and os.path.exists(META)):
        return False, "缓存不存在"
    try:
        m = json.load(open(META, encoding="utf-8"))
    except Exception as e:
        return False, f"缓存元信息损坏（{e}）"
    if m.get("sha1") != sha1:
        return False, "chunks 内容已变（sha1 不一致）"
    if m.get("model") != MODEL:
        return False, "模型不同"
    if m.get("count") != len(rows):
        return False, f"条数不同（缓存 {m.get('count')} / 现在 {len(rows)}）"
    return True, "命中缓存"


# ---------------------------------------------------------------------------
# 第 1 段：向量化 → .npy
# ---------------------------------------------------------------------------
def stage_vec(rows, sha1, force=False):
    ok, why = cache_state(rows, sha1)
    if ok and not force:
        print(f"[vec] 跳过 —— {why}")
        print(f"      缓存文件：{NPY}")
        return
    if force and os.path.exists(NPY):
        print("[vec] --force：忽略现有缓存，重新向量化")

    os.makedirs(CACHE_DIR, exist_ok=True)
    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)

    print(f"[vec] 载入模型 {MODEL} …")
    print(f"      HF_HOME   = {HF_HOME}  （{HF_HOME_SRC}）")
    print(f"      HF_ENDPOINT = {os.environ.get('HF_ENDPOINT')}")
    t = time.time()
    model = SentenceTransformer(MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"      模型就绪 {time.time() - t:.1f}s，维度 {dim}，"
          f"torch 线程数 {torch.get_num_threads()}")

    docs = [(f"{r['standard_name']} {r['article_no']}：{r['text']}"
             if USE_HEADER else r["text"]) for r in rows]
    print(f"[vec] 开始向量化 {len(docs)} 条（CPU 上约 30~60 分钟）")
    print("      ⚠️ 这一段最慢，请让它跑完，不要关窗口。中途中断不会损坏已有库，")
    print("         但向量不会保存，下次要重跑这一段。")
    t = time.time()
    try:
        embs = model.encode(docs, batch_size=ENC_BATCH,
                            normalize_embeddings=True,
                            show_progress_bar=True, convert_to_numpy=True)
    except KeyboardInterrupt:
        print("\n[vec] 已被中断 —— 向量未保存。已有向量库不受影响。")
        raise SystemExit(130)
    dt = time.time() - t

    np.save(NPY, np.asarray(embs, dtype="float32"))
    json.dump({"sha1": sha1, "count": len(rows), "model": MODEL,
               "dim": int(embs.shape[1]), "shape": list(embs.shape),
               "seconds": round(dt, 1),
               "created": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(META, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[vec] 完成 {dt / 60:.1f} 分钟，shape={embs.shape}")
    print(f"      已存 {NPY}  （{os.path.getsize(NPY) / 1024 ** 2:.0f} MB）")
    print(f"      已存 {META}")


# ---------------------------------------------------------------------------
# 第 2 段：读 .npy → 写入 Chroma
# ---------------------------------------------------------------------------
def stage_db(rows):
    _assert_ascii_path(DB_DIR)
    if not os.path.exists(NPY):
        raise SystemExit(f"[db] 找不到向量缓存：{NPY}\n"
                         f"     请先跑：& $PY build_index.py --stage vec")
    embs = np.load(NPY)
    if len(embs) != len(rows):
        raise SystemExit(f"[db] 缓存条数 {len(embs)} 与数据 {len(rows)} 不一致，"
                         f"请重新向量化：--stage vec --force")
    if embs.shape[1] != 1024:
        print(f"[db] 注意：向量维度为 {embs.shape[1]}（预期 1024）")

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.get_collection(COLL)
        client.delete_collection(COLL)
        print(f"[db] 已删除同名旧集合 `{COLL}`（避免重复累积）")
    except Exception:
        print(f"[db] 无同名旧集合，直接新建")
    col = client.get_or_create_collection(
        COLL, metadata={"hnsw:space": "cosine", "description": "建筑规范条文库"})

    docs = [r["text"] for r in rows]
    n_batch = (len(rows) + DB_BATCH - 1) // DB_BATCH
    print(f"[db] 开始写入 {len(rows)} 条，每批 {DB_BATCH}，共 {n_batch} 批")
    t = time.time()
    for b, i in enumerate(range(0, len(rows), DB_BATCH), 1):
        j = min(i + DB_BATCH, len(rows))
        col.add(ids=[r["id"] for r in rows[i:j]],
                documents=docs[i:j],
                embeddings=embs[i:j].tolist(),
                metadatas=[clean_meta(r) for r in rows[i:j]])
        print(f"      第 {b}/{n_batch} 批   {i + 1}~{j}   累计 {col.count()}")
    dt = time.time() - t
    print(f"[db] 写入完成 {dt:.1f}s，col.count() = {col.count()}")

    # ------------------------------------------------------------------
    # 等 HNSW 索引真正落盘再收工。
    # Chroma 把「压缩进 HNSW + 写 *.bin」交给后台线程做，add() 返回**不代表**
    # 索引已在磁盘上。若这时进程被杀（Ctrl+C / 关窗口 / 关终端），就会留下
    # 「sqlite 里 1083 条齐全、但没有 *.bin」的半成品库 —— 这种库连打开都报
    # `Error loading hnsw index`。所以这里主动轮询，等到 4 个 .bin 齐了才宣布成功。
    # ------------------------------------------------------------------
    need = ("header.bin", "data_level0.bin", "length.bin", "link_lists.bin")
    print("      等待 HNSW 索引落盘（最多等 180 秒）…")
    t = time.time()
    while time.time() - t < 180:
        if set(need) <= {os.path.basename(p)
                         for p in glob.glob(os.path.join(DB_DIR, "*", "*.bin"))}:
            break
        time.sleep(1)

    bins = sorted(glob.glob(os.path.join(DB_DIR, "*", "*.bin")))
    names = {os.path.basename(p) for p in bins}
    if set(need) <= names:
        total = sum(os.path.getsize(p) for p in bins)
        print(f"      ✅ 索引已落盘：{len(bins)} 个 .bin，共 {total / 1024 ** 2:.1f} MB"
              f"（等了 {time.time() - t:.0f}s）")
        print("      ✅ 入库阶段结束。现在可以关窗口了（本进程会自己退出）。")
    else:
        print(f"      ⚠️ 等了 180 秒，.bin 仍只有 {len(bins)} 个：{sorted(names) or '无'}")
        if _has_non_ascii(DB_DIR):
            print("         原因几乎肯定是：库目录含非 ASCII 字符（中文路径）——")
            print("         chromadb 的 Rust HNSW 在中文路径下不写 .bin。")
            print(f"         当前目录：{DB_DIR}")
            print("         改到 D:\\WorkBuddy\\chroma_db 之类的纯 ASCII 路径再跑。")
        else:
            print("         索引没落盘 → 这个库打不开。请重跑：--stage db")
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["vec", "db", "all"], default="all")
    ap.add_argument("--force", action="store_true",
                    help="vec: 忽略缓存重新向量化")
    args = ap.parse_args()

    if not os.path.exists(CHUNKS):
        raise SystemExit(f"找不到数据文件：{CHUNKS}")
    _assert_ascii_path(DB_DIR)
    rows, sha1 = read_chunks()
    print("=" * 64)
    print("  建库 / 重建本地 Chroma 向量库")
    print("=" * 64)
    print(f"  数据：{CHUNKS}")
    print(f"        {len(rows)} 条，sha1={sha1[:12]}…")
    print(f"  模型：{MODEL}")
    print(f"  库目录：{DB_DIR}")
    print()

    if args.stage in ("vec", "all"):
        stage_vec(rows, sha1, force=args.force)
        print()
    if args.stage in ("db", "all"):
        stage_db(rows)
        print()
        print("-" * 64)
        print("  下一步：自检（另开一条命令，几秒出结果）")
        print(f'    & $PY "{os.path.join(PROJ, r"输出文档\scripts\verify_index.py")}"')


if __name__ == "__main__":
    main()
