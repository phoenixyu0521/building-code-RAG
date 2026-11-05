# -*- coding: utf-8 -*-
"""索引一致性核心逻辑 —— 纯逻辑，**不 import chromadb / sentence_transformers**。

为什么单独一个模块
------------------
建库脚本（build_index.py）、自检脚本（verify_index.py）与检索端（retriever）需要
同一套判据：向量缓存与正文是否同源、集合的向量模型是否与当前配置一致、某个集合的
HNSW 索引文件在哪个目录、重建时怎么切换才不出现空窗。判据如果只写在脚本里，
测试就没法在不装 chromadb 的机器上跑。所以判据放这里，脚本只负责打印与退出码。

对应审查问题（P0-C v2）
-----------------------
问题 3（向量与正文不同源）：``validate_emb_meta`` + ``load_embeddings``
问题 4（配置散落各处）：集合元数据里的 embedding 模型 → ``assert_collection_model``
问题 5（先删后建不原子）：``atomic_rebuild``（改名链，见该函数的文档字符串）
问题 8（自检不过退出码仍为 0）：``summary_exit_code``

v2 相对 v1 的三处修正
---------------------
1. 切换不再 delete → 灌数据 → 建，而是 **两进程 + 改名链**：
   后台客户端唯一持有库，staging 灌满并校验后，只做两次改名（旧名 → 归档名、
   staging → 正式名），全程不删任何集合、不搬任何数据。
2. .bin 不再用「集合 id 当目录名」去找 —— Chroma 的目录名是 **segment id**，
   与集合 id 不同。现在按 Chroma 自己的规则（``segments`` 表里的向量段 +1）算，
   并支持「小于 sync_threshold 时先落一个临时目录再挪到位」的情况。
3. 去掉「旧集合条数必须等于新条数」的校验：合法缩容（1083 → 972）是同源判断之外的
   事，同源只认 chunks sha1 + emb_meta。
"""
import glob
import hashlib
import json
import os
import time

# ---------------------------------------------------------------------------
# 向量缓存（.npy + emb_meta.json）
# ---------------------------------------------------------------------------
# emb_meta.json 里必须有的字段；任一缺失都算缓存不完整（不猜默认值）。
META_REQUIRED = ("sha1", "model", "dim", "count", "shape")


class EmbMetaError(Exception):
    """向量缓存与正文不同源（或缓存不完整）。

    属性 problems 是问题清单，脚本直接打印；tests 直接断言。
    """

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("向量缓存与正文不同源：\n  - " + "\n  - ".join(self.problems))

    @property
    def rebuild_hint(self):
        return ("请重新向量化（会重算全部向量，CPU 上约 30~60 分钟）：\n"
                "    python build_index.py --stage vec --force")


def read_chunks_file(path):
    """读 chunks.jsonl，返回 (rows, sha1)。sha1 是**文件原始字节**的摘要。"""
    with open(path, "rb") as f:
        raw = f.read()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()
            if line.strip()]
    return rows, hashlib.sha1(raw).hexdigest()


def validate_emb_meta(meta, rows, sha1, cfg):
    """校验「向量缓存」是否就是「这批正文」的向量。

    返回问题清单（list[str]）；空列表 = 通过。逐项都是**硬门禁**：
      · meta 缺失 / 非 dict / JSON 坏了
      · 必需字段 sha1 / model / dim / count / shape 缺任何一个
      · sha1 ≠ 当前 chunks 文件摘要（「chunk 数不变但正文改了一个字」只有这一项能抓到）
      · model ≠ 当前配置的 EMBED_MODEL
      · count ≠ 当前行数；shape 与 (count, dim) 不自洽
      · dim ≠ 当前配置的 EMBED_DIM

    注意：这里比的是「缓存 vs 当前正文」，**没有**、也不该有「旧集合条数」这一项 ——
    合法缩容（删掉节标题一类）会让条数变化，但同源与否只由 sha1 + model 决定。
    """
    problems = []
    if meta is None:
        return ["emb_meta.json 缺失 —— 无法证明向量与正文同源"
                "（缓存可能是旧版脚本写的，或从未向量化过）"]
    if not isinstance(meta, dict):
        return [f"emb_meta.json 内容不是一个 JSON 对象（实际是 {type(meta).__name__}）"]

    missing = [k for k in META_REQUIRED if k not in meta]
    if missing:
        problems.append(f"emb_meta.json 缺字段：{', '.join(missing)}")
        return problems                       # 缺字段时后面的比较没有意义

    if meta["sha1"] != sha1:
        problems.append(
            f"chunks 正文已变：emb_meta 记录 sha1={str(meta['sha1'])[:12]}…，"
            f"当前文件 sha1={sha1[:12]}…（条数可能没变，但内容不是同一批）")

    if meta["model"] != cfg.EMBED_MODEL:
        problems.append(
            f"向量模型不一致：emb_meta 记录 {meta['model']!r}，"
            f"当前配置 EMBED_MODEL={cfg.EMBED_MODEL!r}（换模型 = 换向量空间）")

    n_rows = len(rows)
    if meta["count"] != n_rows:
        problems.append(
            f"条数不一致：emb_meta 记录 {meta['count']}，当前正文 {n_rows} 条")

    shape = meta["shape"]
    dim = meta["dim"]
    if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
        problems.append(f"shape 不是二元数组：{shape!r}")
        shape = None
    if not isinstance(dim, int) or isinstance(dim, bool):
        problems.append(f"dim 不是整数：{dim!r}")
        dim = None
    if shape is not None and dim is not None:
        if list(shape) != [meta["count"], dim]:
            problems.append(
                f"shape {list(shape)} 与 count={meta['count']} / dim={dim} 不自洽")
    if dim is not None and dim != cfg.EMBED_DIM:
        problems.append(
            f"向量维度不一致：emb_meta 记录 {dim}，当前配置 EMBED_DIM={cfg.EMBED_DIM}")
    return problems


def load_emb_meta(path):
    """读 emb_meta.json；文件不存在返回 None，内容坏了抛 EmbMetaError。"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:                    # noqa: BLE001 —— 任何解析失败都算坏缓存
        raise EmbMetaError([f"emb_meta.json 无法解析：{type(e).__name__}: {e}"])


def _import_numpy():
    import numpy as np
    return np


def load_embeddings(meta, rows, sha1, cfg):
    """校验缓存并载入 .npy，返回 (embs, info)。

    任一检查不过都抛 EmbMetaError（消息里带 ``--stage vec --force`` 提示）。
    成功时 info 是给日志用的摘要 dict。
    """
    problems = validate_emb_meta(meta, rows, sha1, cfg)
    if problems:
        raise EmbMetaError(problems)
    np = _import_numpy()
    if not os.path.exists(cfg.NPY_FILE):
        raise EmbMetaError([f"{cfg.NPY_FILE} 不存在，但 emb_meta.json 在 —— 缓存不完整"])
    embs = np.load(cfg.NPY_FILE)
    if embs.ndim != 2 or list(embs.shape) != list(meta["shape"]):
        raise EmbMetaError([
            f"{os.path.basename(cfg.NPY_FILE)} 实际 shape={list(embs.shape)}，"
            f"但 emb_meta 记录 {meta['shape']} —— 两者对不上"])
    if embs.shape[0] != len(rows):
        raise EmbMetaError([
            f"向量条数 {embs.shape[0]} 与正文 {len(rows)} 条不一致"])
    return embs, {"shape": list(embs.shape), "dim": int(embs.shape[1]),
                  "count": int(embs.shape[0]), "model": meta["model"],
                  "sha1": meta["sha1"]}


def cache_state(cfg, rows, sha1):
    """vec 阶段用：能否复用缓存。返回 (能否复用, 原因)。"""
    if not (os.path.exists(cfg.NPY_FILE) and os.path.exists(cfg.META_FILE)):
        return False, "缓存不存在"
    try:
        meta = load_emb_meta(cfg.META_FILE)
    except EmbMetaError as e:
        return False, "；".join(e.problems)
    problems = validate_emb_meta(meta, rows, sha1, cfg)
    if problems:
        return False, "；".join(problems)
    return True, "命中缓存（sha1 / model / count / dim / shape 全一致）"


# ---------------------------------------------------------------------------
# 集合元数据（问题 4：建库/启动时校验，不许静默）
# ---------------------------------------------------------------------------
MODEL_META_KEY = "embedding_model"
DIM_META_KEY = "embedding_dim"
SHA1_META_KEY = "chunks_sha1"


def collection_meta(cfg, sha1=None, description="建筑规范条文库"):
    """写进 Chroma collection 的元数据。

    ``chunks_sha1`` 记录「这批向量对应的正文」，检索端/自检可以用它判断
    集合是不是当前 chunks 建的（合法缩容时它也会跟着变，这是有意的）。
    """
    meta = {"hnsw:space": "cosine",
            "description": description,
            MODEL_META_KEY: cfg.EMBED_MODEL,
            DIM_META_KEY: cfg.EMBED_DIM}
    if sha1:
        meta[SHA1_META_KEY] = sha1
    return meta


def collection_model(meta):
    """从集合元数据里取 embedding 模型；没有就是 None（旧库）。"""
    if not meta:
        return None
    v = meta.get(MODEL_META_KEY)
    return None if v is None or str(v).strip() == "" else str(v)


def collection_problems(meta, cfg, where="集合", sha1=None):
    """集合元数据 vs 当前配置的差距清单（空 = 一致）。"""
    problems = []
    got = collection_model(meta)
    if got is None:
        problems.append(f"{where}的元数据里没有 {MODEL_META_KEY}，无法确认它是哪个模型建的")
    elif got != cfg.EMBED_MODEL:
        problems.append(f"{where}是用 {got!r} 建的，当前配置 EMBED_MODEL="
                        f"{cfg.EMBED_MODEL!r}（向量空间不同，检索结果对不上）")
    dim = (meta or {}).get(DIM_META_KEY)
    if dim is not None and int(dim) != cfg.EMBED_DIM:
        problems.append(f"{where}的维度是 {dim}，当前配置 EMBED_DIM={cfg.EMBED_DIM}")
    if sha1 is not None:
        got_sha = (meta or {}).get(SHA1_META_KEY)
        if got_sha and got_sha != sha1:
            problems.append(f"{where}对应的是另一版正文（{str(got_sha)[:12]}…），"
                            f"当前 chunks 是 {sha1[:12]}…")
    return problems


def assert_collection_model(meta, cfg, where="集合"):
    """集合的向量模型必须与当前配置一致，不一致抛 SystemExit（不静默降级）。"""
    problems = collection_problems(meta, cfg, where=where)
    if problems:
        raise SystemExit(
            "❌ " + "\n   ".join(problems) + "\n"
            "   → 二选一：\n"
            "     · 把配置改回建库时用的模型 / 维度（app/config.py 或 .env）；\n"
            "     · 或换模型后重新向量化并重建：--stage vec --force 再 --stage db。")


# ---------------------------------------------------------------------------
# HNSW 索引文件：按 Chroma 的规则找 segment 目录（问题：.bin 取错目录）
# ---------------------------------------------------------------------------
NEED_BINS = ("header.bin", "data_level0.bin", "length.bin", "link_lists.bin")


def _uuid_to_int(u):
    """Chroma Rust 侧 ``Uuid::as_u128`` 的十进制字符串形式（按大端拼字节）。"""
    return int(str(u).replace("-", ""), 16)


def vector_segment_for(segments):
    """从 ``[(segment_id, scope, collection_id), ...]`` 里挑出向量段。

    Chroma 给一个集合的向量段目录命名规则（``chroma/src/lib.rs``）：取该集合所有
    segment 的 uuid 整数之和，再加 1，按十进制打印。单向量段时就是
    ``int(segment) + 1`` —— 这就是为什么目录名看着像 uuid 却对不上任何 id。
    """
    vec = [(sid, coll) for sid, scope, coll in segments
           if str(scope).upper() == "VECTOR"]
    if not vec:
        vec = [(sid, coll) for sid, scope, coll in segments
               if str(scope).upper() != "METADATA"]
    total = 0
    for sid, _coll in vec:
        total += _uuid_to_int(sid)
    return str(total + 1)


def segment_dirs_for_collection(collection_id, segments):
    """某个集合的 segment 目录候选（按可能性排序）。

    segments: sqlite ``select id, scope, collection from segments`` 的结果
    实测 Chroma 1.5.9：目录名 = 向量段的 segment uuid 本身
    （``<db>/<segment_id>/header.bin``）。旧版按 ``uuid+1`` 推算的规则
    在本机对不上任何目录，会让 972/150 条的小集合永远「缺 4 个文件」。
    这里优先返回向量段的 uuid，再退回集合 id。
    """
    mine = [s for s in segments if s[2] == collection_id]
    if not mine:
        return [collection_id]
    vec = [sid for sid, scope, _coll in mine if str(scope).upper() == "VECTOR"]
    if vec:
        return [str(vec[0]), str(vec[0]) + "-tmp"]
    seg_id = vector_segment_for(mine)
    return [seg_id, seg_id + "-tmp"]


def collection_have_bins(db_dir, seg_dir, bins_by_dir=None):
    """判断某个 **segment 目录**里 4 个 HNSW 文件是否齐全。

    bins_by_dir: {目录名: [该目录下的 .bin 文件名, ...]}（不给就现场扫）
    只认传入的那一个目录 —— 别的集合（尤其是 staging / 归档）的同名文件不能让
    检查通过。
    """
    if bins_by_dir is None:
        bins_by_dir = scan_bins(db_dir)
    names = set(bins_by_dir.get(seg_dir) or ())
    missing = [n for n in NEED_BINS if n not in names]
    return (not missing), missing


def find_bins_dir(db_dir, candidates, bins_by_dir=None):
    """在候选目录里找第一个 4 个 .bin 齐全的；返回 (目录名|None, 缺失列表)。

    多个候选都齐时取第一个（主目录优先）。
    """
    if bins_by_dir is None:
        bins_by_dir = scan_bins(db_dir)
    first_missing = None
    for cand in candidates:
        ok, missing = collection_have_bins(db_dir, cand, bins_by_dir)
        if ok:
            return cand, []
        if first_missing is None:
            first_missing = missing
    return None, (first_missing or list(NEED_BINS))


def scan_bins(db_dir):
    """扫出 DB_DIR 下每个子目录里的 .bin，返回 {子目录名: [文件名...]}。

    只扫一层子目录、只列出文件名，不判断归属 —— 归属由调用方按 segment 解析。
    """
    out = {}
    if not os.path.isdir(db_dir):
        return out
    for p in glob.glob(os.path.join(glob.escape(db_dir), "*", "*.bin")):
        d = os.path.basename(os.path.dirname(p))
        out.setdefault(d, []).append(os.path.basename(p))
    return {k: sorted(v) for k, v in out.items()}


def read_segments(sqlite_path):
    """读 sqlite 的 segments 表（只读）。返回 [(id, scope, collection)]；库不存在返回 []。"""
    if not os.path.exists(sqlite_path):
        return []
    import sqlite3
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        return [(r[0], r[1], r[2]) for r in con.execute(
            "select id, scope, collection from segments")]
    finally:
        con.close()


def collection_id_of(sqlite_path, name):
    """按集合名查 id；没有返回 None。"""
    if not os.path.exists(sqlite_path):
        return None
    import sqlite3
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        row = con.execute("select id from collections where name = ?",
                          (name,)).fetchone()
        return row[0] if row else None
    finally:
        con.close()


def wait_bins_db(db_dir, collection_name, sqlite_path=None, timeout_s=180,
                 poll_s=1.0, sleep=None):
    """轮询等某个**集合**（按名字 → segment → 目录）的 4 个 .bin 落盘。

    返回 (是否齐全, 目录名|None, 缺失列表, 等待秒数)。
    不再拿集合 id 当目录名 —— 那是 v1 的错误（972 条永远「缺 4 个文件」）。
    """
    sleep = time.sleep if sleep is None else sleep
    sqlite_path = sqlite_path or os.path.join(db_dir, "chroma.sqlite3")
    t = time.time()
    missing = list(NEED_BINS)
    while True:
        cid = collection_id_of(sqlite_path, collection_name)
        if cid:
            cands = segment_dirs_for_collection(cid, read_segments(sqlite_path))
            d, missing = find_bins_dir(db_dir, cands)
            if d:
                return True, d, [], time.time() - t
        if time.time() - t >= timeout_s:
            return False, None, missing, time.time() - t
        sleep(poll_s)


# ---------------------------------------------------------------------------
# 原子重建（问题 5）—— 两进程 + 改名链
# ---------------------------------------------------------------------------
class AtomicRebuildError(RuntimeError):
    """重建失败；旧集合已被原样保留（切换开始后失败时是「归档名」，见日志）。"""


STAGING_PREFIX = "staging-"
ARCHIVE_PREFIX = "archive-"


def staging_name(collection, stamp=None):
    """staging 集合名。

    Chroma 要求集合名只能由 [a-zA-Z0-9._-] 组成、**不能以下划线开头**
    （下划线开头会被直接拒绝，实测：_staging_standards → InvalidArgumentError），
    所以这里用 ``staging-`` 前缀。
    """
    stamp = stamp or time.strftime("%Y%m%d%H%M%S")
    return f"{STAGING_PREFIX}{collection}-{stamp}"


def archive_name(collection, stamp=None):
    """切换时旧集合先改到的名字。"""
    stamp = stamp or time.strftime("%Y%m%d%H%M%S")
    return f"{ARCHIVE_PREFIX}{collection}-{stamp}"


def _as_lists(block):
    """numpy 切片 → list；测试里传 list 也照样可用。"""
    return block.tolist() if hasattr(block, "tolist") else [list(v) for v in block]


def _is_worker(client):
    """build_index 传入的是「后台库进程」而不是普通 Chroma client。"""
    return bool(getattr(client, "is_worker", False))


def _rows_payload(rows, clean_meta):
    """给后台进程的 JSON 载荷：只带 id / text / meta，向量单独传。"""
    return [{"id": r["id"], "text": r["text"], "meta": clean_meta(r)} for r in rows]


def write_collection(client, name, rows, embs, clean_meta, meta, db_batch=256,
                     log=print):
    """在 client 所在的库里建一个新集合并灌满（只新建，永不改已有集合）。"""
    col = client.create_collection(name, metadata=meta)
    n = len(rows)
    n_batch = (n + db_batch - 1) // db_batch
    log(f"[db] 写入新集合 `{name}`：{n} 条，每批 {db_batch}，共 {n_batch} 批")
    for b, i in enumerate(range(0, n, db_batch), 1):
        j = min(i + db_batch, n)
        col.add(ids=[r["id"] for r in rows[i:j]],
                documents=[r["text"] for r in rows[i:j]],
                embeddings=_as_lists(embs[i:j]),
                metadatas=[clean_meta(r) for r in rows[i:j]])
        log(f"      第 {b}/{n_batch} 批   {i + 1}~{j}   累计 {col.count()}")
    return col


def verify_collection(col, rows, embs, expect_dim, verify_sample=1):
    """校验一个新灌满的集合：条数 / 维度 / 抽样检索自命中。

    返回 (ok, detail)。**不比旧集合条数** —— 缩容是合法的。
    """
    ids = [r["id"] for r in rows]
    n = col.count()
    if n != len(ids):
        return False, f"条数 {n} ≠ 正文 {len(ids)} 条"
    got = col.get(ids=[ids[0]], include=["embeddings"])
    vecs = (got or {}).get("embeddings")
    dim = len(vecs[0]) if vecs is not None and len(vecs) else None
    if dim != expect_dim:
        return False, f"向量维度 {dim} ≠ 期望 {expect_dim}"
    k = max(1, min(int(verify_sample), n))
    k = min(k, 10)
    hit = 0
    for idx in range(k):
        res = col.query(query_embeddings=_as_lists(embs[idx:idx + 1]),
                        n_results=1, include=["documents"])
        got_id = res["ids"][0][0] if res.get("ids") else None
        if got_id == ids[idx]:
            hit += 1
    if hit != k:
        return False, f"抽样检索 {k} 条只自命中 {hit} 条"
    return True, f"{n} 条，维度 {dim}，抽样 {k}/{k} 自命中"


def _rename(client, old, new):
    """把一个集合改名。真 Chroma 用 modify(name=...)；测试用的假客户端用 rename()。"""
    if hasattr(client, "rename"):
        client.rename(old, new)
        return
    client.get_collection(old).modify(name=new)


def _names(client):
    out = []
    for c in client.list_collections():
        out.append(c if isinstance(c, str) else getattr(c, "name", c))
    return out


def backup_dir_for(db_dir, stamp):
    return os.path.join(db_dir, "backup-" + stamp)


def _copy_bin_dirs(db_dir, dst, names, sqlite_path, log=print):
    """把若干集合的 HNSW 目录复制一份到 ``dst``（离线可读的兜底副本）。"""
    import shutil
    os.makedirs(dst, exist_ok=True)
    segs = read_segments(sqlite_path)
    copied = 0
    for n in names:
        cid = collection_id_of(sqlite_path, n)
        if not cid:
            continue
        cands = segment_dirs_for_collection(cid, segs)
        d, missing = find_bins_dir(db_dir, cands)
        if not d:
            log(f"      备份跳过 `{n}`（缺 {missing}）")
            continue
        target = os.path.join(dst, d)
        if os.path.isdir(target):
            continue
        shutil.copytree(os.path.join(db_dir, d), target)
        shutil.copy2(sqlite_path, os.path.join(dst, "chroma.sqlite3"))
        copied += 1
        log(f"      已备份 `{n}` 的索引目录 {d}")
    return copied


def atomic_rebuild(client, cfg, rows, embs, clean_meta, meta=None,
                   db_batch=256, verify_sample=1, expect_dim=None,
                   swap=None, wait_bins_fn=None, log=print, now=None,
                   keep_backup=True, fail_at_batch=None):
    """原子重建：**先灌一个完整的新集合，再只做改名切换**。

    空窗为什么不会出现（与 v1 的根本区别）
    ------------------------------------
    v1 的切换是 ``delete(旧) → create(新) → 重新 add 全部数据``：旧集合删掉之后、
    新集合灌完之前，库里没有可查的集合 —— 这一段就是空窗，期间任何失败都致命。

    这里把「灌数据」完全挪到切换之前，切换只剩改名：

        S0  旧 = `standards`（完整，在线可查）        新 = `staging-standards-…`（完整）
        S1  `standards`            → `archive-standards-…`   旧集合改名，内容不变
        S2  `staging-standards-…` → `standards`              新集合改名，内容不变

    任一时刻库里要么有一个叫 `standards` 的完整集合（S0 / S2 之后），要么有两个
    完整集合（S1 与 S2 之间）—— 没有「零个完整集合」的时刻，也没有复制 / 重灌
    数据的窗口。改名只是改元数据里的一行，不碰向量、不碰 HNSW 文件。

    失败怎么处理：
      · S1 之前失败：删掉 staging，旧集合的内容 / 名字都没被碰过。
      · S1 之后、S2 之前失败：把归档名改回原名（一次 rename）。
      · 进程被硬杀在 S1 与 S2 之间：库里仍有完整的 archive-*，
        ``cleanup_stale`` 会在下次启动时把它改回原名。

    ``swap`` 是「切换器」：默认在当前进程里直接改名；build_index 传进来的是
    「后台子进程」版本（Chroma 客户端会缓存「名字不存在」，同一客户端既查又建会
    把缓存弄脏，所以判断不存在和建集合必须分属两个客户端）。
    """
    collection = cfg.COLLECTION
    stamp = (now() if now is not None else time.strftime("%Y%m%d%H%M%S"))
    staging = staging_name(collection, stamp)
    archive = archive_name(collection, stamp)
    expect_dim = expect_dim or cfg.EMBED_DIM
    n_rows = len(rows)
    if len(embs) != n_rows:
        raise AtomicRebuildError(
            f"向量条数 {len(embs)} ≠ 正文 {n_rows} 条，拒绝开始重建（库里没有任何改动）")

    names = _names(client)
    had_old = collection in names
    log(f"[db] 目标集合 `{collection}`；新集合 `{staging}`"
        f"（{'旧集合在切换前不动' if had_old else '首次建库'}）")

    t0 = time.time()
    # ---------- 0) 备份（兜底副本，不参与切换） ----------
    if keep_backup and had_old:
        try:
            _copy_bin_dirs(cfg.CHROMA_DIR, backup_dir_for(cfg.CHROMA_DIR, stamp),
                           [collection],
                           os.path.join(cfg.CHROMA_DIR, "chroma.sqlite3"), log=log)
        except Exception as e:                       # noqa: BLE001
            log(f"[db] ⚠️ 备份失败（不影响原子性，真正的保证靠改名）："
                f"{type(e).__name__}: {e}")

    # ---------- 1) 灌新集合（旧集合全程不动） ----------
    try:
        if staging in _names(client):
            raise AtomicRebuildError(f"新集合名 `{staging}` 已存在，换个时间再跑")
        if _is_worker(client):
            # 后台进程是唯一的库持有者：写与校验都走 RPC，主进程不建集合。
            payload = _rows_payload(rows, clean_meta)
            client.call("write_collection", name=staging, rows=payload,
                        embs=_as_lists(embs),
                        meta=meta or collection_meta(cfg), batch=db_batch,
                        fail_at_batch=fail_at_batch)
            res = client.call("verify", name=staging, rows=payload,
                              embs=_as_lists(embs), expect_dim=expect_dim,
                              sample=verify_sample)
            ok, detail = bool(res.get("passed")), res.get("detail")
            col = None
        else:
            col = write_collection(client, staging, rows, embs, clean_meta,
                                   meta or collection_meta(cfg), db_batch=db_batch,
                                   log=log)
            ok, detail = verify_collection(col, rows, embs, expect_dim,
                                           verify_sample=verify_sample)
        if not ok:
            raise AtomicRebuildError(f"新集合自检不过（{detail}），拒绝切换")
        log(f"[db] 新集合校验通过：{detail}")
        if wait_bins_fn is not None:
            rok, d, missing, waited = wait_bins_fn(staging)
            if not rok:
                raise AtomicRebuildError(
                    f"新集合的 HNSW 索引 {waited:.0f}s 内没落盘（缺 {missing}）"
                    "（库目录是否含非 ASCII？），拒绝切换")
            log(f"[db] 索引已落盘：{d}（{waited:.0f}s）")
        del col
    except BaseException as exc:                     # noqa: BLE001 —— 包括 Ctrl+C
        log(f"[db] ❌ 准备阶段失败：{type(exc).__name__}: {exc}")
        try:
            if staging in _names(client):
                client.delete_collection(staging)
                log(f"[db] 已清理新集合 `{staging}`")
        except Exception as e:                       # noqa: BLE001
            log(f"[db] 清理 `{staging}` 失败（下次重建会自动清）：{type(e).__name__}: {e}")
        if had_old:
            log(f"[db] 旧集合 `{collection}` 原样保留，仍可检索")
        raise AtomicRebuildError(
            f"重建失败，旧库{'已保留（名字 / 内容都没动）' if had_old else '不存在（本次也没有留下半成品）'}："
            f"{type(exc).__name__}: {exc}") from exc

    # ---------- 2) 切换：改名链，失败就地回滚 ----------
    renamed_old = False
    if swap is None:
        def swap():
            nonlocal renamed_old
            if had_old:
                log(f"[db] [S1] `{collection}` → `{archive}`（改名，内容不变）")
                _rename(client, collection, archive)
                renamed_old = True
            log(f"[db] [S2] `{staging}` → `{collection}`（改名，内容不变）")
            _rename(client, staging, collection)
            return had_old, archive

    try:
        swap()
    except BaseException as exc:                     # noqa: BLE001
        log(f"[db] ❌ 切换失败：{type(exc).__name__}: {exc}")
        restored = False
        try:
            names_now = _names(client)
            if had_old and archive in names_now and collection not in names_now:
                _rename(client, archive, collection)
                log(f"[db] 已把归档集合改名回 `{collection}`")
            restored = collection in _names(client)
        except Exception as e:                       # noqa: BLE001
            log(f"[db] ⚠️ 回滚改名失败：{type(e).__name__}: {e}"
                f"（旧集合仍完整，在 `{archive}` 名下；下次 build_index 会自动改回）")
        # 正式名已恢复时才删 staging：此时它只是没被引用的半成品，
        # 留着会让下次重建撞名，也不会有人查它。
        if restored or not had_old:
            try:
                if staging in _names(client):
                    client.delete_collection(staging)
                    log(f"[db] 已清理新集合 `{staging}`")
            except Exception as e:                   # noqa: BLE001
                log(f"[db] ⚠️ 清理 `{staging}` 失败（下次重建会自动清）："
                    f"{type(e).__name__}: {e}")
        raise AtomicRebuildError(
            f"切换失败，旧库内容完整（名字见日志）：{type(exc).__name__}: {exc}") from exc

    # ---------- 3) 删 staging / 归档 + 终检 ----------
    for name in _names(client):
        if name == staging or (name.startswith(f"{ARCHIVE_PREFIX}{collection}-")
                               and name != collection):
            try:
                client.delete_collection(name)
                log(f"[db] 已删除 `{name}`")
            except Exception as e:                   # noqa: BLE001
                log(f"[db] ⚠️ `{name}` 删除失败（不影响使用）：{type(e).__name__}: {e}")
    try:
        n_final = client.get_collection(collection).count()
    except Exception as e:                           # noqa: BLE001
        raise AtomicRebuildError(f"切换后取不到 `{collection}`：{e}") from e
    if n_final != n_rows:
        raise AtomicRebuildError(f"切换后 `{collection}` 条数 {n_final} ≠ {n_rows}")

    stats = {"active": collection, "staging": staging, "archive": archive,
             "count": n_final, "dim": expect_dim, "had_old": had_old,
             "seconds": round(time.time() - t0, 1)}
    log(f"[db] ✅ 切换完成：`{collection}` {n_final} 条，耗时 {stats['seconds']}s")
    return collection, stats


def recover_interrupted(client, collection, log=print):
    """修复「上次硬杀在 S1~S2 之间」的现场：把唯一的归档改回原名。

    只在「正式名不存在、且恰好一个 archive-<collection>-*」时才动手。
    """
    names = _names(client)
    if collection in names:
        return None
    arch = [n for n in names if n.startswith(f"{ARCHIVE_PREFIX}{collection}-")]
    if len(arch) == 1:
        if hasattr(client, 'rename'):              # 测试用的假客户端
            client.rename(arch[0], collection)
        else:
            _rename(client, arch[0], collection)
        log(f"[db] 发现上次中断留下的归档 `{arch[0]}`，已改名回 `{collection}`")
        return f"renamed {arch[0]} -> {collection}"
    if len(arch) > 1:
        log(f"[db] ⚠️ 发现多个归档 {arch}，无法自动判断哪个有效，请人工确认")
    return None


def cleanup_stale(client, collection, log=print):
    """重建开始前：收尾上次中断的现场，删掉没用的 staging / 归档。

    · 只有 archive-*、没有正式名 → 改回正式名（保住旧库）；
    · 正式名在 → 归档是上次没删干净的旧版本，删掉；
    · 没人用的 staging-<collection>-* → 删除。
    """
    acted = recover_interrupted(client, collection, log=log)
    if collection in _names(client):
        for a in [n for n in _names(client)
                  if n.startswith(f"{ARCHIVE_PREFIX}{collection}-")]:
            try:
                client.delete_collection(a)
                log(f"[db] 删除上次留下的归档 `{a}`")
            except Exception as e:                   # noqa: BLE001
                log(f"[db] 删除归档 `{a}` 失败：{type(e).__name__}: {e}")
    removed = []
    for n in _names(client):
        if n.startswith(f"{STAGING_PREFIX}{collection}-"):
            try:
                client.delete_collection(n)
                removed.append(n)
            except Exception as e:                   # noqa: BLE001
                log(f"[db] 删除 staging `{n}` 失败：{type(e).__name__}: {e}")
    if removed:
        log(f"[db] 清理了 {len(removed)} 个上次中断残留的 staging：{removed}")
    return acted, removed


# ---------------------------------------------------------------------------
# 自检退出码（问题 8）
# ---------------------------------------------------------------------------
def summary_exit_code(results):
    """results: [{"status": "PASS"/"FAIL"/"SKIP", ...}, ...]

    有 FAIL → 1；有 SKIP（因为前面 FAIL 而跳过）也算失败，因为门禁没跑全 ——
    例如库目录不存在时后面全 SKIP，那种库同样不能用。
    """
    for r in results:
        if r.get("status") in ("FAIL", "SKIP"):
            return 1
    return 0
