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

所以库目录默认 `D:\\WorkBuddy\\chroma_db`（**全 ASCII**，见 app/config.py 的
CHROMA_DIR），**不要**放回项目目录下面 —— 那条路径里「项目」「输出文档」都是中文。
``app/config.py`` 导入时就会做 ASCII 断言，路径不合格直接拒绝启动。

为什么拆成两段
--------------
CPU 上向量化 1083 条要 30~60 分钟。若「向量化 + 入库」连在一个流程里跑，
中途 Ctrl+C 或关窗口会让这 30~60 分钟白费；更糟的是 Chroma 会留下一个
半成品库 —— 向量写进了 sqlite，但 HNSW 索引文件（*.bin）没生成。这种库
**连打开都会报错**（Error loading hnsw index），只能整个删掉重建。

拆开之后：
  [vec] 读 chunks → 向量化 → 存 .npy + emb_meta.json     慢，但只做一次
  [db ] 校验 emb_meta → 写 staging 集合 → 校验 → 切换     秒级，可无限次重试

2026-11 索引一致性整改（P0-C）
------------------------------
· [db] 不再只比 len(embs)==len(rows)：还要比 emb_meta.json 里的 sha1 / model /
  dim / count / shape，任一不符直接非零退出并提示 ``--stage vec --force``。
  「chunk 数不变但正文改了一个字」只有 sha1 这一项能抓到。
· [db] 不再「先 delete 旧集合再 add」：改为 写 staging → 校验 → 切换 → 删旧，
  任一步失败旧集合原样保留，任一时刻都有一个完整可查的库。
· 库目录 / 集合名 / 模型名 / 语料路径一律来自 app/config.py（脚本里经
  ``_config.load_config()`` 取），本文件不再硬编码。

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
import subprocess
import sys
import time

from _config import load_config, project_root         # 必须最先导入：定 HF_HOME

_CFG_EARLY = load_config()

# ---------------------------------------------------------------------------
# 模型缓存位置：必须在 import numpy / torch 之前定下来
# （torch 一 import 就会读 HF_HOME；huggingface_hub 更是在 import 时锁定它）
# 重型依赖（numpy / torch / chromadb / sentence_transformers）一律按需导入，
# 这样「只做 emb_meta 门禁」的失败路径不必先装上几百 MB 的依赖。
# ---------------------------------------------------------------------------
MODEL = _CFG_EARLY.EMBED_MODEL
os.environ.setdefault("HF_HOME", _CFG_EARLY.HF_HOME)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from app import index_ops                              # noqa: E402
from app.config import Config                          # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")           # Windows 控制台默认 GBK
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

CFG = _CFG_EARLY
ENC_BATCH = 32      # 向量化的批大小
DB_BATCH  = int(os.environ.get("BCR_DB_BATCH", "256"))   # 写入 Chroma 的批大小
USE_HEADER = False  # True = 把「规范名 + 条文号」拼进待向量化文本

META_KEYS = ("standard", "standard_name", "version", "force_status", "status",
             "article_no", "chapter", "kind", "page_start", "page_end")


def _chromadb():
    """chromadb 按需导入：只做 emb_meta 门禁时不必装它。"""
    import chromadb as _c
    return _c


def _st():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer


def clean_meta(r):
    """Chroma 不接受 None，统一转成空串。"""
    return {k: ("" if r.get(k) is None else r.get(k)) for k in META_KEYS}


def read_chunks():
    return index_ops.read_chunks_file(CFG.CHUNKS_FILE)


# ---------------------------------------------------------------------------
# 第 1 段：向量化 → .npy + emb_meta.json
# ---------------------------------------------------------------------------
def stage_vec(rows, sha1, force=False):
    ok, why = index_ops.cache_state(CFG, rows, sha1)
    if ok and not force:
        print(f"[vec] 跳过 —— {why}")
        print(f"      缓存文件：{CFG.NPY_FILE}")
        return
    if force and os.path.exists(CFG.NPY_FILE):
        print("[vec] --force：忽略现有缓存，重新向量化")
    elif not ok:
        print(f"[vec] 缓存不可用（{why}）—— 重新向量化")

    os.makedirs(CFG.CACHE_DIR, exist_ok=True)
    import numpy as np                     # 重型依赖按需导入
    import torch
    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)

    print(f"[vec] 载入模型 {MODEL} …")
    print(f"      HF_HOME   = {os.environ.get('HF_HOME')}")
    print(f"      HF_ENDPOINT = {os.environ.get('HF_ENDPOINT')}")
    print(f"      期望维度 {CFG.EMBED_DIM}")
    t = time.time()
    model = _st()(MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"      模型就绪 {time.time() - t:.1f}s，实际维度 {dim}，"
          f"torch 线程数 {torch.get_num_threads()}")
    if dim != CFG.EMBED_DIM:
        raise SystemExit(
            f"[vec] 模型 {MODEL} 的维度是 {dim}，但配置 EMBED_DIM={CFG.EMBED_DIM}。\n"
            "      要么改 app/config.py / .env 里的 EMBED_DIM，要么换模型 ——\n"
            "      维度必须与检索端一致，不能只改一边。")

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

    embs = np.asarray(embs, dtype="float32")
    np.save(CFG.NPY_FILE, embs)
    # meta 最后写：中途死掉时 .npy 可能不完整，但没有 meta → 下次一定重算
    with open(CFG.META_FILE, "w", encoding="utf-8") as f:
        json.dump({"sha1": sha1, "count": len(rows), "model": MODEL,
                   "dim": int(embs.shape[1]), "shape": list(embs.shape),
                   "seconds": round(dt, 1),
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, ensure_ascii=False, indent=1)
    print(f"[vec] 完成 {dt / 60:.1f} 分钟，shape={list(embs.shape)}")
    print(f"      已存 {CFG.NPY_FILE}  （{os.path.getsize(CFG.NPY_FILE) / 1024 ** 2:.0f} MB）")
    print(f"      已存 {CFG.META_FILE}")


# ---------------------------------------------------------------------------
# 第 2 段：校验 emb_meta → 原子重建 Chroma 集合（两进程 + 改名链）
# ---------------------------------------------------------------------------
class WorkerError(RuntimeError):
    """后台库持有者（子进程）返回失败，或中途死掉。"""


class _Named(object):
    """list_collections() 的返回值只需要 .name。"""

    def __init__(self, name):
        self.name = name


class _WorkerCollection(object):
    """后台集合的只读句柄：index_ops 只用到 count()。"""

    def __init__(self, worker, name):
        self.worker = worker
        self.name = name

    def count(self):
        return int(self.worker.call("count", name=self.name)["count"])


class ChromaWorker(object):
    """后台库持有者：唯一长期占用库目录的 Chroma 客户端。

    为什么不能在本进程里直接建/改名（2026-11 实测）
    ----------------------------------------------
    Chroma 1.5.9 的客户端会缓存「某某名字不存在」：同一个客户端上先
    ``get_collection("staging-x")`` 抛 NotFoundError，再 ``create_collection("staging-x")``
    虽然能建出来，但之后按名字对它 ``modify(name=...)`` 会走那份脏缓存。
    根治办法是让「判断不存在」和「建集合」发生在两个客户端里，而客户端按路径复用
    （同一路径第二次 PersistentClient 拿到的还是同一个 system），同进程里分不干净。
    所以：后台子进程是**唯一**的库持有者，主进程每次只开一个新客户端做只读 + 改名。
    实测（probe_route.py）：子进程改名后，主进程同一个客户端 list_collections
    立刻看到新名字；按旧名取是 NotFoundError，按新名取元数据、条数都对。
    """

    is_worker = True

    def __init__(self, db_dir, root=None):
        env = dict(os.environ)
        env.update({"PYTHONPATH": root or project_root(), "PYTHONIOENCODING": "utf-8"})
        # worker 与 build_index.py 放在同一目录，随脚本一起入库；
        # 不再依赖 tests/（tests 可能不随生产包发布）。
        worker_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "chroma_worker.py")
        if not os.path.exists(worker_py):
            raise WorkerError(f"找不到后台库持有者：{worker_py}")
        self.proc = subprocess.Popen(
            [sys.executable, worker_py, db_dir], cwd=env["PYTHONPATH"], env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        hello = self._read()
        if not hello.get("ok"):
            raise WorkerError(f"后台库持有者启动失败：{hello}")

    # -- 传输 --
    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise WorkerError("后台库持有者意外退出（stdout 已关闭）")
        return json.loads(line)

    def call(self, op, progress=None, **kwargs):
        req = dict(kwargs)
        req["op"] = op
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        while True:
            msg = self._read()
            if msg.get("op") == "progress":
                if progress:
                    progress(msg)
                continue
            if not msg.get("ok"):
                raise WorkerError(f"{msg.get('error')}")
            return msg

    # -- client 兼容层（index_ops 只用到这几个） --
    def list_collections(self):
        return [_Named(n) for n in self.call("names")["names"]]

    def delete_collection(self, name):
        self.call("delete", name=name)

    def get_collection(self, name):
        return _WorkerCollection(self, name)

    def rename(self, old, new):
        self.call("rename", old=old, new=new)

    # -- 收尾 --
    def close(self):
        try:
            self.call("quit", timeout_ok=True)
        except Exception:                        # noqa: BLE001
            pass
        try:
            self.proc.stdin.close()
        except Exception:                        # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=30)
        except Exception:                        # noqa: BLE001
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _age(path, seconds):
    """把 mtime 往前拨 —— Chroma 的 .bin 可能是一分钟前才 rename 到位的，
    目录 mtime 比 .bin 旧，单看目录 mtime 会误判两次 rename 太靠近。"""
    if seconds <= 0:
        return
    try:
        t = time.time() - seconds
        os.utime(path, (t, t))
        for p in glob.glob(os.path.join(glob.escape(path), "*")):
            os.utime(p, (t, t))
    except OSError:
        pass


def make_swapper(worker, log=print):
    """切换器：S1 旧名 → 归档名，S2 staging → 正式名；两步之间夹一道隔离窗口。

    为什么两步之间要等：Chroma 会在「上一笔写入转成持久索引」后才把新集合的 .bin
    rename 到位（实测：.bin 的 mtime 比目录 mtime 晚约 1 分钟）。如果 S2 紧跟 S1，
    两个新集合的落盘可能撞在同一分钟里。等待期间库里仍有完整的 archive-*（旧）与
    staging-*（新），所以**这个等待不产生任何空窗**。
    """
    gap = float(os.environ.get("BCR_SWAP_GAP_S", "70"))
    renamed = {"old": False}

    def progress_ignore(_msg):
        return None

    def swap():
        collection = CFG.COLLECTION
        names = worker.call("names", progress=progress_ignore)["names"]
        had_old = collection in names
        staging = [n for n in names if n.startswith(f"staging-{collection}-")]
        if len(staging) != 1:
            raise WorkerError(f"切换前应有且只有一个 staging，实际 {staging}")
        staging = staging[0]
        archive = f"archive-{collection}-" + staging.split("-")[-1]
        if had_old:
            _age(os.path.join(CFG.CHROMA_DIR, _seg_dir_of(collection)), gap)
            log(f"[db] [S1] `{collection}` → `{archive}`（改名，内容不变；旧库仍完整）")
            worker.call("rename", old=collection, new=archive)
            renamed["old"] = True
            if os.environ.get("BCR_FAKE_FAIL_RENAMED"):
                raise WorkerError("[故障注入] 旧集合已改名，S2 之前中断")
            log(f"[db] S1 与 S2 之间等待 {gap:.0f}s（隔离两次落盘；此期间库里有两个完整集合）")
            time.sleep(gap)
        else:
            log("[db] 首次建库：没有旧集合需要归档")
        log(f"[db] [S2] `{staging}` → `{collection}`（改名，内容不变）")
        worker.call("rename", old=staging, new=collection)
        return had_old, archive

    swap.info = renamed                  # 让调用方知道旧集合是否已被改名
    return swap


def _seg_dir_of(collection):
    """当前集合的 segment 目录名（找不到返回 ''，_age 会当成不存在跳过）。"""
    sqlite_path = os.path.join(CFG.CHROMA_DIR, "chroma.sqlite3")
    cid = index_ops.collection_id_of(sqlite_path, collection)
    if not cid:
        return ""
    cands = index_ops.segment_dirs_for_collection(
        cid, index_ops.read_segments(sqlite_path))
    d, _missing = index_ops.find_bins_dir(CFG.CHROMA_DIR, cands)
    return d or ""


def stage_db(rows, sha1):
    # ---------- 1) 同源门禁（先查缓存，再碰库） ----------
    try:
        meta = index_ops.load_emb_meta(CFG.META_FILE)
        problems = index_ops.validate_emb_meta(meta, rows, sha1, CFG)
    except index_ops.EmbMetaError as e:
        problems, meta = e.problems, None
    if problems:
        print("[db] ❌ 向量缓存与正文不同源，拒绝入库：", file=sys.stderr)
        for p in problems:
            print(f"      · {p}", file=sys.stderr)
        print("      " + index_ops.EmbMetaError(problems).rebuild_hint.replace(
            "\n", "\n      "), file=sys.stderr)
        raise SystemExit(3)
    try:
        embs, info = index_ops.load_embeddings(meta, rows, sha1, CFG)
    except index_ops.EmbMetaError as e:
        print("[db] ❌ 向量缓存不完整 / 与正文不同源，拒绝入库：", file=sys.stderr)
        for p in e.problems:
            print(f"      · {p}", file=sys.stderr)
        print("      " + e.rebuild_hint.replace("\n", "\n      "), file=sys.stderr)
        raise SystemExit(3)
    print(f"[db] 向量缓存校验通过：sha1={info['sha1'][:12]}… "
          f"model={info['model']} shape={info['shape']}")

    # ---------- 2) 旧集合的模型校验 + 上次中断的收尾（都在后台进程里做） ----------
    wait_s = float(os.environ.get("BCR_BUILD_WAIT_S", "180"))
    worker = None
    try:
        worker = ChromaWorker(CFG.CHROMA_DIR)
    except WorkerError as e:
        print(f"[db] ❌ 起不了后台库持有者：{e}", file=sys.stderr)
        raise SystemExit(4)

    try:
        names = worker.call("names")["names"]
        acted = worker.call("cleanup", collection=CFG.COLLECTION)
        if acted.get("acted"):
            print(f"[db] 上次中断的收尾：{acted['acted']}")
        names = acted["names"]
        if CFG.COLLECTION in names:
            meta_old = _chromadb().PersistentClient(
                path=CFG.CHROMA_DIR).get_collection(CFG.COLLECTION).metadata
            index_ops.assert_collection_model(
                meta_old, CFG, where=f"旧集合 `{CFG.COLLECTION}`")
            print(f"[db] 旧集合模型 {index_ops.collection_model(meta_old)!r} —— 一致")
        else:
            print("[db] 没有同名旧集合，按首次建库处理")
        print(f"[db] 当前集合：{names}")

        # ---------- 3) 灌新集合 + 校验 + 落盘（全在 atomic_rebuild 内完成） ----------
        meta = index_ops.collection_meta(CFG, sha1=sha1)

        def wait_bins_fn(name):
            return index_ops.wait_bins_db(CFG.CHROMA_DIR, name, timeout_s=wait_s)

        # ---------- 4) 灌数据 + 切换（只改名） ----------
        swap = make_swapper(worker)
        try:
            index_ops.atomic_rebuild(
                worker, CFG, rows, embs, clean_meta, meta=meta,
                db_batch=DB_BATCH, verify_sample=1, expect_dim=CFG.EMBED_DIM,
                swap=swap, wait_bins_fn=wait_bins_fn, log=print,
                fail_at_batch=(int(os.environ["BCR_FAKE_FAIL_ADD_AT"])
                               if os.environ.get("BCR_FAKE_FAIL_ADD_AT") else None))
        except index_ops.AtomicRebuildError as e:
            print(f"[db] ❌ {e}", file=sys.stderr)
            raise SystemExit(4)

        # ---------- 5) 终检（归档已由 atomic_rebuild 清理） ----------
        n_final = worker.call("count", name=CFG.COLLECTION)["count"]
        if n_final != len(rows):
            print(f"[db] ❌ 切换后 `{CFG.COLLECTION}` 条数 {n_final} ≠ {len(rows)}",
                  file=sys.stderr)
            raise SystemExit(4)
    finally:
        if worker is not None:
            worker.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["vec", "db", "all"], default="all")
    ap.add_argument("--force", action="store_true",
                    help="vec: 忽略缓存重新向量化")
    ap.add_argument("--chunks", default=None, help="覆盖语料文件（默认取配置）")
    ap.add_argument("--cache-dir", default=None, help="覆盖向量缓存目录（默认取配置）")
    ap.add_argument("--emb-model", default=None, help="覆盖 EMBED_MODEL（须与缓存里的模型一致，否则拒绝）")
    ap.add_argument("--embed-dim", default=None, help="覆盖 EMBED_DIM")
    ap.add_argument("--db-dir", default=None, help="覆盖库目录（默认取配置，须纯 ASCII）")
    ap.add_argument("--collection", default=None, help="覆盖集合名（默认取配置）")
    args = ap.parse_args()

    global CFG
    CFG = Config(chunks_file=args.chunks, cache_dir=args.cache_dir,
                 chroma_dir=args.db_dir, collection=args.collection)
    if args.cache_dir:
        CFG.NPY_FILE = os.path.join(CFG.CACHE_DIR,
                                    os.path.basename(_CFG_EARLY.NPY_FILE))
        CFG.META_FILE = os.path.join(CFG.CACHE_DIR,
                                     os.path.basename(_CFG_EARLY.META_FILE))
    from app.config import assert_ascii_path
    assert_ascii_path(CFG.CHROMA_DIR)

    if not os.path.exists(CFG.CHUNKS_FILE):
        raise SystemExit(f"找不到数据文件：{CFG.CHUNKS_FILE}")
    rows, sha1 = read_chunks()
    print("=" * 64)
    print("  建库 / 重建本地 Chroma 向量库")
    print("=" * 64)
    print(f"  数据：{CFG.CHUNKS_FILE}")
    print(f"        {len(rows)} 条，sha1={sha1[:12]}…")
    print(f"  模型：{MODEL}（维度 {CFG.EMBED_DIM}）")
    print(f"  库目录：{CFG.CHROMA_DIR}   集合：{CFG.COLLECTION}")
    print(f"  缓存：{CFG.CACHE_DIR}")
    print()

    if args.stage in ("vec", "all"):
        stage_vec(rows, sha1, force=args.force)
        print()
    if args.stage in ("db", "all"):
        stage_db(rows, sha1)
        print()
        print("-" * 64)
        print("  下一步：自检（另开一条命令，几秒出结果）")
        print(f'    & $PY "{os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_index.py")}"')


if __name__ == "__main__":
    main()
