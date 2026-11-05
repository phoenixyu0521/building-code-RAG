# -*- coding: utf-8 -*-
"""本地 Chroma 库自检 —— 一条命令给出全部验收判据，坏库**退出码非零**。

设计要点
--------
1. **不需要载 embedding 模型**。索引是否完好，用一个随机向量探测一次 query 就知道
   （能返回结果 = HNSW 索引可读），几秒出结果。
2. **FAIL 即门禁失败**：任一检查 FAIL（或因前面 FAIL 而 SKIP）→ ``sys.exit(1)``，
   可以直接串进 CI / 部署脚本。
3. **检查项绑定当前集合**：HNSW 的 .bin 只认**当前集合自己的** segment 目录，
   不再 ``DB_DIR/*/*.bin`` 全局 glob —— 别的集合（例如上次重建留下的 staging）
   有同名文件不能让检查通过。集合名 / 库目录 / 模型名一律从 app/config.py 取。
4. ``--json`` 输出机器可读摘要（stdout 只有 JSON，人看的表格走 stderr）。

用法（$PY 的定义见《本地部署操作指引.md》第 0 步）
  & $PY verify_index.py                 # 人看的报告；坏库退出码 1
  & $PY verify_index.py --json          # 机器可读摘要
"""
import argparse
import json
import os
import sqlite3
import sys

from _config import load_config

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from app import index_ops          # noqa: E402
from app.config import Config      # noqa: E402

CFG = load_config()
N_CHECKS = 9

_results = []          # [{"no":int,"name":str,"status":str,"detail":str}]


def _w(s):
    """按终端显示宽度算长度（中文算 2，ASCII 算 1）。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def record(no, name, status, detail=""):
    _results.append({"no": no, "name": name, "status": status, "detail": detail})
    dot = "." * max(2, 42 - _w(name))
    label = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
    print(f"  [{no:>2}/{N_CHECKS}] {name} {dot} {label}  {detail}", file=sys.stderr)


def check_ascii():
    bad = sorted({c for c in CFG.CHROMA_DIR if ord(c) > 127})
    if bad:
        record(1, "库目录为纯 ASCII", "FAIL",
               f"含 {''.join(bad)} —— chromadb 在中文路径下不写 .bin")
        return False
    record(1, "库目录为纯 ASCII", "PASS", CFG.CHROMA_DIR)
    return True


def check_dir_and_sqlite():
    if not os.path.isdir(CFG.CHROMA_DIR):
        record(2, "数据库目录存在", "FAIL", f"{CFG.CHROMA_DIR} 不存在 —— 还没建过库？")
        return False
    record(2, "数据库目录存在", "PASS")
    sqlite_path = os.path.join(CFG.CHROMA_DIR, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        record(3, "chroma.sqlite3 存在", "FAIL", "主文件缺失")
        return False
    record(3, "chroma.sqlite3 存在", "PASS",
           f"{os.path.getsize(sqlite_path) / 1024 ** 2:.1f} MB")
    return True


def collection_ids(sqlite_path):
    """从 sqlite 读「集合名 → id」。返回 ({name: id}, 错误说明)。"""
    try:
        con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        cur = con.cursor()
        cur.execute("select id, name from collections")
        rows = cur.fetchall()
        con.close()
    except Exception as e:                       # noqa: BLE001
        return {}, f"{type(e).__name__}: {e}"
    return {name: cid for cid, name in rows}, None


def segment_dir_for(sqlite_path, coll_id):
    """当前集合的 HNSW 目录候选（按 Chroma 的命名规则：向量段 uuid 之和 + 1）。

    目录名**不是**集合 id，也不是 segment id —— Chroma 的 Rust 侧用
    ``sum(segment uuid as u128) + 1`` 给向量段建目录（见 app/index_ops.vector_segment_for）。
    所以这里先从 sqlite 的 segments 表把当前集合的向量段取出来，再算目录名；
    小于 sync_threshold 的集合还可能先搭一个 ``<dir>-tmp``，两个都认。
    """
    try:
        segs = index_ops.read_segments(sqlite_path)
    except Exception as e:                       # noqa: BLE001 —— 老库没有 segments 表
        return [coll_id], f"（读 segments 失败：{type(e).__name__}，按集合 id 找目录）"
    mine = [s for s in segs if s[2] == coll_id]
    if not mine:
        return [coll_id], "（sqlite 里没有该集合的 segment，按集合 id 找目录）"
    return index_ops.segment_dirs_for_collection(coll_id, segs), ""


def check_hnsw_bins(sqlite_path, coll_id):
    """第 4 项：**当前集合自己的** 向量段目录里 4 个 .bin 是否齐全。

    只认 sqlite 里属于当前集合的那个向量段（及其 -tmp 临时目录）——
    别的集合（staging / 归档 / 上一版）的同名文件不能让检查通过。
    """
    cands, note = segment_dir_for(sqlite_path, coll_id)
    scan = index_ops.scan_bins(CFG.CHROMA_DIR)
    d, missing = index_ops.find_bins_dir(CFG.CHROMA_DIR, cands, scan)
    if not d:
        others = sorted(k for k in scan if k not in cands)
        hint = ("；注意：别的目录里有 .bin（" + str(others) + "），"
                "但那不属于当前集合，不算数" if others else "")
        miss = ", ".join(missing)
        record(4, "HNSW 索引文件齐全（当前集合）", "FAIL",
               f"候选目录 {cands} 缺 {miss}{note}{hint}")
        return False
    n_bytes = sum(os.path.getsize(os.path.join(CFG.CHROMA_DIR, d, f))
                  for f in scan[d])
    record(4, "HNSW 索引文件齐全（当前集合）", "PASS",
           f"目录 {d}（segment 映射）：4 个，共 {n_bytes / 1024 ** 2:.1f} MB{note}")
    return True


def check_count_sqlite(sqlite_path, coll_id):
    """第 5 项：当前集合的向量条数（直接读 sqlite，不经过 chromadb）。"""
    try:
        con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        cur = con.cursor()
        cur.execute("select count(*) from embeddings e join segments s "
                    "on e.segment_id = s.id where s.collection = ?", (coll_id,))
        n = cur.fetchone()[0]
        con.close()
    except Exception as e:                       # noqa: BLE001 —— 表结构对不上
        record(5, "向量条数（sqlite）", "SKIP",
               f"读不了分段计数：{type(e).__name__}: {str(e)[:60]}")
        return None
    record(5, "向量条数（sqlite）", "PASS", f"{n} 条")
    return n


def check_open_and_model():
    """第 6、7 项：库能否打开、集合元数据里的模型是否与配置一致。"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CFG.CHROMA_DIR)
        col = client.get_collection(CFG.COLLECTION)
        record(6, "库能否打开", "PASS",
               f"集合 `{CFG.COLLECTION}` count() = {col.count()}")
    except Exception as e:                       # noqa: BLE001
        raw = str(e).replace("\n", " ")[:100]
        msg = f"HNSW 索引加载失败 —— {raw}" if "hnsw" in str(e).lower() else \
              f"{type(e).__name__}: {raw}"
        record(6, "库能否打开", "FAIL", msg)
        record(7, "集合向量模型与配置一致", "SKIP", "库打不开，跳过")
        return None

    got = index_ops.collection_model(col.metadata)
    if got is None:
        record(7, "集合向量模型与配置一致", "FAIL",
               f"集合元数据里没有 {index_ops.MODEL_META_KEY}（旧库，无法确认）")
    elif got != CFG.EMBED_MODEL:
        record(7, "集合向量模型与配置一致", "FAIL",
               f"集合是 {got!r} 建的，配置是 {CFG.EMBED_MODEL!r}")
    else:
        record(7, "集合向量模型与配置一致", "PASS", got)
    return col


def check_probe(col):
    """第 8 项：随机向量检索探测（不载模型）。"""
    if col is None:
        record(8, "检索探测", "SKIP", "库打不开，跳过")
        return
    try:
        import numpy as np
        q = np.random.rand(1, CFG.EMBED_DIM).astype("float32")
        q /= np.linalg.norm(q)
        res = col.query(query_embeddings=q.tolist(), n_results=3,
                        include=["documents", "metadatas", "distances"])
        ids = res["ids"][0]
        metas = res["metadatas"][0]
        sample = metas[0].get("article_no") if metas else ""
        record(8, "检索探测", "PASS", f"返回 {len(ids)} 条，例如 {ids[0]}（{sample}）")
    except Exception as e:                       # noqa: BLE001
        record(8, "检索探测", "FAIL",
               f"{type(e).__name__}: {str(e).replace(chr(10), ' ')[:100]}")


def check_no_staging():
    """第 9 项：库里不该残留 staging / 归档集合（重建中断的痕迹）。

    正式的「中断现场」由 build_index 下次运行时的 cleanup 收拾；这里只负责把它
    报出来 —— 有残留就意味着上次重建没走完。
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CFG.CHROMA_DIR)
        names = []
        for c in client.list_collections():
            names.append(c if isinstance(c, str) else getattr(c, "name", ""))
        stale = [n for n in names
                 if n.startswith(f"staging-{CFG.COLLECTION}-")
                 or n.startswith(f"archive-{CFG.COLLECTION}-")]
    except Exception as e:                       # noqa: BLE001
        record(9, "无残留 staging / 归档集合", "SKIP",
               f"{type(e).__name__}: {str(e)[:60]}")
        return
    if stale:
        record(9, "无残留 staging / 归档集合", "FAIL",
               f"残留 {stale}（上次重建没走完；下次 build_index 会自动收尾）")
    else:
        record(9, "无残留 staging / 归档集合", "PASS")


def _rep_int(v):
    return f"{v:.0f}" if isinstance(v, float) and v.is_integer() else str(v)


class Report(object):
    """报告分三段：head（表头）、text（结论段）、detail（检查明细）。

    · 普通模式：head + text 走 stdout（人看），detail 走 stderr；
    · ``--json``：stdout 只有一段 JSON，其余全走 stderr，方便管道 / CI 解析。
    """

    def __init__(self, as_json):
        self.as_json = as_json
        self.head = []
        self.detail = []            # 检查明细行

    def out(self, line=""):
        self.head.append(line)

    def say(self, line):
        self.detail.append(line)

    def flush(self, summary):
        if self.as_json:
            print(json.dumps(summary, ensure_ascii=False, indent=1))
        else:
            for line in self.head:
                print(line)
        for line in self.detail:
            print(line, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="stdout 只输出 JSON 摘要")
    ap.add_argument("--db-dir", default=None, help="覆盖库目录（默认取配置）")
    ap.add_argument("--collection", default=None, help="覆盖集合名（默认取配置）")
    ap.add_argument("--model", default=None, help="覆盖期望的 embedding 模型")
    args = ap.parse_args()

    global CFG
    CFG = Config(chroma_dir=args.db_dir, collection=args.collection,
                 embed_model=args.model)

    rep = Report(args.json)
    rep.out("=" * 66)
    rep.out("  本地 Chroma 库自检")
    rep.out("=" * 66)
    rep.out(f"  位置：{CFG.CHROMA_DIR}")
    rep.out(f"  集合：{CFG.COLLECTION}    模型：{CFG.EMBED_MODEL}")
    rep.out("  （检查明细走 stderr；FAIL / SKIP 任一项 → 退出码 1）")
    rep.out()

    try:
        _run_checks(rep)
    except Exception as e:                       # noqa: BLE001 —— 自检崩了也算门禁失败
        record(0, "自检本身没有崩溃", "FAIL", f"{type(e).__name__}: {e}")
    return finish(args.json)




def _run_checks(rep):
    check_ascii()
    if not check_dir_and_sqlite():
        for no, name in ((4, "HNSW 索引文件齐全（当前集合）"),
                         (5, "向量条数（sqlite）"), (6, "库能否打开"),
                         (7, "集合向量模型与配置一致"), (8, "检索探测"),
                         (9, "无残留 staging / 归档集合")):
            record(no, name, "SKIP", "前置检查未通过")
        return

    sqlite_path = os.path.join(CFG.CHROMA_DIR, "chroma.sqlite3")
    ids, err = collection_ids(sqlite_path)
    coll_id = ids.get(CFG.COLLECTION)
    if coll_id is None:
        record(4, "HNSW 索引文件齐全（当前集合）", "SKIP",
               f"sqlite 里没有集合 `{CFG.COLLECTION}`（{err or '未建库'}）")
        record(5, "向量条数（sqlite）", "SKIP", "集合不存在")
    else:
        check_hnsw_bins(sqlite_path, coll_id)
        check_count_sqlite(sqlite_path, coll_id)

    col = check_open_and_model()
    check_probe(col)
    check_no_staging()


def finish(as_json):
    code = index_ops.summary_exit_code(_results)
    n_pass = sum(1 for r in _results if r["status"] == "PASS")
    n_fail = sum(1 for r in _results if r["status"] == "FAIL")
    n_skip = sum(1 for r in _results if r["status"] == "SKIP")
    payload = {
        "ok": code == 0,
        "exit_code": code,
        "db_dir": CFG.CHROMA_DIR,
        "collection": CFG.COLLECTION,
        "embed_model": CFG.EMBED_MODEL,
        "embed_dim": CFG.EMBED_DIM,
        "pass": n_pass, "fail": n_fail, "skip": n_skip,
        "checks": _results,
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
    else:
        print()
        print("=" * 66)
        if code == 0:
            print("  结论：✅ 库完好可用")
            print("  下一步：跑检索验证 →")
            here = os.path.dirname(os.path.abspath(__file__))
            print(f'    & $PY "{os.path.join(here, "query_index.py")}"')
        else:
            print(f"  结论：❌ 库不可用（{n_pass}/{N_CHECKS} 项通过，"
                  f"{n_fail} 项失败，{n_skip} 项跳过）")
            print("  按明细里 FAIL 的项逐条处理；重建命令：")
            here = os.path.dirname(os.path.abspath(__file__))
            print(f'    & $PY "{os.path.join(here, "build_index.py")}" --stage db')
            print("  （退出码 = 1，可直接串进脚本 / CI 做门禁）")
        print("=" * 66)
    return code


if __name__ == "__main__":
    sys.exit(main())
