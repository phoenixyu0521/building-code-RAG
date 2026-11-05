# -*- coding: utf-8 -*-
"""真实 Chroma 上的端到端测试（P0-C v2）。

不是假客户端，跑真的 chromadb 1.5.9：
  1. 150 条假数据（**低于 sync_threshold=1000**，专门覆盖「小集合」场景）
  2. build_index.py --stage db 建库          → exit 0，verify 0
  3. .bin 通过 **segment 映射** 找到（断言目录名 = 向量段 uuid+1，且与集合 id 不同）
  4. 缩容重建 150 → 120                      → exit 0，条数 120、可查、无残留
  5. 故障注入（后台第 2 批 add 前抛异常）    → exit 4，**旧集合（120 条）仍可查**
  6. 故障注入（旧集合已改归档名之后抛异常）  → exit 4，旧集合被改回正式名，仍可查
  7. 硬杀在「旧名→归档名」之后                → 下次运行自动改回，旧集合可查
  8. 把 .bin 挪到别的目录                     → verify exit 1（别的目录不算数）

跑法（用项目自带 Python）：
    $PY -m tests.e2e_atomicity
"""
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "输出文档", "scripts")
BUILD = os.path.join(SCRIPTS, "build_index.py")
VERIFY = os.path.join(SCRIPTS, "verify_index.py")
MODEL = "Qwen/Qwen3-Embedding-0.6B"
DIM = 8
PY = sys.executable


def _run(cmd, env, tag, timeout=300):
    p = subprocess.run([PY] + cmd, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    print("\n$ python " + " ".join(os.path.basename(c) for c in cmd)
          + f"    [{tag}]  → exit {p.returncode}")
    for line in (p.stdout or "").rstrip().splitlines()[-30:]:
        print("  " + line)
    err = (p.stderr or "").rstrip()
    if err:
        for line in err.splitlines()[-18:]:
            print("  ! " + line)
    return p


def _write_chunks(path, n, salt=""):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "id": f"fake-{i}", "text": f"第 {i} 条测试正文{salt}",
                "article_no": f"3.0.{i}", "standard": "GB 50000-2020",
                "standard_name": "测试用标准", "version": "", "force_status": "一般",
                "status": "现行", "chapter": "1", "page_start": 1, "page_end": 1,
                "kind": "条",
            }, ensure_ascii=False) + "\n")


def _write_cache(cache, chunks, n):
    import numpy as np
    raw = open(chunks, "rb").read()
    sha1 = hashlib.sha1(raw).hexdigest()
    np.save(os.path.join(cache, "emb_qwen3-embedding-0.6b.npy"),
            np.eye(n, DIM, dtype="float32"))
    meta = {"sha1": sha1, "count": n, "model": MODEL, "dim": DIM,
            "shape": [n, DIM], "seconds": 0, "created": "e2e"}
    with open(os.path.join(cache, "emb_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return sha1


def _sqlite(db):
    return sqlite3.connect(f"file:{os.path.join(db, 'chroma.sqlite3')}?mode=ro", uri=True)


def _coll_info(db, name):
    con = _sqlite(db)
    try:
        row = con.execute("select id from collections where name=?", (name,)).fetchone()
        if not row:
            return None, []
        cid = row[0]
        segs = con.execute("select id, scope from segments where collection=?",
                           (cid,)).fetchall()
        return cid, segs
    finally:
        con.close()


def _vec_segment_dir(segs):
    """索引目录名：实测 Chroma 1.5.9 直接用向量段的 uuid（旧版算 uuid+1 对不上）。"""
    for sid, scope in segs:
        if str(scope).upper() == "VECTOR":
            return str(sid)
    return None


def _bins(db, d):
    p = os.path.join(db, d)
    return sorted(f for f in os.listdir(p)) if os.path.isdir(p) else []


def _names(db):
    con = _sqlite(db)
    try:
        return sorted(r[0] for r in con.execute("select name from collections"))
    finally:
        con.close()


def _count(db, name):
    """新开一个 chromadb 客户端，确认这个集合真的可查（返回 None = 打不开）。"""
    code = ("import chromadb,sys;"
            "c=chromadb.PersistentClient(path=sys.argv[1]);"
            "print(c.get_collection(sys.argv[2]).count())")
    p = subprocess.run([PY, "-c", code, db, name], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=180)
    if p.returncode != 0:
        return None
    try:
        return int(p.stdout.strip().splitlines()[-1])
    except Exception:                                     # noqa: BLE001
        return None


def main():
    # 临时库落在系统 temp（BCR_E2E_DIR 可改）。别放工作副本里 ——
    # 本会话对含 Everyone:(DC) 标记的目录（工作副本及其子目录）有写入限制。
    base_tmp = os.environ.get("BCR_E2E_DIR") or None
    if base_tmp:
        os.makedirs(base_tmp, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="p0c-e2e-", dir=base_tmp)
    cache = os.path.join(tmp, "cache")
    db = os.path.join(tmp, "db")
    os.makedirs(cache)
    chunks = os.path.join(tmp, "chunks.jsonl")

    env = dict(os.environ)
    env.update({"PYTHONPATH": ROOT, "CHROMA_DIR": db, "CHROMA_COLLECTION": "standards",
                "EMBED_MODEL": MODEL, "EMBED_DIM": str(DIM),
                "CHUNKS_FILE": chunks, "EMB_CACHE_DIR": cache,
                "PYTHONIOENCODING": "utf-8", "BCR_BUILD_WAIT_S": "60"})
    for k in ("PROJECT_ROOT", "BCR_FAKE_FAIL_ADD_AT", "BCR_FAKE_FAIL_RENAMED"):
        env.pop(k, None)

    results = []

    def ok(name, cond, extra=""):
        results.append((name, bool(cond), extra))

    print("=" * 78)
    print("1) 首次建库：150 条（< sync_threshold 1000）")
    _write_chunks(chunks, 150)
    _write_cache(cache, chunks, 150)
    p = _run([BUILD, "--stage", "db"], env, "expect 0")
    ok("首次建库 exit 0", p.returncode == 0)
    p = _run([VERIFY, "--json"], env, "expect 0 + json")
    try:
        data = json.loads(p.stdout)
    except Exception:                                     # noqa: BLE001
        data = {}
    ok("首次自检 ok", data.get("ok") is True,
       ",".join(f"{c['no']}:{c['status']}" for c in data.get("checks", [])))

    print("=" * 78)
    print("2) .bin 归属：目录名 = 向量段 uuid，与集合 id 不同")
    cid, segs = _coll_info(db, "standards")
    seg_dir = _vec_segment_dir(segs)
    if not cid or not seg_dir:
        print(f"❌ 上一步没建出可用的集合：cid={cid!r} seg_dir={seg_dir!r} "
              f"names={_names(db)}")
        print("   建库日志见上面「expect 0」那一段。")
        return 1
    files = _bins(db, seg_dir)
    print(f"   集合 id   = {cid}")
    print(f"   segments  = {segs}")
    print(f"   索引目录   = {seg_dir}  →  {files}")
    print(f"   按集合 id 找目录 = {cid} → {_bins(db, cid) or '（空，这正是 v1 的错误）'}")
    ok("目录名 ≠ 集合 id", seg_dir != cid)
    ok("4 个 .bin 在 segment 目录里",
       {"header.bin", "data_level0.bin", "length.bin", "link_lists.bin"} <= set(files))
    ok("按集合 id 找不到 .bin", _bins(db, cid) == [])

    print("=" * 78)
    print("3) 合法缩容 150 → 120（chunks 与缓存一起变）")
    _write_chunks(chunks, 120, salt="（v2）")
    _write_cache(cache, chunks, 120)
    p = _run([BUILD, "--stage", "db"], env, "expect 0")
    ok("缩容重建 exit 0", p.returncode == 0)
    ok("新集合 120 条可查", _count(db, "standards") == 120, str(_count(db, "standards")))
    ok("只留正式名（无归档 / staging 残留）", _names(db) == ["standards"], str(_names(db)))
    p = _run([VERIFY, "--json"], env, "expect 0 + json")
    ok("缩容后自检 ok", json.loads(p.stdout or "{}").get("ok") is True)

    print("=" * 78)
    print("4) 故障注入：后台第 2 批 add 前抛异常（新集合灌到一半）")
    _write_chunks(chunks, 140, salt="（v3）")
    _write_cache(cache, chunks, 140)
    # 每批 50 条 → 140 条分 3 批，第 2 批前中断，staging 里只有 50 条。
    p = _run([BUILD, "--stage", "db"],
             dict(env, BCR_FAKE_FAIL_ADD_AT="2", BCR_DB_BATCH="50"), "expect 4")
    ok("写入中断 exit 4", p.returncode == 4)
    ok("旧集合仍是 120 条、可查", _count(db, "standards") == 120)
    ok("没有残留 staging", [n for n in _names(db) if n.startswith("staging-")] == [],
       str(_names(db)))

    print("=" * 78)
    print("5) 故障注入：旧集合已改名成 archive-*（S1 之后）时中断")
    p = _run([BUILD, "--stage", "db"], dict(env, BCR_FAKE_FAIL_RENAMED="1"), "expect 4")
    ok("S1 之后中断 exit 4", p.returncode == 4)
    ok("旧集合被改回正式名且可查", _count(db, "standards") == 120, str(_names(db)))
    ok("无归档 / staging 残留",
       [n for n in _names(db) if n.startswith(("archive-", "staging-"))] == [],
       str(_names(db)))

    print("=" * 78)
    print("6) 硬杀：把 standards 改名成 archive-standards-*（模拟 S1 之后进程被杀）")
    code = ("import chromadb,sys;c=chromadb.PersistentClient(path=sys.argv[1]);"
            "c.get_collection('standards').modify(name='archive-standards-20260101')")
    subprocess.run([PY, "-c", code, db], check=True, timeout=180)
    print("   现在库里的集合:", _names(db))
    ok("硬杀现场：只有归档，正式名不可查", _count(db, "standards") is None)
    p = _run([BUILD, "--stage", "db"], env, "expect 0（启动时自动改回）")
    ok("下次运行自动还原并重建 exit 0", p.returncode == 0)
    ok("还原后正式名可用、140 条", _count(db, "standards") == 140, str(_names(db)))
    ok("归档已清理", [n for n in _names(db) if n.startswith("archive-")] == [],
       str(_names(db)))

    print("=" * 78)
    print("7) 把当前集合的 .bin 挪到别的目录 → 自检 exit 1")
    _cid, segs = _coll_info(db, "standards")
    seg_dir = _vec_segment_dir(segs)
    other = os.path.join(db, "00000000-0000-0000-0000-000000000000")
    os.makedirs(other, exist_ok=True)
    for f in os.listdir(os.path.join(db, seg_dir)):
        if f.endswith(".bin"):
            shutil.move(os.path.join(db, seg_dir, f), os.path.join(other, f))
    p = _run([VERIFY], env, "expect 1")
    ok("别的目录有 .bin 也不算数（exit 1）",
       p.returncode == 1 and "不算数" in (p.stderr or ""))

    print("\n" + "=" * 78)
    allok = all(v for _, v, _ in results)
    for name, v, extra in results:
        print(f"  {'PASS' if v else 'FAIL'}  {name}" + (f"   [{extra}]" if extra and not v else ""))
    print("=" * 78)
    print(f"临时库：{tmp}")
    if allok:
        shutil.rmtree(tmp, ignore_errors=True)
        print("（全部通过，临时库已删除）")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
