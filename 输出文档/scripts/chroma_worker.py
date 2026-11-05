# -*- coding: utf-8 -*-
"""后台库持有者：一个长期存活的子进程，独占 D:\\...\\chroma_db 的 Chroma 客户端。

为什么必须单独一个进程
--------------------
Chroma 1.5.9 的客户端会**缓存「某某名字不存在」**，而建集合前必须先确认名字不存在；
同一个客户端既查又建会把这个缓存写脏，之后按名字对 staging 改名就会 NotFoundError。
（实测：同一 client 上 ``get_collection("probe_b")`` 抛 NotFoundError 后
``get_or_create_collection("probe_b")`` 建出来的集合，按名改名的路走不通。）

所以约定：
  · **后台进程（main.py--serve）** 是唯一的库持有者，负责所有写操作与改名；
  · **主进程** 每次执行动作都开一个新的客户端，只做只读 + 改名，并且从不建集合；
  · 主进程用同一个客户端时，``list_collections`` 能立刻看到后台刚做的改名（实测）；
  · 后台进程启动时还会顺带把上次被硬杀留下的 ``archive-*`` 改回正式名。

协议：stdin 一行一个 JSON 动作，stdout 一行一个 JSON 回执。
  {"op":"ping"}
  {"op":"resize","name":..,"n":..,"cnt":..,"freq":[..]}
  {"op":"write_collection","name":..,"rows":..,"embs":..,"meta":..,"batch":..,
   "fail_at_batch":..,"delay_ms":..}
  {"op":"verify","name":..,"rows":..,"embs":..,"expect_dim":..,"sample":..}
  {"op":"count","name":..}
  {"op":"delete","name":..}
  {"op":"rename","old":..,"new":..}
  {"op":"names"}
  {"op":"cleanup","collection":..}
  {"op":"quit"}
"""
import json
import os
import sys
import time
import traceback

import numpy as np


def _emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    from app import index_ops as ops
    import chromadb

    db = sys.argv[1]
    client = chromadb.PersistentClient(path=db)
    _emit({"ok": True, "op": "up", "pid": os.getpid(), "db": db})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:                       # noqa: BLE001
            _emit({"ok": False, "error": f"bad json: {e}"})
            continue
        op = req.get("op")
        try:
            if op == "ping":
                _emit({"ok": True, "pid": os.getpid()})

            elif op == "resize":
                name, n = req["name"], int(req["n"])
                cnt = int(req.get("cnt") or n)
                freq = np.asarray(req.get("freq") or [], dtype=float)
                col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
                have = col.count()
                if n > have:
                    if freq.size:
                        new = np.random.choice(len(freq), size=n - have, p=freq / freq.sum())
                    else:
                        new = np.arange(have, n)
                    ids = [f"{name}-r{i}-{int(v)}" for i, v in enumerate(new)]
                    col.add(ids=ids, documents=[f"正文 {int(v)}" for v in new],
                            embeddings=[[float((int(v) * 7 + k) % 13 + 1) for k in range(1024)]
                                        for v in new],
                            metadatas=[{"article_no": str(int(v))} for v in new])
                _emit({"ok": True, "op": "resize", "name": name, "count": col.count()})

            elif op == "write_collection":
                name = req["name"]
                rows = req["rows"]
                embs = np.asarray(req["embs"], dtype="float32")
                meta = req.get("meta") or {}
                batch = int(req.get("batch") or 256)
                fail_at = req.get("fail_at_batch")
                delay = int(req.get("delay_ms") or 0)
                n = len(rows)
                n_batch = max(1, (n + batch - 1) // batch)
                col = client.create_collection(name, metadata=meta)
                for b, i in enumerate(range(0, n, batch), 1):
                    if fail_at is not None and b == int(fail_at):
                        raise RuntimeError(f"[故障注入] 第 {b} 批 add 前中断（{name}）")
                    j = min(i + batch, n)
                    col.add(ids=[r["id"] for r in rows[i:j]],
                            documents=[r["text"] for r in rows[i:j]],
                            embeddings=embs[i:j].tolist(),
                            metadatas=[{k: ("" if v is None else v)
                                        for k, v in r["meta"].items()} for r in rows[i:j]])
                    _emit({"ok": True, "op": "progress", "name": name,
                           "batch": b, "batches": n_batch, "count": col.count()})
                    if delay:
                        time.sleep(delay / 1000.0)
                _emit({"ok": True, "op": "write_collection", "name": name,
                       "count": col.count()})

            elif op == "verify":
                name = req["name"]
                rows = req["rows"]
                embs = np.asarray(req["embs"], dtype="float32")
                col = client.get_collection(name)
                ok, detail = ops.verify_collection(
                    col, rows, embs, int(req["expect_dim"]),
                    verify_sample=int(req.get("sample") or 1))
                _emit({"ok": True, "op": "verify", "passed": ok, "detail": detail})

            elif op == "count":
                col = client.get_collection(req["name"])
                _emit({"ok": True, "op": "count", "name": req["name"],
                       "count": col.count()})

            elif op == "delete":
                client.delete_collection(req["name"])
                _emit({"ok": True, "op": "delete", "name": req["name"]})

            elif op == "rename":
                old, new = req["old"], req["new"]
                client.get_collection(old).modify(name=new)
                _emit({"ok": True, "op": "rename", "old": old, "new": new})

            elif op == "names":
                out = []
                for c in client.list_collections():
                    out.append(c if isinstance(c, str) else getattr(c, "name", c))
                _emit({"ok": True, "op": "names", "names": sorted(out)})

            elif op == "cleanup":
                acted, removed = ops.cleanup_stale(client, req["collection"],
                                                   log=lambda *a: None)
                out = []
                for c in client.list_collections():
                    out.append(c if isinstance(c, str) else getattr(c, "name", c))
                _emit({"ok": True, "op": "cleanup", "acted": acted,
                       "removed": removed, "names": sorted(out)})

            elif op == "quit":
                _emit({"ok": True, "op": "quit"})
                return

            else:
                _emit({"ok": False, "error": f"unknown op {op!r}"})
        except Exception as e:                       # noqa: BLE001
            _emit({"ok": False, "op": op, "error": f"{type(e).__name__}: {e}",
                   "trace": traceback.format_exc()[-800:]})


if __name__ == "__main__":
    main()
