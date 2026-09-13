# -*- coding: utf-8 -*-
"""本地 Chroma 库自检 —— 一条命令给出全部验收判据。

设计要点：**不需要载 embedding 模型**。索引是否完好，用一个随机向量探测
一次 query 就知道（能返回结果 = HNSW 索引可读），几秒出结果。

用法（$PY 的定义见《本地部署操作指引.md》第 0 步）
  & $PY verify_index.py
"""
import glob
import os
import sqlite3
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DB_DIR = r"D:\WorkBuddy\chroma_db"
COLL = "standards"
EXPECT_COUNT = 1083
NEED_BINS = ("header.bin", "data_level0.bin", "length.bin", "link_lists.bin")

_ASCII_BAD = [c for c in DB_DIR if ord(c) > 127]

_rows = []


def _w(s):
    """按终端显示宽度算长度（中文算 2，ASCII 算 1）。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def record(no, total, name, status, detail=""):
    _rows.append(status)
    dot = "." * max(2, 42 - _w(name))
    label = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
    print(f"  [{no}/{total}] {name} {dot} {label}  {detail}")


def main():
    print("=" * 66)
    print("  本地 Chroma 库自检")
    print("=" * 66)
    print(f"  位置：{DB_DIR}")
    print()

    TOTAL = 7
    sqlite_path = os.path.join(DB_DIR, "chroma.sqlite3")

    # --- 1 路径必须纯 ASCII（决定后面 4 个 .bin 能不能写出来）---
    if _ASCII_BAD:
        record(1, TOTAL, "库目录为纯 ASCII", "FAIL",
               f"含 {''.join(sorted(set(_ASCII_BAD)))} —— chromadb 在中文路径下不写 .bin")
    else:
        record(1, TOTAL, "库目录为纯 ASCII", "PASS")

    # --- 2 目录 ---
    if not os.path.isdir(DB_DIR):
        record(2, TOTAL, "数据库目录存在", "FAIL", "目录不存在 —— 还没建过库？")
        summary()
        return
    record(2, TOTAL, "数据库目录存在", "PASS")

    # --- 3 sqlite ---
    if os.path.exists(sqlite_path):
        record(3, TOTAL, "chroma.sqlite3 存在", "PASS",
               f"{os.path.getsize(sqlite_path) / 1024 ** 2:.1f} MB")
    else:
        record(3, TOTAL, "chroma.sqlite3 存在", "FAIL", "主文件缺失")
        summary()
        return

    # --- 4 HNSW 索引文件（关键判据）---
    bins = sorted(glob.glob(os.path.join(DB_DIR, "*", "*.bin")))
    names = [os.path.basename(b) for b in bins]
    missing = [n for n in NEED_BINS if n not in names]
    if missing:
        record(4, TOTAL, "HNSW 索引文件齐全", "FAIL",
               f"期望 4 个，实际 {len(bins)} 个；缺 {', '.join(missing)}"
               + (f"；现有 {names}" if names else ""))
    else:
        record(4, TOTAL, "HNSW 索引文件齐全", "PASS", f"4 个，共 "
               f"{sum(os.path.getsize(b) for b in bins) / 1024 ** 2:.1f} MB")

    # --- 5 向量条数（直接读 sqlite，不经过 chromadb）---
    n_emb = n_queue = None
    try:
        con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        cur = con.cursor()
        cur.execute("select count(*) from embeddings")
        n_emb = cur.fetchone()[0]
        cur.execute("select count(*) from embeddings_queue")
        n_queue = cur.fetchone()[0]
        con.close()
        if n_emb == EXPECT_COUNT:
            record(5, TOTAL, "向量条数", "PASS", f"{n_emb} 条（queue 中待压缩 {n_queue} 行，正常）")
        else:
            record(5, TOTAL, "向量条数", "FAIL", f"{n_emb} 条，期望 {EXPECT_COUNT}")
    except Exception as e:
        record(5, TOTAL, "向量条数", "FAIL", f"读 sqlite 失败：{type(e).__name__}: {e}")

    # --- 6 库能否打开 ---
    col = None
    try:
        import chromadb
        client = chromadb.PersistentClient(path=DB_DIR)
        col = client.get_collection(COLL)
        n = col.count()
        if n == EXPECT_COUNT:
            record(6, TOTAL, "库能否打开", "PASS", f"集合 `{COLL}` count() = {n}")
        else:
            record(6, TOTAL, "库能否打开", "FAIL", f"能打开，但 count() = {n}，期望 {EXPECT_COUNT}")
    except Exception as e:
        col = None
        raw = str(e).replace("\n", " ")[:100]
        msg = f"HNSW 索引加载失败 —— {raw}" if "hnsw" in str(e).lower() else \
              f"{type(e).__name__}: {raw}"
        record(6, TOTAL, "库能否打开", "FAIL", msg)

    # --- 7 检索探测（随机向量，不载模型）---
    if col is None:
        record(7, TOTAL, "检索探测", "SKIP", "库打不开，跳过")
    else:
        try:
            import numpy as np
            q = np.random.rand(1, 1024).astype("float32")
            q /= np.linalg.norm(q)
            res = col.query(query_embeddings=q.tolist(), n_results=3,
                            include=["documents", "metadatas", "distances"])
            ids = res["ids"][0]
            metas = res["metadatas"][0]
            sample = metas[0].get("article_no") if metas else ""
            record(7, TOTAL, "检索探测", "PASS",
                   f"返回 {len(ids)} 条，例如 {ids[0]}（{sample}）")
        except Exception as e:
            record(7, TOTAL, "检索探测", "FAIL",
                   f"{type(e).__name__}: {str(e).replace(chr(10), ' ')[:100]}")

    summary()


def summary():
    n_pass = _rows.count("PASS")
    n_fail = _rows.count("FAIL")
    print()
    print("=" * 66)
    if n_fail == 0 and n_pass == len(_rows):
        print("  结论：✅ 库完好可用")
        print("  下一步：跑检索验证 →")
        print('    & $PY "D:\\WorkBuddy\\项目\\输出文档\\scripts\\query_index.py"')
    elif _ASCII_BAD:
        print(f"  结论：❌ 库不可用（通过 {n_pass} / {len(_rows)} 项）")
        print(f"  原因：库目录含非 ASCII 字符（{''.join(sorted(set(_ASCII_BAD)))}）。")
        print("        chromadb 的 Rust HNSW 在中文路径下**不写 .bin**，")
        print("        必定做出打不开的半成品库 —— 这是本机反复踩的坑。")
        print("  处理：把脚本里的 DB_DIR 改到纯 ASCII 路径（如 D:\\WorkBuddy\\chroma_db），")
        print("        删掉旧库后重建：")
        print('    第 1 步 删旧库：Remove-Item -Recurse -Force "<库目录>"')
        print('    第 2 步 重建  ：& $PY "...\\scripts\\build_index.py" --stage db')
        print("    详见《本地部署操作指引.md》附录 B14。")
    elif n_fail and _rows[:1].count("FAIL") == 0 and _rows[2:4].count("FAIL"):
        print(f"  结论：❌ 库不可用（通过 {n_pass} / {len(_rows)} 项）")
        print("  原因：HNSW 索引文件缺失，但路径是纯 ASCII —— 建库进程没有正常结束。")
        print("  处理：重建（缓存已在的话只要几秒）→")
        print('    第 1 步 删旧库：Remove-Item -Recurse -Force "<库目录>"')
        print('    第 2 步 重建  ：& $PY "...\\scripts\\build_index.py" --stage db')
        print("    详见《本地部署操作指引.md》第 3~4 步。")
    else:
        print(f"  结论：⚠️ 部分通过（{n_pass} / {len(_rows)}）")
        print("  按上面 FAIL 的那几项逐条处理，详见《本地部署操作指引.md》第 6 节。")
    print("=" * 66)


if __name__ == "__main__":
    main()
