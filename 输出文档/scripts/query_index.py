# -*- coding: utf-8 -*-
"""本地规范库检索验证 —— 输入问题，打印命中的条文与相似度。

用法（$PY 的定义见《本地部署操作指引.md》第 0 步）
  & $PY query_index.py
"""
import os
import sys

# 模型缓存位置必须在 import sentence_transformers 之前定下来。
# setdefault：你若自己设过 HF_HOME，以你的为准。
os.environ.setdefault("HF_HOME", r"D:\WorkBuddy\models\hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import chromadb                                       # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")           # Windows 控制台默认 GBK
except Exception:
    pass

DB_DIR = r"D:\WorkBuddy\chroma_db"   # 必须纯 ASCII！见 build_index.py 文件头「中文路径陷阱」
MODEL  = "Qwen/Qwen3-Embedding-0.6B"   # 必须和 build_index.py 一致！
COLL   = "standards"
TOP_K  = 5


def fmt(meta, doc, dist, rank):
    # cosine 距离 → 相似度
    sim = 1.0 - float(dist)
    print(f"\n[{rank}] 相似度 {sim:.4f}（距离 {dist:.4f}）")
    print(f"     《{meta.get('standard_name', '')}》{meta.get('standard', '')}"
          f"  第 {meta.get('article_no', '')} 条  ［{meta.get('force_status', '')}］")
    print(f"     {doc[:220]}")


def main():
    print("载入模型（首次约 1~7 分钟）…")
    model = SentenceTransformer(MODEL)
    client = chromadb.PersistentClient(path=DB_DIR)
    col = client.get_collection(COLL)
    print(f"库内 {col.count()} 条，模型 {MODEL}")
    print("提示：查询侧必须带 prompt_name='query'（模型内置检索指令），漏了不会报错，"
          "只会悄悄变差。")

    while True:
        q = input("\n提问（直接回车退出）> ").strip()
        if not q:
            break
        qv = model.encode([q], prompt_name="query",
                          normalize_embeddings=True).tolist()
        res = col.query(query_embeddings=qv, n_results=TOP_K,
                        include=["documents", "metadatas", "distances"])
        for rank, (doc, meta, dist) in enumerate(
                zip(res["documents"][0], res["metadatas"][0],
                    res["distances"][0]), 1):
            fmt(meta, doc, dist, rank)


if __name__ == "__main__":
    main()
