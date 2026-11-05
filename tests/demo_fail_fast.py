# -*- coding: utf-8 -*-
"""P0-C 演示脚本：用临时目录里的小样本，把「该拒绝的必须拒绝」真跑一遍。

只跑不依赖重型库的前半段（emb_meta 门禁、配置解析），
真正的 chromadb 原子重建见 tests/e2e_atomicity.py。

    python -m tests.demo_fail_fast
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "输出文档", "scripts", "build_index.py")


def write_chunks(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"id": f"demo-{i}", "text": t, "article_no": str(i),
                                "standard": "GB 50000-2020"}, ensure_ascii=False) + "\n")


def run_build(env, *extra):
    cmd = [sys.executable, BUILD, "--stage", "db"] + list(extra)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    print(f"\n$ build_index.py --stage db  →  exit {p.returncode}")
    for line in (p.stderr or "").rstrip().splitlines():
        print("  ! " + line)
    return p


def main():
    tmp = tempfile.mkdtemp(prefix="p0c-demo-")
    cache = os.path.join(tmp, "cache")
    os.makedirs(cache)
    chunks = os.path.join(tmp, "chunks.jsonl")
    write_chunks(chunks, ["第 1 条正文", "第 2 条正文", "第 3 条正文"])
    sha1 = hashlib.sha1(open(chunks, "rb").read()).hexdigest()
    meta_path = os.path.join(cache, "emb_meta.json")
    base_meta = {"sha1": sha1, "count": 3, "model": "Qwen/Qwen3-Embedding-0.6B",
                 "dim": 1024, "shape": [3, 1024]}

    env = dict(os.environ, PYTHONPATH=ROOT)
    env.update({"CHUNKS_FILE": chunks, "EMB_CACHE_DIR": cache,
                "CHROMA_DIR": os.path.join(tmp, "db"),
                "EMBED_MODEL": "Qwen/Qwen3-Embedding-0.6B", "EMBED_DIM": "1024",
                "PYTHONIOENCODING": "utf-8"})
    for k in ("PROJECT_ROOT", "BCR_FAKE_FAIL_ADD_AT"):
        env.pop(k, None)

    results = []

    print("=" * 70)
    print("A) 缓存不存在（emb_meta.json 缺失）→ 必须拒绝")
    p = run_build(env)
    results.append(("meta 缺失 exit 3", p.returncode == 3))
    results.append(("提示 --stage vec --force", "--stage vec --force" in (p.stderr or "")))

    print("=" * 70)
    print("B) 条数不变、正文改一个字（sha1 变）→ 必须拒绝")
    write_chunks(chunks, ["第 1 条正文", "第 2 条正文（改）", "第 3 条正文"])
    json.dump(base_meta, open(meta_path, "w", encoding="utf-8"))
    p = run_build(env)
    results.append(("sha1 变了 exit 3", p.returncode == 3))
    results.append(("错误里点到 sha1", "sha1" in (p.stderr or "")))

    print("=" * 70)
    print("C) 换模型（EMBED_MODEL 与缓存不一致）→ 必须拒绝")
    write_chunks(chunks, ["第 1 条正文", "第 2 条正文", "第 3 条正文"])
    sha1 = hashlib.sha1(open(chunks, "rb").read()).hexdigest()
    json.dump(dict(base_meta, sha1=sha1), open(meta_path, "w", encoding="utf-8"))
    p = run_build(dict(env, EMBED_MODEL="BAAI/bge-m3"))
    results.append(("换模型 exit 3", p.returncode == 3))
    results.append(("错误里点到模型", "向量模型不一致" in (p.stderr or "")))

    print("=" * 70)
    print("D) 配置全部对上（只差 .npy）→ 走到载向量这步才失败，且不是 meta 门禁")
    p = run_build(env)
    # 这一档 meta 五项全对，只剩 .npy 缺失 —— 走「缓存不完整」分支，退出码同样是 3
    results.append(("meta 全对上时失败点落在 .npy（exit 3）",
                    p.returncode == 3 and "缓存不完整" in (p.stderr or "")))
    results.append(("坏缓存提示 .npy 不存在", ".npy" in (p.stderr or "")))

    print("\n" + "=" * 70)
    ok = all(v for _, v in results)
    for name, v in results:
        print(f"  {'PASS' if v else 'FAIL'}  {name}")
    print("=" * 70)
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
