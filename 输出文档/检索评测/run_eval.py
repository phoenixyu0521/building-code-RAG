# -*- coding: utf-8 -*-
"""检索评测跑分器 —— 同一份评测集，两种后端，多组检索参数。

它只评「检索」这一层：该召回的条文，有没有被召回、排在第几。
不评「答案写得对不对」——那是生成阶段的评测（见上级目录 知识库测试题.md）。

后端
  dify    Dify Cloud / 自托管 的「召回测试」API：POST /datasets/{id}/hit-testing
  chroma  本机 Chroma 库（输出文档\\chroma_db），用于做本地 vs 云端的 A/B

参数组（--profile）已经把要对比的检索配置写好了，直接跑即可：
  base      混合检索 + rerank(qwen3-rerank) + TopK5 + 阈值0.5   ← 线上默认
  sem       纯向量检索 + rerank
  pure      纯向量检索，无 rerank、无阈值（A/B 用，与 chroma 后端可比）
  kw        全文/倒排检索，无 rerank
  w73/w55/w91  混合检索 + 加权打分（关键词权重 0.3 / 0.5 / 0.1），不调 rerank 模型
  no_rk     混合检索，rerank 关闭（用来验证「Score 阈值必须配 rerank」）
  rk_gte    混合检索 + rerank(gte-rerank-v2)
  top10     混合检索 + rerank + TopK10
  thr03/thr07  混合检索 + rerank，阈值 0.3 / 0.7

用法示例
  set DIFY_API_KEY=dataset-xxxx          (PowerShell: $env:DIFY_API_KEY="dataset-xxxx")
  set DIFY_DATASET_ID=xxxxxxxx
  python run_eval.py --backend dify --profile base
  python run_eval.py --backend dify --all
  python run_eval.py --backend chroma --profile local
  python run_eval.py --backend dify --profile base --only-type table --limit 6

注意
  · Dify Sandbox 免费版知识库请求 10 次/分钟 → 默认 --delay 6.5s，86 条约 9 分钟。
  · Dify hit-testing 单次最多返回 10 条，top_k > 10 无意义。
  · query 长度上限 250 字符（评测集已校验）。
"""
import argparse
import collections
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 本地 chroma 后端要载 embedding 模型：缓存位置必须在 import 之前定下来。
# setdefault：你若自己设过 HF_HOME，以你的为准。
os.environ.setdefault("HF_HOME", r"D:\WorkBuddy\models\hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(BASE_DIR))          # D:\WorkBuddy\项目
EVAL_PATH = os.path.join(BASE_DIR, "评测集.jsonl")
CHROMA_DIR = r"D:\WorkBuddy\chroma_db"   # 必须纯 ASCII！见 build_index.py「中文路径陷阱」
COLL = "standards"
MODEL = "Qwen/Qwen3-Embedding-0.6B"

HEADER_RE = re.compile(r"】(.+?)（")            # 从检索结果正文里抠出「条文号」

RERANKERS = {                                   # 通义 / 硅基流动 的两支可用 rerank
    "qwen3": ("tongyi", "qwen3-rerank"),
    "gte": ("tongyi", "gte-rerank-v2"),
}


def hybrid(rerank="qwen3", top_k=5, thr=0.5, thr_on=True,
           weights=None, search="hybrid_search", rerank_on=True):
    rm = {
        "search_method": search,
        "reranking_enable": rerank_on,
        "reranking_mode": "reranking_model" if rerank_on else None,
        "reranking_model": {
            "reranking_provider_name": RERANKERS[rerank][0] if rerank_on else "",
            "reranking_model_name": RERANKERS[rerank][1] if rerank_on else "",
        },
        "weights": weights,
        "top_k": top_k,
        "score_threshold_enabled": thr_on,
        "score_threshold": thr if thr_on else None,
    }
    return rm


def W(kw, vec):
    return {
        "weight_type": "customized",
        "keyword_setting": {"keyword_weight": kw},
        "vector_setting": {"vector_weight": vec,
                           "embedding_provider_name": "tongyi",
                           "embedding_model_name": "text-embedding-v4"},
    }


PROFILES = {
    "base":   hybrid(rerank="qwen3", top_k=5,  thr=0.5),
    "sem":    hybrid(rerank="qwen3", top_k=5,  thr=0.5, search="semantic_search"),
    "pure":   hybrid(top_k=10, thr_on=False, search="semantic_search", rerank_on=False),
    "kw":     hybrid(top_k=10, thr_on=False, search="full_text_search", rerank_on=False),
    "w73":    hybrid(top_k=10, thr_on=False, rerank_on=False, weights=W(0.3, 0.7)),
    "w55":    hybrid(top_k=10, thr_on=False, rerank_on=False, weights=W(0.5, 0.5)),
    "w91":    hybrid(top_k=10, thr_on=False, rerank_on=False, weights=W(0.1, 0.9)),
    "no_rk":  hybrid(top_k=5,  thr_on=True, thr=0.5, rerank_on=False),
    "rk_gte": hybrid(rerank="gte", top_k=5, thr=0.5),
    "top10":  hybrid(rerank="qwen3", top_k=10, thr=0.5),
    "thr03":  hybrid(rerank="qwen3", top_k=5, thr=0.3),
    "thr07":  hybrid(rerank="qwen3", top_k=5, thr=0.7),
    # 本地 Chroma 对照组：纯向量、无 rerank、无阈值、top10
    "local":  {"_local": True, "top_k": 10},
}


# --------------------------------------------------------------------------
# 后端
# --------------------------------------------------------------------------
class DifyBackend:
    name = "dify"

    def __init__(self, cfg, profile, top_k_override=None, timeout=60):
        self.base = cfg["base"].rstrip("/")
        self.key = cfg["api_key"]
        self.ds = cfg["dataset_id"]
        self.rm = dict(PROFILES[profile])
        if top_k_override:
            self.rm["top_k"] = top_k_override
        self.timeout = timeout

    @property
    def top_k(self):
        return int(self.rm.get("top_k") or 5)

    def search(self, query):
        body = {"query": query, "retrieval_model": self.rm}
        req = urllib.request.Request(
            f"{self.base}/datasets/{self.ds}/hit-testing",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        out = []
        for rec in data.get("records") or []:
            seg = rec.get("segment") or {}
            content = seg.get("content") or ""
            m = HEADER_RE.search(content)
            out.append({
                "score": rec.get("score"),
                "gold_key": (m.group(1) if m else ""),
                "doc": (seg.get("document") or {}).get("name", ""),
                "head": content.split("\n", 1)[0][:60],
            })
        return out


class ChromaBackend:
    name = "chroma"

    def __init__(self, cfg, profile, top_k_override=None, timeout=None):
        import chromadb
        from sentence_transformers import SentenceTransformer

        print("  载入本地模型（首次约 1~7 分钟）…")
        t = time.time()
        self.model = SentenceTransformer(MODEL)
        print(f"  模型就绪 {time.time() - t:.1f}s")
        try:
            self.col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLL)
        except Exception as e:
            raise SystemExit(
                f"\n打不开本地库：{CHROMA_DIR}\n"
                f"  {type(e).__name__}: {str(e)[:150]}\n"
                f"  → 先自检并重建：& $PY "
                f"\"D:\\WorkBuddy\\项目\\输出文档\\scripts\\verify_index.py\"\n"
                f"     详见《本地部署操作指引.md》第 3~4 步。")
        self._tk = int(top_k_override or PROFILES[profile].get("top_k") or 10)

    @property
    def top_k(self):
        return self._tk

    def search(self, query):
        try:                        # Qwen3-Embedding 支持 query 指令前缀
            emb = self.model.encode([query], prompt_name="query",
                                    normalize_embeddings=True)[0]
        except Exception:
            emb = self.model.encode([query], normalize_embeddings=True)[0]
        res = self.col.query(query_embeddings=[emb.tolist()], n_results=self._tk,
                             include=["documents", "metadatas", "distances"])
        out = []
        for doc, meta, dist in zip(res["documents"][0],
                                   res["metadatas"][0],
                                   res["distances"][0]):
            out.append({
                "score": max(0.0, min(1.0, 1.0 - float(dist))),   # cosine 距离 → 相似度
                "gold_key": str(meta.get("article_no") or ""),
                "doc": str(meta.get("standard") or ""),
                "head": (doc or "").split("\n", 1)[0][:60],
            })
        return out


# --------------------------------------------------------------------------
# 跑分
# --------------------------------------------------------------------------
def load_eval(only_type=None, only_standard=None, limit=None):
    items = [json.loads(l) for l in open(EVAL_PATH, encoding="utf-8") if l.strip()]
    if only_type:
        items = [i for i in items if i["type"] == only_type]
    if only_standard:
        items = [i for i in items if i["standard"] == only_standard]
    if limit:
        items = items[:limit]
    return items


def rank_of_gold(item, hits):
    """返回金标首次出现的名次（1 起）；没命中返回 0。"""
    gold = set(item["gold"])
    if not gold:
        return 0
    for i, h in enumerate(hits, 1):
        if h.get("gold_key") in gold:
            return i
    return 0


def run(backend, items, delay):
    pos, neg, details = [], [], []
    for n, it in enumerate(items, 1):
        t = time.time()
        try:
            hits = backend.search(it["query"])
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:200]
            raise SystemExit(f"[{it['qid']}] HTTP {e.code}：{body}")
        except Exception as e:
            raise SystemExit(f"[{it['qid']}] {type(e).__name__}: {e}")
        cost = time.time() - t

        if it["type"] == "unanswerable":
            mx = max([h["score"] for h in hits if h["score"] is not None] or [0.0])
            neg.append({"qid": it["qid"], "max_score": mx, "n": len(hits)})
        else:
            r = rank_of_gold(it, hits)
            rec = {"qid": it["qid"], "rank": r, "n": len(hits),
                   "type": it["type"], "standard": it["standard"],
                   "difficulty": it["difficulty"], "cost": round(cost, 2),
                   "top1_score": (hits[0]["score"] if hits else None),
                   "gold_score": (hits[r - 1]["score"] if r else None)}
            if r == 0:
                rec["top"] = [h["head"] for h in hits[:3]]
            pos.append(rec)
        mark = "—" if it["type"] == "unanswerable" else (
            f"@{pos[-1]['rank']}" if pos and pos[-1]["rank"] else "miss")
        sys.stdout.write(f"\r  {n}/{len(items)}  {it['qid']:9s} {mark:5s}")
        sys.stdout.flush()
        if delay and n < len(items):
            time.sleep(delay)
    print()
    details = {"pos": pos, "neg": neg}
    return details


def summarize(details, top_k, thr):
    pos, neg = details["pos"], details["neg"]
    n = len(pos)
    ks = [k for k in (1, 3, 5, 10) if k <= top_k]
    m = {"n_pos": n, "top_k": top_k, "threshold": thr}
    for k in ks:
        m[f"recall@{k}"] = sum(1 for p in pos if 0 < p["rank"] <= k) / n if n else 0
    m["mrr"] = sum((1.0 / p["rank"]) for p in pos if p["rank"]) / n if n else 0
    m["miss"] = sum(1 for p in pos if not p["rank"])
    m["neg_over_thr"] = sum(1 for v in neg if v["max_score"] >= thr)
    m["neg_max"] = max([v["max_score"] for v in neg] or [0.0])

    # 阈值窗口：负例最高分 < 阈值 ≤ 正例金标分数（取 5% 分位，避开个别离群低分）
    gs = sorted(p["gold_score"] for p in pos if p["gold_score"] is not None)
    if gs:
        p05 = gs[max(0, int(len(gs) * 0.05) - 1)]
        m["gold_score_p05"] = p05
        m["gold_score_min"] = gs[0]
        m["gold_score_med"] = gs[len(gs) // 2]
        lo, hi = m["neg_max"], p05
        m["thr_window"] = [round(lo, 3), round(hi, 3)]
        m["thr_suggest"] = round((lo + hi) / 2, 3) if hi > lo else round(lo, 3)
    return m


def md_report(profile, backend, m, details, cfg_note=""):
    pos, neg = details["pos"], details["neg"]
    L = []
    L.append(f"# 检索评测结果 · {backend} · {profile}")
    L.append("")
    L.append(f"> {cfg_note}  ·  TopK={m['top_k']}  阈值={m['threshold']}  ·  "
             f"正例 {m['n_pos']} 条 / 负例 {len(neg)} 条")
    L.append("")
    L.append("## 总体")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    for k in (1, 3, 5, 10):
        key = f"recall@{k}"
        if key in m:
            L.append(f"| {key.upper()} | **{m[key]:.1%}** |")
    L.append(f"| MRR | **{m['mrr']:.3f}** |")
    L.append(f"| 未命中条数 | {m['miss']} / {m['n_pos']} |")
    L.append(f"| 负例最高分 | {m['neg_max']:.3f} |")
    L.append(f"| 负例超过阈值的条数 | **{m['neg_over_thr']} / {len(neg)}** |")
    if m.get("thr_window"):
        lo, hi = m["thr_window"]
        L.append(f"| 正例金标分数（5% 分位 / 中位 / 最低） | "
                 f"{hi:.3f} / {m['gold_score_med']:.3f} / {m['gold_score_min']:.3f} |")
    L.append("")

    if m.get("thr_window"):
        lo, hi = m["thr_window"]
        L.append("## 阈值定在哪")
        L.append("")
        L.append(f"负例最高分 **{lo:.3f}**，正例金标分数 5% 分位 **{hi:.3f}**。")
        if hi > lo:
            L.append(f"→ 窗口 **({lo:.3f}, {hi:.3f}]**，建议取值 **{m['thr_suggest']:.2f}**"
                     f"（窗口中点）。比它低则负例会漏进来，比它高则正例开始被误伤。")
        else:
            L.append("→ **窗口不存在**（正例分数没有明显高于负例）。这说明当前配置下"
                     "召回与拒答无法靠阈值同时满足，需要先换 rerank 模型或调整权重，"
                     "不能只靠调阈值。")
        L.append("")

    def brk(key, title):
        g = collections.defaultdict(lambda: [0, 0])
        for p in pos:
            g[p[key]][1] += 1
            if 0 < p["rank"] <= min(5, m["top_k"]):
                g[p[key]][0] += 1
        kk = min(5, m["top_k"])
        L.append(f"## {title}（召回@{kk}）")
        L.append("")
        L.append("| 分组 | 命中 / 总数 | 召回率 |")
        L.append("|---|---|---|")
        for kk2 in sorted(g, key=lambda x: -g[x][1]):
            c, t = g[kk2]
            L.append(f"| {kk2 or '—'} | {c} / {t} | {c / t:.1%} |")
        L.append("")

    brk("standard", "按规范")
    brk("type", "按题型")
    brk("difficulty", "按难度")

    miss = [p for p in pos if not p["rank"]]
    if miss:
        L.append("## 未命中明细")
        L.append("")
        for p in miss:
            L.append(f"- **{p['qid']}**（{p['standard']} · {p['type']}）"
                     f" 实际返回：{'；'.join(p.get('top') or ['（空）'])}")
        L.append("")

    hit1 = sum(1 for p in pos if p["rank"] == 1)
    L.append("## 名次分布")
    L.append("")
    L.append("| 名次 | 条数 |")
    L.append("|---|---|")
    dist = collections.Counter(p["rank"] or 0 for p in pos)
    for r in sorted(dist):
        L.append(f"| {'未命中' if r == 0 else '第 ' + str(r) + ' 位'} | {dist[r]} |")
    if m["n_pos"]:
        L.append(f"\n（第 1 位命中 {hit1} 条，占 {hit1 / m['n_pos']:.1%}）")
    L.append("")
    return "\n".join(L)


def load_cfg():
    cfg = {"base": os.environ.get("DIFY_BASE", "https://api.dify.ai/v1"),
           "api_key": os.environ.get("DIFY_API_KEY", ""),
           "dataset_id": os.environ.get("DIFY_DATASET_ID", "")}
    f = os.path.join(BASE_DIR, "config.json")
    if os.path.exists(f):
        with open(f, encoding="utf-8") as fh:
            cfg.update({k: v for k, v in json.load(fh).items() if v})
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["dify", "chroma"], default="dify")
    ap.add_argument("--profile", default="base", choices=sorted(PROFILES))
    ap.add_argument("--all", action="store_true", help="跑全部参数组（仅 dify 后端）")
    ap.add_argument("--only-type", default=None,
                    help="keyword/semantic/numeric/table/unanswerable")
    ap.add_argument("--only-standard", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--delay", type=float, default=None,
                    help="每次请求间隔秒；默认 dify=6.5（10次/分钟），chroma=0")
    ap.add_argument("--out-tag", default="", help="结果文件后缀，便于区分多轮实验")
    args = ap.parse_args()

    items = load_eval(args.only_type, args.only_standard, args.limit)
    if not items:
        raise SystemExit("评测集为空（检查 --only-type / --only-standard）")
    print(f"评测集：{len(items)} 条")

    if args.backend == "chroma":
        backend = ChromaBackend({}, "local", args.top_k)
        thr = 0.5
        cfg_note = f"本地 Chroma · {MODEL} · cosine"
    else:
        cfg = load_cfg()
        if not cfg["api_key"] or not cfg["dataset_id"]:
            raise SystemExit("缺配置：请设 DIFY_API_KEY / DIFY_DATASET_ID 环境变量，"
                             "或在本目录建 config.json（已 gitignore）")
        backend = DifyBackend(cfg, args.profile, args.top_k)
        thr = PROFILES[args.profile].get("score_threshold") or 0.5
        cfg_note = f"Dify hit-testing · {PROFILES[args.profile].get('search_method')}"

    delay = args.delay
    if delay is None:
        delay = 0.0 if args.backend == "chroma" else 6.5
    if delay:
        est = delay * (len(items) - 1) / 60
        print(f"  限速 {delay}s/次 → 预计 {est:.1f} 分钟")

    details = run(backend, items, delay)
    m = summarize(details, backend.top_k, thr)

    tag = args.out_tag or args.profile
    stem = f"结果_{args.backend}_{tag}"
    with open(os.path.join(BASE_DIR, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump({"profile": tag, "backend": args.backend, "metrics": m,
                   "details": details}, f, ensure_ascii=False, indent=1)
    md = md_report(tag, args.backend, m, details, cfg_note)
    with open(os.path.join(BASE_DIR, stem + ".md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(md)

    print()
    print("=" * 56)
    for k in (1, 3, 5, 10):
        if f"recall@{k}" in m:
            print(f"  Recall@{k:<2d} {m[f'recall@{k}']:.1%}")
    print(f"  MRR       {m['mrr']:.3f}")
    print(f"  未命中    {m['miss']} / {m['n_pos']}")
    print(f"  负例最高分 {m['neg_max']:.3f}（超阈 {m['neg_over_thr']} / {len(details['neg'])}）")
    if m.get("thr_window"):
        lo, hi = m["thr_window"]
        if hi > lo:
            print(f"  正例金标 5% 分位 {hi:.3f} → 阈值窗口 ({lo:.3f}, {hi:.3f}]"
                  f"，建议 {m['thr_suggest']:.2f}")
        else:
            print(f"  正例金标 5% 分位 {hi:.3f} ＜ 负例最高分 {lo:.3f} → **阈值窗口不存在**")
            print("        （负例分数没低于正例）⇒ 不能只靠阈值拒答，"
                  "必须在 Prompt 层加拒答约束，或先换 rerank / 调权重。")
    print("=" * 56)
    print(f"→ {stem}.md / {stem}.json")


if __name__ == "__main__":
    main()
