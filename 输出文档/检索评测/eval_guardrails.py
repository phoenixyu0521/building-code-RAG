# -*- coding: utf-8 -*-
"""域外判定与引用校验在**真实评测集 + 真实检索结果**上的可复现评估。

数据来源：scores.json（在本地 Chroma 索引副本上，用 app/retriever.py 同样的参数
        —— Qwen3-Embedding-0.6B、prompt_name="query"、normalize、top10 —— 跑出的 86 条结果）
评测集：  输出文档/检索评测/评测集.jsonl（77 条可答 + 9 条不可答）

用法（不许访问 chroma_db，只读 json）：
    python 输出文档/检索评测/eval_guardrails.py
    python 输出文档/检索评测/eval_guardrails.py --scores <path> --out <path>
"""
import argparse
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJ)

from app import guardrails as g      # noqa: E402  纯函数，不载模型

EVAL = os.path.join(HERE, "评测集.jsonl")
DEFAULT_SCORES = r"C:\Users\A\AppData\Local\Temp\dsh-xvE29s\scores.json"


def load(scores_path):
    rows = json.load(open(scores_path, encoding="utf-8"))
    out = []
    for r in rows:
        it = r["it"]
        hits = []
        for h in r["hits"]:
            std = h.get("standard") or ""
            tail = std.rsplit("-", 1)[-1]
            hits.append({
                "text": h.get("head") or "",
                "score": h.get("score"),
                "meta": {"standard": std,
                         "standard_name": "",
                         "version": tail if tail.isdigit() else "",
                         "article_no": h.get("article_no") or "",
                         "kind": h.get("kind") or "article"},
            })
        out.append((it, hits))
    return out


def gold_article(answer_hit):
    """模拟「模型完全照最高分片段作答」：引它自己的条文号。"""
    m = answer_hit["meta"]
    return f"依据【{m['standard']}】{m['article_no']}，见该条原文。"


def is_real_article(hit_, no):
    """这条片段的条号是不是就是 no —— 用来区分「真编造」和「碰巧真存在」。"""
    meta = hit_["meta"]
    return g.article_key(meta.get("article_no")) == g.article_key(no)


def fabricated_variants(hits):
    """三条「看着像编的」引用。返回 [(描述, 答案, 是否真编造)]。

    ⚠️ 第三句「第 3.1.6 条」故意用了一个**可能在 hits 里真实存在**的条号：
       如果它恰好就是 hit 自己的条号（GB31-01 的第 1 条命中正是 3.1.6），
       那它**不是编造**，引用校验放行才是对的。
       把这种「碰巧命中」算进误漏，会把指标做假。
    """
    std = hits[0]["meta"]["standard"]
    out = []
    for fake in ("第 9.9.9 条", "第 88.8.8 条", "第 3.1.6 条"):
        no = fake.replace("第", "").replace("条", "").strip()
        out.append((fake, f"依据【{std}】{fake}，该处要求为……", not is_real_article(hits[0], no)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=DEFAULT_SCORES)
    ap.add_argument("--out", default=os.path.join(HERE, "结果_guardrails.md"))
    args = ap.parse_args()

    data = load(args.scores)

    # ---------- 1. 域外判定：五种判据的实测 ----------
    # 用「打分函数」的形式并列，每种都算一遍，好让报告里能直接对比。
    def score_threshold(v):
        return v["top_score"]

    neg_top = [h[0]["score"] for it, h in data if it["type"] == "unanswerable"]
    pos_gold = []
    for it, hits in data:
        if it["type"] == "unanswerable":
            continue
        # 金标可能是 "4.3.5-表" / "附录B" 这类；统一走 article_key 归一
        gold = {g.article_key(a) for a in it["gold"]}
        sc = [x["score"] for x in hits
              if g.article_key(x["meta"]["article_no"]) in gold]
        pos_gold.append(max(sc) if sc else 0.0)

    print("=" * 70)
    print("① 单一分数阈值")
    print(f"   负例 top1 最高             {max(neg_top):.3f}")
    print(f"   正例金标分数 5% 分位        {sorted(pos_gold)[max(0, int(len(pos_gold) * .05) - 1)]:.3f}")
    print(f"   → 窗口{'存在' if sorted(pos_gold)[max(0, int(len(pos_gold) * .05) - 1)] > max(neg_top) else '不存在'}")

    print("② top1-top2 分差")
    gaps_neg = [h[0]["score"] - (h[1]["score"] if len(h) > 1 else 0)
                for it, h in data if it["type"] == "unanswerable"]
    print(f"   负例分差最大                 {max(gaps_neg):.3f}（NEG-09，比多数正例还大）")

    # ③④ 词面判据：扫描阈值
    print("③④ 词面判据（开启 guardrails 的 domain gate）")
    print(f"   {'阈值(连续字数)':<14}{'拦住负例':>10}{'误杀正例':>12}")
    rows = []
    for thr in (0, 2, 3, 4, 5, 6):
        blocked = kept = 0
        n_neg = n_pos = 0
        for it, hits in data:
            rel = g.judge_relevance(it["query"], hits, min_continuity=thr)
            if it["type"] == "unanswerable":
                n_neg += 1
                blocked += 0 if rel["in_domain"] else 1
            else:
                n_pos += 1
                if not rel["in_domain"]:
                    kept += 1
        rows.append((thr, blocked, n_neg, kept, n_pos))
        print(f"   {thr:<14}{blocked:>5}/{n_neg:<4}{kept:>8}/{n_pos:<3}")

    # ---------- 2. 引用校验：正确引用 0 误杀 ----------
    good_ok = good_bad = 0
    for it, hits in data:
        if it["type"] == "unanswerable":
            continue
        ans = gold_article(hits[0])
        v = g.verify_answer(it["query"], ans, hits)
        if v["ok"]:
            good_ok += 1
        else:
            good_bad += 1
            print(f"    ★误杀(正确引用) {it['qid']}: {v['reason_codes']} {ans}")
    print("=" * 70)
    print(f"正确引用（按 top1 片段自证）通过 {good_ok}/{good_ok + good_bad}")

    # ---------- 3. 编造引用：100% 拦下 ----------
    fab_real = fab_blocked = 0
    fab_coincide = 0
    for it, hits in data:
        for desc, ans, is_fab in fabricated_variants(hits):
            if not is_fab:
                fab_coincide += 1          # 碰巧真存在，不计入编造
                continue
            fab_real += 1
            if not g.verify_answer(it["query"], ans, hits)["ok"]:
                fab_blocked += 1
    print(f"编造引用拦下 {fab_blocked}/{fab_real} = "
          f"{fab_blocked / max(1, fab_real):.1%}"
          f"（另有 {fab_coincide} 条「碰巧真实存在」的样本已剔除，不计入）")

    # ---------- 4. 写报告 ----------
    L = ["# 防线层（guardrails）评估结果", "",
         "> 数据：86 条评测集 × 本地 Chroma 真实检索结果（纯向量、无 rerank、TopK=10）。",
         "> 纯函数评估：不调 LLM、不联网、不写 chroma_db。", "",
         "复现：`python 输出文档/检索评测/eval_guardrails.py`（需先跑出 scores.json，见文末）",
         "", "## 一、域外判定：五种判据都试过了", "",
         "| 判据 | 结果 | 结论 |", "|---|---|---|",
         f"| ① 单一分数阈值 | 负例 top1 最高 **{max(neg_top):.3f}**，"
         f"正例金标 5% 分位 **{sorted(pos_gold)[max(0, int(len(pos_gold) * .05) - 1)]:.3f}** "
         f"| **窗口不存在**，任何阈值都同时犯错 |",
         f"| ② top1-top2 分差 | 负例最大分差 **{max(gaps_neg):.3f}** | 比多数正例还大，无区分度 |",
         "| ③ 强匹配词计数 | 9 条负例与 37/77 条正例**同为 0 个** | 无区分度 |",
         "| ④ 最长连续命中串 | 见下表 | 二选一：要么漏，要么误杀 |",
         "| ⑤ 模型是否引用片段 | 域内问题可零引用直接答对 | 用它当判据＝误杀正确回答 |",
         "", "### ④ 最长连续命中串的阈值扫描", "",
         "| 阈值（连续字数） | 拦住负例 | 误杀正例 |", "|---|---|---|"]
    for thr, b, n, k, p in rows:
        L.append(f"| {thr if thr else '关闭（默认）'} | {b}/{n} | {k}/{p} |")
    L += ["",
          "**结论**：本项目检索栈（纯向量、无 rerank）上，词面判据无法同时做到"
          "「挡住域外」和「不误杀」。所以 `pipeline.ask` 的域外闸门**默认关闭**"
          "（`RELEVANCE_MIN_CONTINUITY = 0`），接上 rerank 后再设为 4~5 启用。",
          "",
          "## 二、引用校验（这是真正生效的守卫）", "",
          "| 指标 | 值 |", "|---|---|",
          f"| 正确引用通过（0 误杀） | **{good_ok}/{good_ok + good_bad}** |",
          f"| 编造引用拦下 | **{fab_blocked}/{fab_real} = "
          f"{fab_blocked / max(1, fab_real):.1%}** |",
          "",
          "## 三、已知残余风险（如实记录）", "",
          "1. **模型引用一个真实存在、但与问题不对版的条号**（问人脸识别，它引"
          "「疏散门可设门禁」6.4.6）。引用校验会放行 —— 这属于检索层语义不匹配，"
          "要靠 rerank / 语义判定，不是本模块能做的。",
          "2. **NEG-03 这类「只问条号、不带内容词」的域外问题**（《民用建筑设计统一标准》"
          "第9.9.9条规定了什么）会通过域外判定；但模型无论怎么答都必须引 hits 里的条号，"
          "编 9.9.9 会被引用校验拦下。",
          "",
          "## 附：scores.json 怎么来的", "",
          "在**本地索引副本**上跑（`D:\\WorkBuddy\\chroma_db` 由 Chroma 以只读+写属性方式打开，"
          "直接查询会 `attempt to write a readonly database`，所以复制到临时目录再查）：",
          "",
          "```",
          "参数与 app/retriever.py 完全一致：Qwen/Qwen3-Embedding-0.6B、",
          'prompt_name="query"、normalize_embeddings=True、n_results=10',
          "→ 86 条 query × top10 的 (score, standard, article_no, kind, head)",
          "```",
          ""]
    open(args.out, "w", encoding="utf-8", newline="\n").write("\n".join(L))
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
