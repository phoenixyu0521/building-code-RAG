# -*- coding: utf-8 -*-
"""串联层 —— 问题 → 检索 → 域外判定 → 提示词 → 生成 → 引用校验，一步到位。

这一层只做「编排」，不写业务细节：检索交给 retriever，措辞交给 prompts，
调用交给 llm，**生成之后的核对交给 guardrails**。所以它很短 —— 短到一眼能看完，
这就是分层的意义。

三步闸门（顺序不能换）
----------------------
① 检索 0 条           → 不调模型，直接拒答（连钱都不花）
② 域外判定（guardrails.judge_relevance）
                      → 命中了但不是这个问题的内容，也不调模型
③ 生成后引用校验（guardrails.verify_answer）
                      → 模型说的话不算数，按 hits 逐条核对；不过就降级

为什么②要放在生成之前
    实测「负例最高分 0.727 > 正例 5% 分位 0.675」，单阈值挡不住域外问题。
    但域外问题一旦生成，模型就会顺着片段编（NEG-02：问「人脸识别门禁怎么设计」，
    检索命中「疏散门可设门禁」→ 模型答出一整套人脸识别方案）。
    所以「这次检索到底是不是在回答这个问题」要在生成前判，而不是生成后补救。

自检（要 API Key）：
    python -m app.pipeline "图书馆阅览室到最近疏散门的最大距离是多少"
"""
import sys

from app import guardrails, prompts
from app.llm import chat
from app.retriever import Retriever

# 域外判定开关：0 = 不拦截（默认，理由见 guardrails.judge_relevance 的长注释）
# 接上 rerank / 语义层判定之后改成正整数（4~5）即可启用。
RELEVANCE_MIN_CONTINUITY = 0


def ask(question, top_k=None, where=None, allow_llm=True,
        min_continuity=None):
    """回答一个规范问题。

    返回 dict：
      question      原问题
      hits          检索到的片段（list）
      answer        最终回答文本（可能是降级结果）
      used_llm      是否真的调用了大模型 —— 语义不变：只有真调了才是 True
      verification  引用校验结果 + 采用的策略（见 guardrails.verify_answer）
      degraded      答案是否被降级（抽取式 / 因不合格而拒答）
      relevance     域外判定结果（默认 gate=off，只记录不断言）
    """
    retriever = Retriever()
    hits = retriever.search(question, top_k=top_k, where=where)

    # 🔑 闸门①：**一条都没检索到就直接拒答，根本不调大模型。**
    #    理由有两条：
    #      ① 没有片段时，模型只能靠常识编 —— 正是本项目要防的事；
    #      ② 不调用 = 一定不会编，还省一次 API 费用。
    if not hits:
        v = guardrails.verify_answer(question, prompts.NO_HIT_REPLY, [])
        return {
            "question": question, "hits": [], "used_llm": False,
            "answer": prompts.NO_HIT_REPLY,
            "degraded": True,
            "verification": _seal(v, "refuse", True),
            "relevance": {"in_domain": False, "longest_run": 0, "n_hits": 0,
                          "reasons": ["检索 0 条"], "gate": "off"},
        }

    # 🔑 闸门②：域外判定。默认关闭 —— 实测在现有检索栈上没有可用的词面判据
    #    （依据写在 guardrails.py「结论」一节和评测报告里）。
    mc = RELEVANCE_MIN_CONTINUITY if min_continuity is None else int(min_continuity)
    relevance = guardrails.judge_relevance(question, hits, min_continuity=mc)
    if not relevance["in_domain"]:
        v = guardrails.verify_answer(question, prompts.NO_HIT_REPLY, hits)
        return {
            "question": question, "hits": hits, "used_llm": False,
            "answer": prompts.NO_HIT_REPLY,
            "degraded": True,
            "verification": _seal(v, "refuse", True),
            "relevance": relevance,
        }

    if not allow_llm:
        # 只检索不生成：没有模型答案，自然没有可校验的引用。
        # 仍然给出 verification 结构（ok=False），保持返回形状统一。
        v = guardrails.verify_answer(question, "", hits)
        return {
            "question": question, "hits": hits, "used_llm": False,
            "answer": "(--no-llm：只检索不生成，片段见下)",
            "degraded": False,
            "verification": _seal(v, "no_llm", False),
            "relevance": relevance,
        }

    messages = prompts.build_messages(question, hits)
    text, usage = chat(messages)
    text = (text or "").strip()

    # 🔑 闸门③：生成后引用校验 + 降级。
    verification = guardrails.verify_answer(question, text, hits)
    decision = guardrails.apply_policy(text, verification, hits, relevance)

    return {
        "question": question,
        "hits": hits,
        "used_llm": True,                 # ← 语义不变：确实调了模型
        "answer": decision["answer"],
        "degraded": decision["degraded"],
        "verification": _seal(verification, decision["strategy"],
                              decision["degraded"], decision["sources"]),
        "relevance": relevance,
        "usage": usage,
    }


def _seal(v, strategy, degraded, sources=None):
    """把「校验结果」和「最后采用了什么策略」合并成一个 dict。

    为什么要合并：校验说「不合格」（ok=False）和「降级成什么」是两件事。
    只看 ok 的话，抽取式回答也是 ok=False —— 但它已经不是模型的瞎编了。
    对外只暴露这一个结构，调用方不必自己拼。
    """
    out = dict(v)
    out["strategy"] = strategy
    out["degraded"] = bool(degraded)
    if sources is not None:
        out["sources"] = sources
    return out


def format_result(result):
    """把 ask() 的结果打印成人看的样式：先答案，再溯源，最后校验结论。"""
    out = []
    out.append("=" * 66)
    out.append(f"❓ {result['question']}")
    out.append("=" * 66)
    out.append("💬 " + result["answer"])

    v = result.get("verification") or {}
    rel = result.get("relevance") or {}
    out.append("")
    out.append("-" * 66)
    if v.get("degraded"):
        out.append(f"⚠️ 已降级（策略 {v.get('strategy')}）："
                   f"{'；'.join(v.get('reasons') or []) or '见上'}")
    elif v.get("strategy") == "keep":
        out.append(f"✅ 引用校验通过（覆盖率 {v.get('coverage')}）")
    if rel:
        out.append(f"🎯 域外判定：{'域内' if rel.get('in_domain') else '域外'}"
                   f"（覆盖度 {rel.get('coverage')}，top {rel.get('top_score')}）")

    if result["hits"]:
        out.append("")
        out.append("-" * 66)
        out.append("📎 检索来源（用于核对条文号，也是「引用」块的内容）：")
        for i, h in enumerate(result["hits"], 1):
            m = h["meta"]
            out.append(f"  [{i}] {h['score']:.4f}  《{m.get('standard_name', '?')}》"
                       f"{m.get('standard', '?')}  第 {m.get('article_no', '?')} 条"
                       f"  ［{m.get('status', '')}］")
            if not result["used_llm"]:
                out.append(f"      {h['text'].strip()[:180]}")
    else:
        out.append("")
        out.append("（检索 0 条 → 未调用大模型，直接拒答）")
    if result.get("usage"):
        out.append("")
        out.append(f"（用量：{result['usage']}）")
    return "\n".join(out)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    qs = sys.argv[1:] or ["图书馆阅览室到最近疏散门的最大距离是多少"]
    for q in qs:
        print(format_result(ask(q)))
        print()
