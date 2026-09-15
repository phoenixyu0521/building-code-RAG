# -*- coding: utf-8 -*-
"""串联层 —— 问题 → 检索 → 提示词 → 生成，一步到位。

这一层只做「编排」，不写业务细节：检索交给 retriever，措辞交给 prompts，
调用交给 llm。所以它很短 —— 短到一眼能看完，这就是分层的意义。

自检（要 API Key）：
    python -m app.pipeline "图书馆阅览室到最近疏散门的最大距离是多少"
"""
import sys

from app import prompts
from app.llm import chat
from app.retriever import Retriever


def ask(question, top_k=None, where=None, allow_llm=True):
    """回答一个规范问题。

    返回 dict：
      question  原问题
      hits      检索到的片段（list）
      answer    最终回答文本
      used_llm  是否真的调用了大模型
    """
    retriever = Retriever()
    hits = retriever.search(question, top_k=top_k, where=where)

    # 🔑 一个刻意的短路：**一条都没检索到就直接拒答，根本不调大模型。**
    #    理由有两条：
    #      ① 没有片段时，模型只能靠常识编 —— 正是本项目要防的事；
    #      ② 不调用 = 一定不会编，还省一次 API 费用。
    if not hits:
        return {"question": question, "hits": [], "used_llm": False,
                "answer": prompts.NO_HIT_REPLY}

    if not allow_llm:
        return {"question": question, "hits": hits, "used_llm": False,
                "answer": "(--no-llm：只检索不生成，片段见下)"}

    messages = prompts.build_messages(question, hits)
    text, usage = chat(messages)
    return {"question": question, "hits": hits, "used_llm": True,
            "answer": text.strip(), "usage": usage}


def format_result(result):
    """把 ask() 的结果打印成人看的样式：先答案，再溯源。"""
    out = []
    out.append("=" * 66)
    out.append(f"❓ {result['question']}")
    out.append("=" * 66)
    out.append("💬 " + result["answer"])
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
