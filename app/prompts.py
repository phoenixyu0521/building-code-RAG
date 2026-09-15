# -*- coding: utf-8 -*-
"""提示词层 —— 本项目「防编造」的第一道闸门。

⚠️ RULES 这段文字与 Dify 版**逐字一致**，是经过实测的（v2 demo 3/3 条文号、2/2 拒答）。
   改一个字都要重新验收，别随手润色。

自检（会把本地这段文字和《Dify导入说明.md》第三节的原文逐字对比）：
    python -m app.prompts
"""
import os
import re

from app.config import ROOT

# ---------------------------------------------------------------------------
# 7 条铁律（system 消息）。与 输出文档/Dify导入说明.md「三、防止模型编造」代码块逐字一致。
# ---------------------------------------------------------------------------
RULES = """你是建筑规范查询助手。你的唯一依据是下面提供的【检索片段】。
1. 只依据【检索片段】回答，禁止使用片段之外的任何知识、经验或推测。
2. 每个回答必须标注出处，格式：【规范名称+版本】条文号。如【GB 50352-2019】2.0.16
3. 条文号必须逐字出现在片段中，不得推断、拼接、补全。
4. 若片段中没有任何与问题相关的内容，只回复：本知识库未收录该内容。
5. 若片段中缺少具体数值，明确说"该处数值在收录内容中缺失"，绝不用常识补齐。
6. 不得把不同规范的条文合并成一条回答。
7. 涉及图示、设备型号等知识库未覆盖的内容，直接说明未收录。"""

# 检索为空时的兜底回答（与铁律第 4 条同文，保证口径一致）
NO_HIT_REPLY = "本知识库未收录该内容。"


def render_hits(hits):
    """把检索结果渲染成给模型看的【检索片段】文本。

    为什么每条片段都要带上「规范名 + 版本 + 条文号」：
    铁律第 2、3 条要求模型标注出处、且条文号必须逐字出现在片段中。
    如果片段里只有正文、没有条文号，模型要么编一个，要么干脆不标 —— 两条都会违规。
    所以出处信息必须由**我们**塞进片段，而不是指望模型自己想出来。
    """
    lines = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        head = (f"【片段 {i}】【{m.get('standard_name', '?')} "
                f"{m.get('standard', '?')}（{m.get('version', '?')}）】"
                f"条文号 {m.get('article_no', '?')} "
                f"［{m.get('status', '')}；{m.get('force_status', '')}］")
        lines.append(head)
        lines.append(h["text"].strip())
        lines.append("")                      # 片段之间空一行，便于模型切分
    return "\n".join(lines).strip()


def build_messages(question, hits):
    """拼出 chat 接口要的 messages。

    刻意**不用**字符串模板去替换占位符 —— 铁律原文里有【】、引号、冒号，
    任何模板引擎都可能把某个字符当成语法。拆成 system + user 两条消息，
    RULES 就能原封不动地传进去。
    """
    user = (f"【检索片段】\n{render_hits(hits)}\n\n"
            f"【用户问题】\n{question}")
    return [
        {"role": "system", "content": RULES},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# 自检：把 RULES 与《Dify导入说明.md》第三节的原文逐字对比
# ---------------------------------------------------------------------------
def _extract_rules_from_doc():
    """从 Dify导入说明.md 里抽出「三、防止模型编造」下面的第一个代码块。"""
    path = os.path.join(ROOT, "输出文档", "Dify导入说明.md")
    if not os.path.isfile(path):
        return None, f"找不到文档：{path}"
    text = open(path, encoding="utf-8").read()
    sec = re.search(r"^##\s*三、防止模型编造.*$", text, re.M)
    if not sec:
        return None, "文档里找不到「三、防止模型编造」小节标题"
    block = re.search(r"```[a-z]*\n(.*?)\n```", text[sec.end():], re.S)
    if not block:
        return None, "该小节后面找不到代码块"
    return block.group(1), None


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 66)
    print("本地提示词 RULES：")
    print("=" * 66)
    print(RULES)
    print("=" * 66)

    doc_rules, err = _extract_rules_from_doc()
    if err:
        print(f"⚠️  无法自动比对：{err}")
        raise SystemExit(1)

    if doc_rules.strip() == RULES.strip():
        print("✅ 与《Dify导入说明.md》第三节**逐字一致**（共 "
              f"{len(RULES.splitlines())} 行）")
    else:
        print("❌ 不一致！下面是差异（- 文档原文 / + 本地）")
        import difflib
        for line in difflib.unified_diff(doc_rules.strip().splitlines(),
                                         RULES.strip().splitlines(),
                                         "Dify导入说明.md", "app/prompts.py",
                                         lineterm=""):
            print("   " + line)
        raise SystemExit(2)
