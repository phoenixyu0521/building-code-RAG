# -*- coding: utf-8 -*-
"""防线层 —— 生成之后的**代码级**引用校验。

为什么要有这一层
----------------
提示词（`app/prompts.py` 的七条铁律）只能「劝」模型别编，拦不住它。
实测提示词能让 3/3 的答案条文号正确，但那是在模型**愿意**听话的前提下。
这一层的职责是：**模型说了什么不算数，我们按 hits 逐条核对它说的每条引用。**
核对不过就走降级（抽取式回答或拒答），绝不让未支撑的条文号流出。

三个纯函数
----------
verify_answer(question, answer, hits)   答案里每条引用能不能在 hits 里对上
judge_relevance(question, hits)         这次检索结果是不是真的「有关」
apply_policy(answer, verification, hits, relevance)  不合格时怎么降级

为什么全是纯函数
----------------
不 import torch / chromadb / 网络，也不读配置。
所以能在 tests/ 里毫秒级跑完，也能在 API 线程里每答一次跑一次。

⚠️ 归一化的红线：**只允许「等价写法」归一，绝不允许「近似」归一。**
   3.1.6 与 3.16 是两条不同的条文；跨规范引用即使条文号对得上也不算数。
"""
import re

# 与 prompts.NO_HIT_REPLY 同文。这里**不能** import prompts：
# prompts 会 import config，config 会读 .env —— 防线层要保持零副作用。
NO_HIT_REPLY = "本知识库未收录该内容。"

# 铁律第 4 条的原文，模型拒答时会原样回复它
REFUSAL_PHRASES = ("本知识库未收录该内容",)

# 空引用时给的降级答复（抽取不到任何可用片段时用）
_EXTRACT_HEAD = "【降级·抽取式回答】模型生成的回答未通过引用校验，以下直接引用检索片段原文："


# ===========================================================================
# 一、归一化
# ===========================================================================
_STD_RE = re.compile(
    r"(?<![A-Za-z])"                       # 前面不能紧贴字母（避免把单词切开）
    r"(GBJ|GB|JGJ|CJJ|DB\d{2}|T)"          # 规范前缀；T 单独出现时交给下面排除
    r"\s*(/(?:\s*T)?)?\s*"                 # 可选「/T」推荐性标准后缀
    r"(\d{2,5})"
    r"(?:\s*[-—–]\s*(\d{4}))?",            # 可选年号
)

# 规范号 → 已知版本（hits 里出现过才算数；这张表只用于「补全缺省年号」的容错）
_DASHES = str.maketrans({"－": "-", "—": "-", "–": "-", "‐": "-", "−": "-"})


def normalize_standard(s, known_versions=None):
    """把规范号统一成 ``GB 50352-2019`` 这种形式。

    "GB50352-2019" / "gb 50352—2019" / "GB/T50352-2019" 都能归一。
    没有年号时，用 ``known_versions``（本次 hits 里出现过的规范号）补全；
    补不上的保留无年号形式，校验时按「前缀匹配」处理。
    """
    if not s:
        return ""
    t = str(s).translate(_DASHES).upper()
    t = re.sub(r"\s+", " ", t).strip()
    m = _STD_RE.search(t)
    if not m:
        return t
    prefix, slash_t, num, year = m.groups()
    if prefix == "T":
        # 单独一个 T 不是规范前缀
        return t
    head = f"{prefix}{'/T' if slash_t else ''} {num}"
    if year:
        return f"{head}-{year}"
    for cand in known_versions or ():
        c = normalize_standard(cand)
        if c.startswith(head + "-"):
            return c
    return head


def article_key(no):
    """条文号归一成 key。

    "第 3.1.6 条" → "3.1.6"；"表4.3.5-1" → "4.3.5"；"附录A" → "附录A"。
    ⚠️ 只去壳，不改数字：3.16 还是 3.16，不会变成 3.1.6。
    """
    if no is None:
        return ""
    t = str(no).strip()
    t = t.replace("．", ".").replace("。", ".")
    # ⚠️ 附录要先判：「附录 B」去掉前缀只剩一个 "B"，顺序反了就认不出来
    m = re.match(r"^(?:附\s*[表录]|附录)\s*([A-Za-z]|[一二三四五六七八九十]+)", t)
    if m:
        gp = m.group(1)
        return f"附录{gp.upper()}" if gp.isalpha() else f"附录{gp}"
    t = re.sub(r"^(第|表|附表|条文号)", "", t)
    t = re.sub(r"(条|款|项)$", "", t)
    t = t.strip()
    m = re.match(r"^([A-Za-z])\s*-?\s*(\d+)$", t)          # 裸的附录编号 B / B-1
    if m:
        return f"附录{m.group(1).upper()}"
    m = re.match(r"^(\d+(?:\.\d+)+)", t)                   # 3.1.6 / 4.2.4-1
    if m:
        return m.group(1)
    m = re.match(r"^(\d+)\s*[-—]\s*(\d+)$", t)             # 3.1
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return t


_ART_LABELLED_RE = re.compile(
    r"(?:第\s*)?(\d{1,2}(?:\s*[.．]\s*\d{1,3}){1,3})"      # 至少两级：3.1.6
    r"(?:\s*[.．]\s*条)?"                                  # 「3.1.6.条」这种笔误
    r"\s*条"                                               # 必须以「条」收尾
    r"|第\s*(\d{1,2}(?:\s*[.．]\s*\d{1,3}){1,3})"          # 「第 3.1.6」
)

_ART_BARE_RE = re.compile(
    r"(?:表|附表)?\s*(\d{1,2}(?:\s*[.．]\s*\d{1,3}){1,3})(?:\s*-\s*\d{1,2})?"
)

_APPENDIX_RE = re.compile(r"(?:附\s*[表录]\s*([A-Za-z])\b|附录\s*([一二三四五六七八九十]+))")


def extract_article_nos(text):
    """从一段话里抽出所有条文号（已归一）。返回 set。

    三种写法都认：``第 3.1.6 条`` / ``3.1.6`` / ``表 4.3.5-1``。
    只认**带小数点**的数字（至少两级），所以 "2.40m"、"100万册"、"2019年"
    这类量和日期不会被误抽 —— 这正是「正确引用 0 误杀」的前提。
    """
    if not text:
        return set()
    t = str(text).replace("．", ".")
    out = set()

    for m in _ART_LABELLED_RE.finditer(t):
        raw = m.group(1) or m.group(2)
        key = article_key(raw)
        if key:
            out.add(key)

    # 裸数字：去掉规范号自身（GB 50352-2019 的 50352 不是条文号），
    # 再要求至少两级小数 —— 一级数字（"3"）不当条文号。
    stripped = _STD_RE.sub(" ", t)
    stripped = re.sub(r"\d{4}\s*年", " ", stripped)          # 年份
    stripped = re.sub(r"\d+(?:\.\d+)?\s*(?:m|mm|cm|℃|%|kg|级|座|人|册|米|米²|米2)(?![.\d])",
                      " ", stripped)                        # 带单位的量
    for m in _ART_BARE_RE.finditer(stripped):
        key = article_key(m.group(1))
        if key and "." in key:
            out.add(key)

    for m in _APPENDIX_RE.finditer(t):
        if m.group(1):
            out.add(f"附录{m.group(1).upper()}")
    return out


def extract_standards(text):
    """抽出文中出现的规范号（归一后）。"""
    if not text:
        return set()
    t = str(text).translate(_DASHES)
    return {normalize_standard(m.group(0)) for m in _STD_RE.finditer(t)}


# ===========================================================================
# 二、hits 侧：建索引
# ===========================================================================
def _meta(h):
    return h.get("meta") or {}


def _hit_standard(h):
    m = _meta(h)
    return normalize_standard(m.get("standard") or "")


def _article_index(hits):
    """{(规范号, 条文key): 最高分} —— 记录每条引用可能的出处。

    表格片段（article_no = "4.3.5-表"）同时登记它的正文条号 4.3.5 ——
    表格与正文条例在规范里是一体的，模型写哪个都算对。
    """
    idx = {}
    for h in hits:
        m = _meta(h)
        std = _hit_standard(h)
        keys = {article_key(m.get("article_no"))}
        raw = str(m.get("article_no") or "")
        if "-" in raw:                       # 4.3.5-表 / 4.3.5-1
            keys.add(article_key(raw.split("-")[0]))
        score = float(h.get("score") or 0.0)
        for k in keys:
            if not k:
                continue
            if score > idx.get((std, k), -1):
                idx[(std, k)] = score
    return idx


# ===========================================================================
# 三、核心校验
# ===========================================================================
def _sentence_cites(answer):
    """逐句找出「这一句引用了哪条条文、说的是哪本规范」。

    为什么按句而不按整段：跨规范张冠李戴（铁律第 6 条）只看整段是看不出来的 ——
    整段里两本规范都出现过，只有落到某一句，才知道谁配谁。
    """
    pieces = re.split(r"[。；;\n]", answer)
    out = []
    for p in pieces:
        if not p.strip():
            continue
        arts = extract_article_nos(p)
        stds = extract_standards(p)
        out.append((p, arts, stds))
    return out


def verify_answer(question, answer, hits):
    """核对答案里的每条引用是否都被本次 hits 支撑。

    返回
    ----
    {
      ok                  : bool，能不能直接返回这个答案
      reasons             : 人话理由列表（可直接塞进日志或接口）
      reason_codes        : 机器可读的原因码
      cited               : 被判定为「已核实」的引用 [{"standard","article_no"}]
      unverified_numbers  : 判定不了的条文号（编造 / 不在 hits / 出处不对）
      coverage            : 已核实引用数 / 抽到的条文号总数（无条文号时为 1.0）
      standards_cited     : 答案里出现的全部规范号
      refusal_only        : 答案本身就是铁律第 4 条的拒答
    }
    """
    answer = "" if answer is None else str(answer)
    reasons, codes = [], []

    if not answer.strip():
        codes.append("empty_answer")
        reasons.append("答案为空")
        return _result(False, reasons, codes, [], [], [], 0.0, False)

    # 模型原样拒答：不是「编造」，是「该拒答」
    if any(p in answer for p in REFUSAL_PHRASES) and \
            len(answer.strip()) <= max(len(p) for p in REFUSAL_PHRASES) + 20:
        return _result(False, ["模型按铁律拒答"], ["model_refused"], [], [], [], 1.0, True)

    idx = _article_index(hits)
    stds_cited = sorted(extract_standards(answer))

    cited, unverified = [], []
    total = 0

    # 逐句核对：句子是「引用与规范配对」的最小单位
    for _sentence, arts, stds in _sentence_cites(answer):
        for a in sorted(arts):
            total += 1
            owners = sorted({s for (s, k) in idx if k == a})
            if not owners:
                unverified.append(a)
                codes.append("not_in_hits")
                reasons.append(f"条文号 {a} 不在本次检索片段里")
                continue

            # 本句点名的规范（若有）
            named = {s for s in stds if s}
            if named:
                ok_owners = named & set(owners)
                if not ok_owners:
                    unverified.append(a)
                    codes.append("standard_mismatch")
                    reasons.append(
                        f"第 {a} 条出自 {'、'.join(owners)}，答案却标成 "
                        f"{'、'.join(sorted(named))}")
                    continue
                cited.append({"standard": sorted(ok_owners)[0], "article_no": a})
                continue

            if len(owners) == 1:
                cited.append({"standard": owners[0], "article_no": a})
                continue

            # 条文号在多本规范里都有，句子又没说哪本 → 无法核实
            unverified.append(a)
            codes.append("ambiguous_article")
            reasons.append(
                f"第 {a} 条在 {'、'.join(owners)} 里都有，答案没有指明出处")

    # 铁律第 6 条（不得把不同规范的条文合并成一条回答）：
    # ⚠️ 这一条**故意不在代码里判**。合规答案经常同时引用两三本规范的条文
    #    （实测 GB52-02 / GB52-03 / GB52-07 的金标本身就是跨规范的），
    #    机器无法区分「按问题分工引用」和「张冠李戴式拼接」——
    #    一刀切会把正确引用整片误杀，这违背「正确引用 0 误杀」的红线。
    #    张冠李戴的实质风险由上面的 standard_mismatch 承担：把 A 规范的条文
    #    标成 B 规范，在代码层必拦。真正的「拼接」留在提示词第 6 条约束。

    # 铁律第 2 条：必须标注出处
    if total == 0:
        codes.append("no_citation")
        reasons.append("答案没有标注任何条文号，无法核实出处")

    coverage = (len(cited) / total) if total else (1.0 if not codes else 0.0)
    ok = not codes
    return _result(ok, reasons, codes, cited, unverified, stds_cited,
                   round(coverage, 4), False)


def _result(ok, reasons, codes, cited, unverified, stds, coverage, refusal_only):
    # 去重但保序（便于日志阅读）
    seen, r2 = set(), []
    for x in reasons:
        if x not in seen:
            seen.add(x)
            r2.append(x)
    c2 = []
    for x in codes:
        if x not in c2:
            c2.append(x)
    return {
        "ok": bool(ok),
        "reasons": r2,
        "reason_codes": c2,
        "cited": cited,
        "unverified_numbers": unverified,
        "coverage": coverage,
        "standards_cited": stds,
        "refusal_only": refusal_only,
    }


# ===========================================================================
# 四、域外判定（relevance）
# ===========================================================================
# 结论（详细依据见 输出文档/检索评测/结果_guardrails.md）
# ----------------------------------------------------
# **在本项目的检索栈上（纯向量、无 rerank、TopK=10），「域外判定」做不到好用。**
# 四种判据都在 86 条评测集上实测过，没有一种能同时满足「不误杀」和「挡得住」：
#
#   ① 单一分数阈值        负例 top1 最高 0.727 > 正例金标 5% 分位 0.675 → 窗口不存在
#   ② top1 - top2 分差    负例 NEG-09 分差 0.180，比多数正例还大 → 无区分度
#   ③ 强匹配词计数        9 条负例和 37/77 条正例**都是 0 个** → 无区分度
#   ④ 问句最长连续命中串  阈值 4 字：负例挡住 5/9，正例误杀 9/77；
#                        阈值 5 字：负例挡 6/9，正例误杀 35/77
#   ⑤ 模型是否引用了片段  域内问题可以零引用直接答对（「答案就在语料里」），
#                        用它当判据等于误杀正确回答 —— 违反红线
#
# 根因是语言：负例问的是「人脸识别/充电桩/碳排放」，正例问的是「净高/疏散门」，
# 而这些词**用字面匹配**区分不开 —— 负例里同样有「门禁」「停车场」「声学」这类
# 规范里确实存在的字眼。所以这里的决定性判据不能是词面。
#
# 所以：**不在这里做自动拦截**。域外问题改由「引用校验」兜住（见下），
# 并在 pipeline 里保留一个**默认关闭**的开关 `relevance_gate`，留作接 rerank 之后启用。
# 真正让域外问题答不出来的，是下面这条链条：
#
#   域外问题 → 检索命中的条文与问题议题不符 → 模型要么按铁律第 4 条拒答，
#   要么硬答并引用条文号 → **代码核对：这是不是 hits 里真有的条号？**
#     · 编造条号（NEG-03 的 9.9.9）           → 拦住 → 降级/拒答
#     · 引用 hits 里真有的条号，但答非所问     → 引用校验通过，**这是已知残余风险**
#
# 已知残余风险（如实记录，不粉饰）：模型如果引用一个**真实存在**、但和问题
# 不对版的条号（例如问人脸识别，它引「疏散门可设门禁」6.4.6），引用校验会放行。
# 这一类只能靠 rerank / 语义层判定，属于检索层的问题（见审查问题 2 的第 3 点）。

_MIN_HITS_DOMAIN = 1
_DEFAULT_MIN_CONTINUITY = 4     # 见下方说明；默认 0 表示不启用
_MAX_DF = 0.5

# 疑问词 / 功能词：出现在任何一句规范条文里都不代表「这个问题库里有」
_QUESTION_CHARS = set(
    "的了吗呢吧啊什么怎样怎么如何为啥哪些哪个多少能否可否可以应该需要要求规定"
    "是否包括请给出说明一下为什么这那有以为对把被几多个条条款项")
_QUESTION_WORDS = (
    "什么", "怎么", "怎样", "如何", "哪些", "哪个", "多少", "可以", "需要", "应该",
    "要求", "规定", "定义", "包括", "是否", "能否", "可否", "给出", "说明", "一下",
    "为什么", "是什么", "有哪些", "哪个", "条款", "内容", "问题", "两者", "区别",
    "之间", "其中", "以及", "或者", "还是", "是否应", "能否设",
)

# 规范标题里的词：本库所有规范都是"建筑"，问句里出现它不说明任何问题
_TITLE_WORDS = (
    "建筑", "设计", "规范", "标准", "图书馆", "民用", "工程", "技术", "统一",
    "强制性", "通用", "设施", "场所", "规定", "要求", "原文", "条文",
)


def _question_segments(question):
    """问句去掉疑问词/规范号之后，剩下的**实词片段**（按功能字切开）。"""
    t = _STD_RE.sub(" ", str(question or ""))
    for w in _QUESTION_WORDS:
        t = t.replace(w, " ")
    cut = "[" + re.escape("".join(sorted(_QUESTION_CHARS))) + "]"
    segs = []
    for p in re.split(r"[^\u4e00-\u9fffA-Za-z0-9]+", t):
        for s in re.split(cut, p):
            s = s.strip()
            if s:
                segs.append(s)
    return segs


def longest_match_run(question, hits):
    """问句里最长的一段**逐字连续出现在某条命中片段里**的中文串长度。

    为什么用「连续串」而不是「词数」：中文没有词边界，唯一能自动对齐的
    客观单位就是「连续的字」。「疏散门」「净高」「无障碍」这类词一旦真的
    出现在条文里，就是连续串。
    """
    texts = [str(h.get("text") or "") for h in hits]
    if not texts:
        return 0
    best = 0
    for seg in _question_segments(question):
        seg = re.sub(r"[^\u4e00-\u9fff]", "", seg)
        n = len(seg)
        i = 0
        while i < n:
            for j in range(n, i + 1, -1):
                sub = seg[i:j]
                if len(sub) >= 2 and any(sub in t for t in texts):
                    best = max(best, len(sub))
                    i = j - 1
                    break
            i += 1
    return best


def judge_relevance(question, hits, min_continuity=0):
    """这次检索结果与问题是否「真的有关」。

    参数
    ----
    min_continuity : 要求的最长连续命中串长度。``0`` = **不拦截**（默认）。

    为什么默认 0
        见上面「结论」一节：在本项目现有的纯向量检索上，任何词面阈值都要
        在「漏过域外」和「误杀域内」之间二选一，而误杀正确回答是不可接受的。
        接上 rerank / 语义层判定之后，把 min_continuity 设为 4~5 即可启用。

    返回 {in_domain, longest_run, n_hits, top_score, reasons}
    """
    if not hits:
        return {"in_domain": False, "longest_run": 0, "n_hits": 0,
                "top_score": 0.0, "reasons": ["检索 0 条"], "gate": "off"}
    top_score = max(float(h.get("score") or 0.0) for h in hits)
    run = longest_match_run(question, hits)
    reasons = []
    in_domain = True
    if len(hits) < _MIN_HITS_DOMAIN:
        in_domain = False
        reasons.append("命中条数不足")
    if min_continuity and run < min_continuity:
        in_domain = False
        reasons.append(f"问句最长连续命中串 {run} 字 < {min_continuity} 字")
    return {"in_domain": in_domain, "longest_run": run,
            "min_continuity": min_continuity, "n_hits": len(hits),
            "top_score": round(top_score, 4), "reasons": reasons,
            "gate": "on" if min_continuity else "off"}


# ===========================================================================
# 五、降级策略
# ===========================================================================
def extractive_answer(hits, max_chars=400):
    """抽取式回答：直接引用**最高分**那条片段的原文，并标注出处。

    为什么是「最高分一条」而不是「拼几条」：
      拼条数越多，越像在替模型做综合 —— 那正是要避免的事。
      一条原文 + 出处，读者可以自己翻规范核对，最稳。

    返回 (文本, 出处 dict)；没有可用片段时返回 (None, None)。
    """
    for h in sorted(hits, key=lambda x: float(x.get("score") or 0), reverse=True):
        m = _meta(h)
        body = (h.get("text") or "").strip()
        if not body:
            continue
        std = str(m.get("standard") or "").strip()
        no = str(m.get("article_no") or "").strip()
        if not std or not no:
            continue                      # 没有出处的片段不能当抽取式回答
        name = str(m.get("standard_name") or "").strip()
        label = f"《{name}》" if name else ""
        label += std
        if m.get("kind") == "table":
            label += f" 表 {no}"
        elif "-" in no:
            label += f" 表 {no}"
        else:
            label += f" 第 {no} 条"
        body = body if len(body) <= max_chars else body[:max_chars] + "……"
        text = f"{_EXTRACT_HEAD}\n{body}\n——出处：{label}"
        return text, {"standard": std, "article_no": no,
                      "standard_name": name, "score": round(float(h.get("score") or 0), 4)}
    return None, None


def apply_policy(answer, verification, hits, relevance=None):
    """根据校验结果决定最终对外输出。

    策略只有三种，互斥（**不混用**）：
      keep        校验通过 → 原样返回模型答案
      refuse      该拒答   → 返回 NO_HIT_REPLY，不进模型、不给片段
      extractive  不合格   → 丢弃模型答案，改为引用最高分片段原文 + 出处

    「无法抽取则拒答」的边界很明确：hits 为空、或所有片段缺正文/缺出处。
    **绝不**在抽取式回答后面再补一句模型的话 —— 混起来读者分不清哪句有依据。

    返回 {answer, strategy, degraded, sources}
    """
    if answer is None:
        answer = ""

    if verification and verification.get("refusal_only"):
        return {"answer": NO_HIT_REPLY, "strategy": "refuse",
                "degraded": False, "sources": []}

    if verification and verification.get("ok"):
        return {"answer": answer, "strategy": "keep",
                "degraded": False, "sources": verification.get("cited") or []}

    # 域外：连片段本身都不该给，直接拒答
    if relevance is not None and not relevance.get("in_domain", True):
        return {"answer": NO_HIT_REPLY, "strategy": "refuse",
                "degraded": True, "sources": []}

    text, src = extractive_answer(hits)
    if text:
        return {"answer": text, "strategy": "extractive",
                "degraded": True, "sources": [src] if src else []}
    return {"answer": NO_HIT_REPLY, "strategy": "refuse",
            "degraded": True, "sources": []}


def summarize(verification, relevance=None):
    """给接口用的摘要 —— 只含判定结论，不含密钥、路径、模型原文。"""
    v = verification or {}
    out = {
        "ok": bool(v.get("ok")),
        "reason_codes": list(v.get("reason_codes") or []),
        "cited": list(v.get("cited") or []),
        "unverified_numbers": list(v.get("unverified_numbers") or []),
        "coverage": v.get("coverage", 0.0),
        "strategy": v.get("strategy"),
        "degraded": bool(v.get("degraded")),
    }
    if relevance is not None:
        out["relevance"] = {
            "in_domain": relevance.get("in_domain"),
            "coverage": relevance.get("coverage"),
            "n_hits": relevance.get("n_hits"),
            "top_score": relevance.get("top_score"),
        }
    return out
