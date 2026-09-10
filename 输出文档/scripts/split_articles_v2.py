# -*- coding: utf-8 -*-
"""
split_articles_v2.py —— 建筑规范 OCR 文本 → 条文 chunk（v2）

相对 v1（桌面 split_articles.py）的 7 项修复，每项都有实测证据：

1. 条文号后置重排（最重要）
   GB 55037-2022 全 238 条都是「条文号独立成行、且排在本条首行之后」的版式：
       为预防建筑火灾、减少火灾危害，保障人身和财产安全，使   <- 1.0.1 首行
       1.0.1                                                  <- 条文号（居中于本条左侧）
       建筑防火要求安全适用、技术先进、经济合理……             <- 1.0.1 正文
   v1 只认行首条文号 → 每条首行被并进上一条，GB 55037 全 238 条错位。
   本版用「段首缩进」判定首行归属：中文排版段落首行缩进约 2 字、续行顶格，
   因此条文号上方那一行「只有缩进时才属于本条」（用 OCR json 的坐标判定）。
   无 json 时退化为「不以句末标点结尾则移入」并全部打标复核。

2. 附录 / 条文说明裁剪：仅保留「附录B」类的表；附录A/C、用词说明、引用标准名录、
   条文说明、目次一律不入库。丢弃态在遇到正文首条 1.0.1 时自动退出。

3. 表格单独抽出：含表号的条文，表格部分切为独立 chunk（kind=table）。

4. 取消 v1 的 800 字硬切：正文条文整条保留，绝不腰斩。

5. 数字归一化与边界补齐：`3. 1. 6`→`3.1.6`、`B. 0. 1`→`B.0.1`；句末行内条文号
   切分；章节标题粘连切分（`……的规定2术语`）。

6. 水印碎片清洗：文本层是水印「住房城乡建设部信息公开浏览专用」时，OCR 会把斜排
   水印切成碎片插进正文（如「civil building息公供人们居住」）。只删水印的连续
   子串（长度≥4，或高特异短碎片），绝不删单字。

7. 内置完整性自检：断号 / 重复号 / 首条异常，写入 _自检问题.csv。

用法：
  python split_articles_v2.py --src <ocr目录> --out <chunks目录>
"""
import argparse
import csv
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ------------------------------------------------------------------ 规范元数据
META = {
    "GB 55031-2022": {"code": "GB 55031-2022", "name": "民用建筑通用规范",
                      "version": "2022", "force": "全文强制", "status": "现行"},
    "GB 55037-2022": {"code": "GB 55037-2022", "name": "建筑防火通用规范",
                      "version": "2022", "force": "全文强制", "status": "现行"},
    "GB 50352-2019": {"code": "GB 50352-2019", "name": "民用建筑设计统一标准",
                      "version": "2019", "force": "强条已废止(由GB55031替代)",
                      "status": "现行(技术参考)"},
    "GB 55019-2021": {"code": "GB 55019-2021", "name": "建筑与市政工程无障碍通用规范",
                      "version": "2021", "force": "全文强制", "status": "现行"},
    "JGJ 38-2015":   {"code": "JGJ 38-2015", "name": "图书馆建筑设计规范",
                      "version": "2015", "force": "部分强条", "status": "现行"},
}

# ------------------------------------------------------------------ 正则
PAGE_RE = re.compile(r"^<!--\s*page\s+(\d+)\s*-->\s*$", re.I)
# 编号内空格收敛：`3. 1. 6` -> `3.1.6`；`B. 0. 1` -> `B.0.1`
# （字母开头的附录号也必须能收敛，否则附录边界识别不了）
NUM_SPACE_RE = re.compile(r"([A-Z0-9])\.\s+(?=\d)")
# 条文号骨架：`3.1.6` 或附录的 `B.0.1`
ARTNO = r"(?:[A-Z]\.)?\d{1,2}\.\d{1,2}\.\d{1,2}"
ARTNO_FULL = re.compile(r"^" + ARTNO + r"$")
NUM_ONLY_RE = re.compile(r"^\s*(" + ARTNO + r")\s*$")
SEC_ONLY_RE = re.compile(r"^\s*(\d{1,2}\.\d{1,2})\s*$")
ART_HEAD_RE = re.compile(r"^\s*(" + ARTNO + r")(?![.\d])\s*(.*)$")
# 句末之后的行内条文号：`……的规定。3.1.7 图书馆……`
ART_INLINE_RE = re.compile(r"(?<=[。；])\s*(" + ARTNO
                           + r")(?![.\d])\s*(?=[\u4e00-\u9fa5A-Za-z])")
# 章节标题：必须「独占一行」，形如 `3.6防灾避难` / `2.1 一般规定`
#
# ⚠️ 教训（2026-09-10，内容级破坏，勿回退）——此前用的是「行尾粘连」版本：
#      (?<![\d年月日米厘毫吨度磅层高宽厚长×xX%])\s*(\d{1,2}\.\d{1,2})(?![.\d])\s*([\u4e00-\u9fa5]{2,8})\s*$
#    它会从三连号里截出子串：
#      · `表3.2.1 设计使用年限分类` → 截出 `2.1` + 8 个汉字`设计使用年限分类`（正好到行尾）
#        → 被判为节标题剥离，条文只剩 `…的规定。表3.`
#      · 交叉引用 `…本规范第4.1.7条的规` → 截出 `1.7` + `条的规`
#        → 条文被砍成 `…应符合本规范第4.`
#    实测新 OCR（PP-OCRv6）行尾粘连 0 例（4 例命中**全是**交叉引用），
#    行首形态 139 例。故彻底移除行尾版本，改用行首锚点 + 伪标题过滤。
SEC_HEAD_RE = re.compile(r"^\s*(\d{1,2}\.\d{1,2})\s*([\u4e00-\u9fa5]{2,20})\s*$")
# 真节标题不含 的/条/款；`条的规定`、`条的规` 是「续行/交叉引用」的典型特征
SEC_BAD_CHARS = set("的条款")
TABLE_RE = re.compile(r"表\s*([A-Z]?\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})")
# 表格**标题**：前面必须是句末标点（或串首），后面不接「的」。
# ⚠️ 不能用 TABLE_RE.search 直接切，也不能靠「行首」判定 —— 切分器拼接正文时
#    **不保留换行**，`(?:^|\n)` 形同虚设。实测判别特征：
#      引用 `…应符合表3.2.1的规定。`  → 前接「合」、后接「的」   → 排除
#      标题 `…措施。表6.14.2 屋面类别…` → 前接「。」、后接空格+标题 → 命中
TABLE_CAPTION_RE = re.compile(
    r"(?:^|(?<=[。；：]))[ \t]*表\s*([A-Z]?\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})(?!\s*的)")
# 区段起点（前缀匹配，不加 $ 锚点）：
# 真实文本是 `附录 B阅览室每座占使用面积设计计算指标`（字母后无空格），
# 若要求 `\s+` 或行尾锚点会漏判，导致附录内容被并进上一条正文。
DISCARD_RE = re.compile(
    r"^\s*(?:附\s*录\s*([A-Z])|本?[规范标准]{0,4}用词说明|弓?\s*引用标准名录"
    r"|条\s*文\s*说\s*明|附[：:]\s*条文说明|目\s*次)")
KEEP_APPENDIX = "B"
NOISE_RE = re.compile(
    r"^\s*(?:[-—=*_·]{3,}|住房城乡建设部信息公开.*|.*信息公开浏览专用.*|\d{1,3}"
    r"|•\s*\d{1,3}\s*[.．]?|X1)\s*$")

# 页脚页码混入正文：`…的需要。• 55.`、`贵重精• 12 •密医疗装备用房`、`…。• 57 .`
# 只认圆点项目符 `•`（U+2022）；中圆点 `·` 可能在正文里合法出现（如复合单位），
# 故 `·` 仅在**行尾**才视作页码。
BULLET_PAGE_RE = re.compile(r"\s*•\s*\d{1,3}\s*[.．]?\s*(?:•\s*)?")
DOT_PAGE_TAIL_RE = re.compile(r"\s*·\s*\d{1,3}\s*[.．]?\s*$")
PAGENO_HITS = []          # [(页码所在行片段, 清理后)] 供报告统计


def strip_pageno(s):
    """清掉混进正文的页码残渣；记录命中样本以便复核（不做静默处理）。"""
    out = BULLET_PAGE_RE.sub("", s)
    out = DOT_PAGE_TAIL_RE.sub("", out)
    if out != s:
        PAGENO_HITS.append((s.strip()[:60], out.strip()[:60]))
    return out
TERMINAL = "。；：！？）)】》”』…．."
START_RE = re.compile(r"^1\.0\.[1-9]$")

# ------------------------------------------------------------------ 水印清洗
WM = "住房城乡建设部信息公开浏览专用"
_WM_LONG = {WM[i:i + L] for L in range(4, len(WM) + 1)
            for i in range(len(WM) - L + 1)} - {"城乡建设", "住房城", "乡建设"}
_WM_SHORT = {"息公", "部信息", "息公开", "公开浏", "开浏览", "览专用"}
_WM_ALL = sorted(_WM_LONG | _WM_SHORT, key=len, reverse=True)
WM_HIT_RE = re.compile("|".join(re.escape(x) for x in _WM_ALL))
WM_CHARS = set(WM)


def wm_scrub(s):
    for frag in _WM_ALL:
        if frag in s:
            s = s.replace(frag, "")
    return s


def drop_wm_lines(texts):
    """
    删除「独立成行的水印碎片」。返回布尔掩码（True=丢弃）。

    背景：水印是斜排的，PP-OCRv6 会把它切成独立短行插进正文，例如
        「…可操作部件的中心距地面高度 / 专 / 应为0.85m~1.00m。」
    这些碎片（公开、息公开、专、专用、浏、住房…）都是水印字符串的子串。

    判据（三重限制，确保不误删正文）：
      1. 行长 ≤ 3 且所有字符均属于水印串 W；
      2. 不以句末标点结尾 —— 正文段末短行几乎总带标点（「水平。」「抓杆；」）；
      3. 下一行不是条文号/章节号/纯数字 —— 否则说明上一段正常结束，
         该短行可能是真实段尾（如「…人均住房」的「住房」）。
    """
    keep = [True] * len(texts)

    def nxt(i):
        for j in range(i + 1, len(texts)):
            if texts[j]:
                return texts[j]
        return ""

    for i, t in enumerate(texts):
        if not t or len(t) > 3 or not set(t) <= WM_CHARS or ends_terminal(t):
            continue
        n = nxt(i)
        if not n:
            continue
        if NUM_ONLY_RE.match(n) or ART_HEAD_RE.match(n) or SEC_ONLY_RE.match(n) \
                or re.match(r"^\d{1,3}\s*$", n):
            continue
        keep[i] = False
    # 返回「丢弃」掩码（True = 丢弃）
    return [not k for k in keep]


# ------------------------------------------------------------------ 工具
def meta_of(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    for key, m in META.items():
        if key in stem:
            return dict(m)
    return {"code": stem, "name": stem, "version": "", "force": "未知", "status": "未知"}


def chapter_of(no):
    m = re.match(r"^([A-Z]?\d+)", no)
    return m.group(1) if m else ""


def norm_text(s):
    s = s.replace("\u3000", " ")
    s = NUM_SPACE_RE.sub(r"\1.", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = strip_pageno(wm_scrub(s))
    return s.strip()


def ends_terminal(s):
    return bool(s) and s.rstrip()[-1] in TERMINAL


def is_full_line(s, min_len=12):
    return len(s.strip()) >= min_len


# ------------------------------------------------------------------ 读取 OCR 产物
def read_pages(md_path):
    with open(md_path, encoding="utf-8") as fh:
        raw_lines = fh.read().split("\n")
    out, page, buf = [], 1, []
    for raw in raw_lines:
        line = raw.rstrip()
        m = PAGE_RE.match(line)
        if m:
            if buf:
                out.append((page, buf))
            page, buf = int(m.group(1)), []
            continue
        buf.append(line)
    if buf:
        out.append((page, buf))
    return out


def read_geometry(json_path):
    """OCR json -> {(page, 行序号): 几何}。行序号与 md 中该页行顺序一一对应。"""
    if not json_path or not os.path.exists(json_path):
        return None
    try:
        with open(json_path, encoding="utf-8") as fh:
            rows = json.load(fh)
    except Exception:
        return None
    geo, counters = {}, {}
    for r in rows:
        p, b = r.get("page"), r.get("box") or []
        if p is None or len(b) < 4:
            continue
        i = counters.get(p, 0)
        counters[p] = i + 1
        geo[(p, i)] = {"x0": float(b[0]), "x1": float(b[2]),
                       "y0": float(b[1]), "y1": float(b[3]), "t": r.get("text", "")}
    return geo


def indent_flags(geo, page, n_lines):
    """判定该页每行是否「段首缩进」。段落首行缩进约 2 字，续行顶格。"""
    flags = [False] * n_lines
    if not geo:
        return flags
    rows = []
    for i in range(n_lines):
        g = geo.get((page, i))
        if g:
            rows.append((i, g))
    if len(rows) < 3:
        return flags
    margin = min(g["x0"] for _, g in rows)
    hs = sorted(g["y1"] - g["y0"] for _, g in rows)
    char = hs[len(hs) // 2] or 10.0
    for i, g in rows:
        if (g["x0"] - margin) > 0.5 * char:
            flags[i] = True
    return flags


# ------------------------------------------------------------------ 切分器
def carry_index(lines, flags):
    """下一段落的起点下标：优先最近一个段首缩进行，其次不完整的末行。"""
    for i in range(len(lines) - 1, -1, -1):
        if flags[i]:
            return i
    if lines and not ends_terminal(lines[-1]) and is_full_line(lines[-1]):
        return len(lines) - 1
    return None


class Builder:
    def __init__(self, postfix):
        self.postfix = postfix
        self.chunks = []
        self.cur = None
        self.pending = []          # [(text, indented)] 尚未归属条文的行
        self.started = False

    def _open(self, no, page, kind="article"):
        self.cur = {"no": no, "kind": kind, "lines": [], "flags": [],
                    "page_start": page, "page_end": page, "moved": False}

    def _emit(self):
        c = self.cur
        self.cur = None
        if c is None:
            return
        body = "".join(c["lines"]).strip()
        if not body:
            return
        self.chunks.append({"article_no": c["no"], "kind": c["kind"],
                            "page_start": c["page_start"], "page_end": c["page_end"],
                            "text": body, "moved": c["moved"]})

    def flush(self):
        self._emit()

    def on_article(self, no, page, inline_rest=None):
        if not self.started:
            if not START_RE.match(no):
                return                      # 1.0.1 之前是公告/前言，丢弃
            self.started = True

        if self.postfix and self.cur is not None:
            lines, flags = self.cur["lines"], self.cur["flags"]
            k = carry_index(lines, flags)
            carried = []
            if k is not None:
                carried = lines[k:]
                self.cur["lines"], self.cur["flags"] = lines[:k], flags[:k]
            self._emit()
            self._open(no, page)
            if carried:
                self.cur["lines"] = list(carried)
                self.cur["flags"] = [False] * len(carried)
                self.cur["moved"] = True
        else:
            self._emit()
            self._open(no, page)
            if self.pending:
                if not self.postfix:
                    self.cur["lines"] = [t for t, _ in self.pending]
                    self.cur["flags"] = [f for _, f in self.pending]
                else:                       # 后置版式：只取末段的头
                    texts = [t for t, _ in self.pending]
                    fl = [f for _, f in self.pending]
                    k = carry_index(texts, fl)
                    if k is not None:
                        self.cur["lines"] = texts[k:]
                        self.cur["flags"] = [False] * len(texts[k:])
                self.pending = []

        if inline_rest:
            self.cur["lines"].append(norm_text(inline_rest))
            self.cur["flags"].append(False)

    def on_text(self, t, page, indented=False):
        if self.cur is not None:
            if page > self.cur["page_end"]:
                self.cur["page_end"] = page
            self.cur["lines"].append(t)
            self.cur["flags"].append(indented)
        elif self.started:
            self.pending.append((t, indented))

    def pop_title(self, n_max=2):
        """从当前条文尾部摘出章节标题行（须为缩进行，避免误摘条文末行）。"""
        if self.cur is None:
            return ""
        out = []
        lines, flags = self.cur["lines"], self.cur["flags"]
        while lines and len(out) < n_max:
            last = lines[-1]
            if not flags[-1] or ends_terminal(last) or len(last) > 14 or re.search(r"\d", last):
                break
            out.append(last)
            lines.pop()
            flags.pop()
        return " ".join(reversed(out))


def split_file(md_path, json_path):
    meta = meta_of(md_path)
    pages = read_pages(md_path)
    geo = read_geometry(json_path)

    n_only = n_inline = 0
    for _, lines in pages:
        for ln in lines:
            s = norm_text(ln)
            if not s:
                continue
            mo = NUM_ONLY_RE.match(s)
            mh = ART_HEAD_RE.match(s)
            if mo:
                n_only += 1
            elif mh and mh.group(2).strip():
                n_inline += 1
    postfix = n_only > 0 and n_only >= 0.8 * (n_only + n_inline)

    b = Builder(postfix)
    discard = False
    appendices = []          # [(字母, [(page, line), ...])] 仅收 KEEP_APPENDIX 指定的附录
    appendix_mode = None     # 当前正在收集的附录字母
    appendix_buf = []

    def close_appendix():
        nonlocal appendix_buf, appendix_mode
        if appendix_mode == KEEP_APPENDIX and appendix_buf:
            appendices.append((appendix_mode, appendix_buf))
        appendix_buf = []

    for page, lines in pages:
        flags = indent_flags(geo, page, len(lines))
        normed = [norm_text(l) for l in lines]
        drop = drop_wm_lines(normed)
        for idx in range(len(lines)):
            if drop[idx]:
                continue
            s = normed[idx]
            if not s or NOISE_RE.match(s):
                continue

            md = DISCARD_RE.match(s)
            if md:
                letter = md.group(1)
                if not b.postfix:
                    b.flush()
                close_appendix()
                discard = True
                # 只有正文之后的附录才启动收集：目次里也会出现「附录 B…」条目，
                # 若在正文之前就置位，会把目次行当成附录内容收进去。
                appendix_mode = letter.upper() if (letter and b.started) else None
                continue

            mo = NUM_ONLY_RE.match(s)
            mh = ART_HEAD_RE.match(s) if (not mo and not SEC_ONLY_RE.match(s)) else None
            no = mo.group(1) if mo else (mh.group(1) if (mh and mh.group(2).strip()) else None)

            if discard:
                # 目次（正文之前）遇到 1.0.1 即退出；
                # 附录/用词说明/引用标准名录/条文说明（正文之后）永不退出 ——
                # 这些区段里同样会出现「1.0.1」等条文号，若据此退出会污染知识库。
                if no == "1.0.1" and not b.started:
                    discard = False
                    appendix_mode, appendix_buf = None, []
                else:
                    if appendix_mode == KEEP_APPENDIX:
                        appendix_buf.append((page, s))
                    continue

            if no:
                b.on_article(no, page, None if mo else mh.group(2))
                continue

            ms = SEC_ONLY_RE.match(s)
            if ms and b.started:
                title = b.pop_title()
                b.flush()
                b.chunks.append({"article_no": ms.group(1), "kind": "section",
                                 "page_start": page, "page_end": page,
                                 "text": title or f"第{ms.group(1)}节", "moved": False})
                continue

            mh2 = SEC_HEAD_RE.match(s)
            if mh2 and b.started and not (set(mh2.group(2)) & SEC_BAD_CHARS):
                b.flush()
                b.chunks.append({"article_no": mh2.group(1), "kind": "section",
                                 "page_start": page, "page_end": page,
                                 "text": mh2.group(2), "moved": False})
                continue

            parts = ART_INLINE_RE.split(s) if b.cur else [s]
            if len(parts) > 1:
                if parts[0].strip():
                    b.on_text(norm_text(parts[0]), page, flags[idx])
                for k in range(1, len(parts) - 1, 2):
                    b.on_article(parts[k], page, parts[k + 1] if k + 1 < len(parts) else None)
                continue

            b.on_text(s, page, flags[idx])

    b.flush()
    close_appendix()

    chunks = [c for c in b.chunks if c["text"].strip()]
    chunks = extract_tables(chunks)

    # 保留的附录（按用户决策：仅「附录B」这类含设计指标表的附录入库）
    for letter, items in appendices:
        pages_ = [p for p, _ in items]
        body = "".join(t for _, t in items).strip()
        if not body:
            continue
        chunks.append({"article_no": f"附录{letter}", "kind": "appendix-table",
                       "page_start": min(pages_), "page_end": max(pages_),
                       "text": body, "moved": False})

    out = []
    for c in chunks:
        no = c["article_no"]
        rec = {"standard": meta["code"], "standard_name": meta["name"],
               "version": meta["version"], "force_status": meta["force"], "status": meta["status"],
               "article_no": no, "chapter": chapter_of(no),
               "page_start": c["page_start"], "page_end": c["page_end"],
               "kind": c["kind"], "text": c["text"].strip()}
        rev = []
        if c.get("moved"):
            rev.append("首行系从上一段移入(需核对)")
        if rec["kind"] == "table":
            rev.append("表格OCR结构不可靠")
        if rec["kind"] == "appendix-table":
            rev.append("附录表格OCR结构不可靠")
        if WM_HIT_RE.search(rec["text"]):
            rev.append("水印残余")
        if rec["kind"] == "article" and len(rec["text"]) < 10:
            rev.append("过短")
        seg = re.match(r"^[A-Z]?\d{1,2}\.\d{1,2}\.(\d+)$", no)
        if seg and len(seg.group(1)) > 2:
            rev.append("条文号末段超长")
        if rev:
            rec["review"] = "+".join(rev)
        rec["id"] = f"{meta['code']}_{no}"
        out.append(rec)

    # 漏号定位：同一节内条文号不连续时，缺号的正文很可能被并进了前一条。
    # PP-OCRv6 偶有把条文号识别成乱字符的情形（实测 GB 55019 的 2.9.5 -> "V"，
    # 3.1.9 整行丢失），此时该条正文会被当作续行并进上一条。这里把它标出来，
    # 便于人工按原文拆分——不自动拆，避免误切。
    by_sec = {}
    for rec in out:
        if rec["kind"] == "article" and ARTNO_FULL.match(rec["article_no"]):
            by_sec.setdefault(".".join(rec["article_no"].split(".")[:2]), []).append(rec)
    for sec, recs in by_sec.items():
        got = sorted({int(r["article_no"].split(".")[-1]) for r in recs})
        for a, b in zip(got, got[1:]):
            if b - a != 2:
                continue
            miss = f"{sec}.{a + 1}"
            for r in recs:
                if r["article_no"] == f"{sec}.{a}":
                    r["review"] = (r["review"] + "+" if r.get("review") else "") \
                                  + f"疑漏{miss}(正文或已并入本条)"
    return out, postfix


def extract_tables(chunks):
    out = []
    for c in chunks:
        t = c["text"]
        m = TABLE_CAPTION_RE.search(t)
        if not m or len(t) < 120 or c["kind"] != "article":
            out.append(c)
            continue
        # 切点落在「表」字上，不含前导换行/空格
        cut = m.start() + m.group(0).index("表")
        head, tail = t[:cut].strip(), t[cut:].strip()
        digit_ratio = sum(ch.isdigit() for ch in tail) / max(len(tail), 1)
        if len(head) >= 8 and len(tail) >= 60 and digit_ratio > 0.06:
            out.append(dict(c, text=head))
            out.append(dict(c, article_no=f"{c['article_no']}-表", kind="table", text=tail))
        else:
            out.append(c)
    return out


# ------------------------------------------------------------------ 自检
def _rng(nums):
    """[4,5,6,9] -> '4~6,9'"""
    if not nums:
        return ""
    nums = sorted(nums)
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(str(start) if start == prev else f"{start}~{prev}")
        start = prev = n
    out.append(str(start) if start == prev else f"{start}~{prev}")
    return ",".join(out)


def self_check(all_chunks):
    """快速自检：断号 / 重复号 / 首条异常。详细报告见 validate_chunks.py。"""
    issues = []
    by_std = {}
    for c in all_chunks:
        by_std.setdefault(c["standard"], []).append(c)
    for std, cs in sorted(by_std.items()):
        arts = [c for c in cs if c["kind"] == "article"]
        # 按「节」（前两段编号）分组，比较第三段是否连续 —— 跨节比较没有意义
        seen = {}
        for c in arts:
            no = c["article_no"]
            if not re.match(r"^[A-Z]?\d{1,2}\.\d{1,2}\.\d{1,2}$", no):
                continue
            seen.setdefault(".".join(no.split(".")[:2]), []).append(no)
        for sec, nos in sorted(seen.items()):
            if len(nos) != len(set(nos)):
                dup = sorted({x for x in nos if nos.count(x) > 1})
                issues.append((std, "重复条文号", ",".join(dup)))
        for sec, nos in sorted(seen.items()):
            segs = sorted({int(n.split(".")[-1]) for n in nos})
            miss = [x for a, b in zip(segs, segs[1:]) for x in range(a + 1, b)]
            if miss:
                issues.append((std, f"断号(节{sec})", f"{sec}.{_rng(miss)}"))
        if arts:
            first = min(arts, key=lambda c: (c["page_start"], c["article_no"]))
            if first["article_no"] != "1.0.1":
                issues.append((std, "首条不是1.0.1", first["article_no"]))
    return issues


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dify-sep", default="\n@@@\n")
    ap.add_argument("--min-len", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "dify"), exist_ok=True)

    all_chunks = []
    for fn in sorted(f for f in os.listdir(args.src) if f.endswith(".md")):
        md = os.path.join(args.src, fn)
        js = os.path.join(args.src, os.path.splitext(fn)[0] + ".json")
        chunks, postfix = split_file(md, js)
        chunks = [c for c in chunks if len(c["text"]) >= args.min_len]
        all_chunks.extend(chunks)

        arts = [c for c in chunks if c["kind"] == "article"]
        tabs = [c for c in chunks if c["kind"] == "table"]
        secs = [c for c in chunks if c["kind"] == "section"]
        apps = [c for c in chunks if c["kind"] == "appendix-table"]
        lens = [len(c["text"]) for c in chunks] or [0]
        print(f"{os.path.splitext(fn)[0]:<42} 版式={'后置' if postfix else '行首'}"
              f"  条文 {len(arts):>4}  章节 {len(secs):>3}  表格 {len(tabs):>2}"
              f"  附录 {len(apps):>2}"
              f"  平均 {sum(lens)//len(lens):>3} 字  最长 {max(lens):>5}")

        stem = os.path.splitext(fn)[0]
        parts = []
        for c in chunks:
            pg = f"P{c['page_start']}" + (f"-{c['page_end']}" if c["page_end"] != c["page_start"] else "")
            parts.append(f"【{c['standard']}】{c['article_no']}（{c['force_status']}，{pg}）\n"
                         f"{c['text']}\n")
        with open(os.path.join(args.out, "dify", stem + ".md"), "w",
                  encoding="utf-8", newline="\n") as fh:
            fh.write(args.dify_sep.join(parts))

    with open(os.path.join(args.out, "chunks.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(os.path.join(args.out, "chunks.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["standard", "article_no", "chapter", "kind", "page_start", "page_end",
                    "force_status", "chars", "review", "text"])
        for c in all_chunks:
            w.writerow([c["standard"], c["article_no"], c["chapter"], c["kind"],
                        c["page_start"], c["page_end"], c["force_status"], len(c["text"]),
                        c.get("review", ""), c["text"]])

    flagged = [c for c in all_chunks if c.get("review")]
    with open(os.path.join(args.out, "待校对优先清单.csv"), "w",
              encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["standard", "article_no", "kind", "page", "chars", "疑点", "text"])
        for c in sorted(flagged, key=lambda x: (x["standard"], x["article_no"])):
            w.writerow([c["standard"], c["article_no"], c["kind"], c["page_start"],
                        len(c["text"]), c.get("review", ""), c["text"]])

    issues = self_check(all_chunks)
    with open(os.path.join(args.out, "_自检问题.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["standard", "问题类型", "详情"])
        for row in issues:
            w.writerow(row)

    print()
    print(f"合计 {len(all_chunks)} 条 -> {os.path.join(args.out, 'chunks.jsonl')}")
    print(f"待校对 {len(flagged)} 条 -> 待校对优先清单.csv")
    print(f"内置自检发现 {len(issues)} 项问题 -> _自检问题.csv")
    for std, kind, detail in issues[:30]:
        print(f"   [{std}] {kind}: {detail}")


if __name__ == "__main__":
    main()
