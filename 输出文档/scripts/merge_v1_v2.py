# -*- coding: utf-8 -*-
"""merge_v1_v2.py —— 用旧 RapidOCR(v1) 补齐 PP-OCRv6(v2) 丢失的内容。

## 为什么需要
两版 OCR 的失效模式是**互补**的：
  · v1（RapidOCR，旧）：文字齐全，但有斜排水印碎片、错别字（`生活二作`、`Z.0m`）、
    且把附录/条文说明误挂到末条。
  · v2（PP-OCRv6，新）：干净得多（水印残余 0、无假节标题），但**会丢整行** ——
    实测 GB 50352 的 2.0.1 / 2.0.21 / 2.0.35 定义行、5.5.2 的「应采用城市统一的
    坐标系统和高程系统」、以及 17 个条文号（如 5.2.2 / 2.9.5 / 2.1.2）**全部只在 v1 里有**。

## 算法
以 v2 的行流为骨架；逐行在 v1 中定位对应位置（用 10 字探针在 v1 的「行拼接串」里
查找并换算回行号）；命中后向前扫描 v1 的后续行，把**判定为 v2 缺失**的行插入其后。
「缺失」的判定：该行的任意 10-gram 都不出现在 v2 的行拼接串里。
进入 附录/条文说明 等丢弃区段后停止插入（那些区段本就不入库）。

## 产出
  ocr_v3/<规范>.md   —— 合并后的分页 markdown，可直接喂 split_articles_v2.py
  ocr_v3/_merge_report.txt
"""
import bisect
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE = "D:/WorkBuddy/项目/输出文档"
OLD = os.path.join(BASE, "_backup_v1_20260910/ocr_text")
NEW = os.path.join(BASE, "ocr_v2")
DST = os.path.join(BASE, "ocr_v3")

WM_FRAG = re.compile(
    r"住房城乡|住房城|住房|城乡建设部|建设部|信息公开|息公开|公开浏览|开浏览"
    r"|浏览专用|览专用|息公|部信息")
PAGE_RE = re.compile(r"^<!--\s*page\s+(\d+)\s*-->\s*$")
DISCARD_RE = re.compile(
    r"^\s*(?:附\s*录\s*[A-Z]|本?[规范标准]{0,4}用词说明|弓?\s*引用标准名录"
    r"|条\s*文\s*说\s*明|附[：:]\s*条文说明|目\s*次)")
PUNCT = re.compile(r"[\s。，、；：？！“”‘’（）()《》〈〉\-—·•.,;:!?\"'/\\|]+")
NOISE_LINE = re.compile(r"^[\s\d。，、；：·•.\-—_=*]+$")
# 目录行（含英文目录）：`6. 6厕所、卫生间、盟洗室、浴室和母婴室25`
# v1 的正文行可能匹配到 v1 目录里的同名条目，若不排除会把整段目录搬进正文。
TOC_RE = re.compile(r"^\s*\d{1,2}\.\s*\d{1,2}\s*[^\d]{2,40}\d{1,3}\s*$")
# 正文起点：1.0.1（其前是公告 / 前言 / 目录 / 编写人名单，一律不用）
BODY_START = re.compile(r"^\s*1\.\s*0\.\s*1(?![.\d])")

NGRAM = 10
MIN_KEY = 4


def key(s):
    """归一化用于匹配：去水印碎片 → 去标点空白。"""
    return PUNCT.sub("", WM_FRAG.sub("", s))


def is_noise(line):
    if TOC_RE.match(line):
        return True
    k = key(line)
    return len(k) < MIN_KEY or NOISE_LINE.match(line)


def parse_pages(text):
    """-> [('P', n) | ('L', text)]，跳过空行。"""
    seq = []
    for raw in text.split("\n"):
        m = PAGE_RE.match(raw)
        if m:
            seq.append(("P", int(m.group(1))))
            continue
        if raw.strip():
            seq.append(("L", raw.strip()))
    return seq


def build_index(v1_lines):
    off, blob = [], ""
    for l in v1_lines:
        off.append(len(blob))
        blob += key(l)
    off.append(len(blob))
    return blob, off


def locate(v2key, v1_blob, v1_off, hint):
    """在 v1 串里找 v2 行对应的 v1 行号。

    单点探针会因 OCR 错别字而失配（v1 有 `生活二作`、`Z.0m` 这类错误），
    故沿线取 4 个候选探针（0/¼/½/¾ 处），命中即用。
    """
    n = len(v2key)
    if n < NGRAM:
        return None
    for frac in (0.5, 0.25, 0.75, 0.0):
        p = int((n - NGRAM) * frac)
        probe = v2key[p:p + NGRAM]
        pos = v1_blob.find(probe, hint)
        if pos < 0:
            pos = v1_blob.find(probe)
        if pos >= 0:
            return bisect.bisect_right(v1_off, pos) - 1
    return None


def present(v1key, v2_blob):
    """该 v1 行的内容是否已在 v2 中出现（任一 10-gram 命中）。"""
    n = len(v1key)
    if n < NGRAM:
        return v1key in v2_blob
    for i in range(0, n - NGRAM + 1, 2):
        if v1key[i:i + NGRAM] in v2_blob:
            return True
    return v1key[-NGRAM:] in v2_blob


def mostly_present(k, blob, ratio=0.3, n=4):
    """模糊存在判定：4-gram 命中率 ≥ ratio 即认为 v2 已含该内容。

    用途：v1 的表格碎片与 v2 的对应行**字面不同**（`建筑高度大于 1100m` vs
    `建筑高度大于100m`），严格 10-gram 判不出「已有」，会把大量碎片灌进正文。
    """
    if len(k) < n:
        return k in blob
    grams = [k[i:i + n] for i in range(len(k) - n + 1)]
    hit = sum(1 for g in grams if g in blob)
    return hit / len(grams) >= ratio


# 条文号开头的行（缺失条文，价值最高，绕过模糊过滤）
NUMHEAD_RE = re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{1,2}(?![.\d\-–—~～条款])")
# 节标题开头的行：**不插入**。非必需（章号可由条文号推出），而 v1 对节标题的
# 误认率很高（`7.1 3光环境`、`8.1丝给水排水`、`3.13建筑面积`、`6.8 #楼梯`），
# 插进来会凭空造出假节标题。
SECHEAD_RE = re.compile(r"^\s*\d{1,2}\.\d{1,2}(?![.\d])")


def artno_of(line):
    """取行首条文号（归一化空格），无则返回空串。"""
    m = re.match(r"^\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})(?![.\d\-–—~～条款])", line)
    return f"{int(m.group(1))}.{int(m.group(2))}.{int(m.group(3))}" if m else ""


def secnum_of(line):
    """取行首节号 `X.Y`（或条文号的 X.Y 前缀）。"""
    m = re.match(r"^\s*(\d{1,2})\s*\.\s*(\d{1,2})", line)
    return f"{int(m.group(1))}.{int(m.group(2))}" if m else ""


def can_insert(line, prev_raw):
    """是否允许把这一行 v1 内容补进 v2。

    只收两类，其余（v1 的表格碎片、错字变体）一律丢弃：
      1) 以条文号/节号开头的行 —— 修复 v2 丢掉的整条内容；
      2) 高置信续行 —— 上一行（v2 已收）在句中截断，且本行是像样的中文。
    """
    k = key(line)
    if len(k) < MIN_KEY:
        return False
    if NUMHEAD_RE.match(line):
        return True
    # 续行：上一行在句中截断（末字符不是句末标点）
    p = prev_raw.rstrip()
    if not p or p[-1] in "。；：）%”！?":
        return False
    if len(k) < 12:
        return False
    cjk = sum(1 for c in k if "\u4e00" <= c <= "\u9fa5")
    return cjk >= 8 and cjk / len(k) >= 0.7


def merge(std, fname, log):
    v2_seq = parse_pages(open(os.path.join(NEW, fname), encoding="utf-8").read())
    v1_lines = [t for k, t in parse_pages(open(os.path.join(OLD, fname), encoding="utf-8").read())
                if k == "L"]
    # 裁掉 v1 的公告/前言/目录/名单：从第一条 1.0.1 起才作为候选来源。
    # 否则正文行会匹配到目录里的同名条目，把整段目录搬进正文（实测发生）。
    for i, l in enumerate(v1_lines):
        if BODY_START.match(l):
            v1_lines = v1_lines[i:]
            break
    v1_blob, v1_off = build_index(v1_lines)
    v2_blob = "".join(key(t) for k, t in v2_seq if k == "L")
    # v2 中作为「行首条文号」出现过的编号集合。
    # ⚠️ 判断某条文是否已被 v2 收录，**必须按编号判断**，不能只靠文本匹配：
    #    v1 的条文头常有错字（`2.5.6全玻璃闪…` vs v2 `2.5.6全玻璃门…`），
    #    短行里一个错字就会让所有 10-gram 失配 → 误当成「缺失」而重复插入。
    v2_heads = {artno_of(t) for k, t in v2_seq if k == "L" and artno_of(t)}

    out, hint, stopped = [], 0, False
    inserted, used = [], set()      # used: 已插入的 v1 行号，防止重复插入
    got = set()                     # got: 已插入的条文号
    defer_bare = []                 # [(编号行, 其正文首行的 key)] 待前移
    lines_v2 = [(k, v) for k, v in v2_seq]

    for i, (kind, val) in enumerate(lines_v2):
        if kind == "P":
            out.append((kind, val))
            continue

        out.append((kind, val))
        if stopped:
            continue
        if DISCARD_RE.match(val):
            stopped = True          # 进入 附录/条文说明 等区段，不再插 v1 内容
            continue

        j = locate(key(val), v1_blob, v1_off, hint)
        if j is None:
            continue
        hint = v1_off[j]

        # 立即向后排空：v1 中紧随其后、而 v2 已丢失的行
        prev_raw = val
        nxt = lines_v2[i + 1] if i + 1 < len(lines_v2) else None
        next_k = key(nxt[1]) if (nxt and nxt[0] == "L") else ""
        k, cnt, scanned = j + 1, 0, 0
        while k < len(v1_lines) and cnt < 30 and scanned < 15:
            scanned += 1
            if is_noise(v1_lines[k]) or k in used:
                k += 1
                continue
            kk = key(v1_lines[k])
            if present(kk, v2_blob):
                # ⚠️ 这里**不能 break**：锚点行之后往往还有若干 v2 已有的续行，
                #    缺失的条文（如 5.2.5）就在它们再往后几行；一旦 break 就够不到。
                k += 1
                continue
            line = v1_lines[k]
            if NUMHEAD_RE.match(line):
                # 条文号开头的行价值最高，**绕过**模糊过滤（`5.2.2基地道路设计
                # 应符合下列规定` 与 v2 的 5.2.1 标题共享多个 4-gram）。
                # 但必须先按**编号**判重，避免 v1 的错字头重复插入。
                no = artno_of(line)
                # 孤立编号（行内只有号码、无正文）：若紧随其后的一行 v2 已有，
                # 说明这是「编号后置」——正文其实排在编号**之前**，已被上一条收走。
                # 不能就地插入（会造出空条文），改为登记待**前移**到其正文之前。
                if no and key(line) == no.replace(".", ""):
                    nxt_l = v1_lines[k + 1] if k + 1 < len(v1_lines) else ""
                    if nxt_l and present(key(nxt_l), v2_blob):
                        defer_bare.append((line, key(nxt_l)))
                    k += 1
                    continue
                if len(kk) >= MIN_KEY and no and no not in v2_heads and no not in got:
                    out.append(("L", line))
                    inserted.append(line)
                    got.add(no)
                    used.add(k)
                    prev_raw = line
                    cnt += 1
                k += 1
                continue
            if SECHEAD_RE.match(line):
                k += 1
                continue        # 节标题不入（v1 误认率高，且非必需）
            # 模糊判为「v2 已有」→ v1 的错字/表格碎片变体，跳过
            if mostly_present(kk, v2_blob):
                k += 1
                continue
            # 行分段差异（v1 把一行切成两半）不算丢失
            if kk in key(prev_raw) or kk in next_k or key(prev_raw) in kk:
                k += 1
                continue
            if not can_insert(line, prev_raw):
                k += 1
                continue
            out.append(("L", line))
            inserted.append(line)
            used.add(k)
            prev_raw = line
            k += 1
            cnt += 1

    # ---- 「编号后置」修复：把孤立编号移到它自己正文的前面 ----
    # 实测 GB 50352 的 6.17.2：v2 丢了编号，而正文（`1 室内装修不得遮挡…`）
    # 被排在了编号本该在的位置——结果正文被 6.17.1 吸收。把编号前移才能恢复。
    # 必须**全局扫描**：该编号在 v1 中的位置可能远在锚点窗口之外，主循环够不到。
    for k, line in enumerate(v1_lines):
        if k in used:
            continue
        no = artno_of(line)
        if not no or key(line) != no.replace(".", ""):
            continue
        nxt_l = v1_lines[k + 1] if k + 1 < len(v1_lines) else ""
        if not nxt_l or not present(key(nxt_l), v2_blob):
            continue                    # 正文也不在 v2 → 交给常规补齐路径
        if any(artno_of(t) == no for kk2, t in out if kk2 == "L"):
            continue                    # 该编号已存在，无需处理
        probe = key(nxt_l)[:12]
        for i, (kind, val) in enumerate(out):
            if kind == "L" and probe in key(val):
                out.insert(i, ("L", line))
                inserted.append(line)
                used.add(k)
                got.add(no)
                break

    # ---- 定点补齐：主循环的窗口（锚点后 15 行）够不到的漏条 ----
    # 按条文序定位：缺号 N 一定排在同节的下一个现存条文之前
    # （实测 5.3.4→5.3.5、6.17.2→6.17.3、4.3.5→4.3.6）。
    # ⚠️ 不能加 `if not stopped` 门控：`stopped` 到文末的「用词说明」才置位，
    #    而主循环只处理到那里；加了门控会让整个补齐被跳过（实测漏掉 5.3.4/6.17.2）。
    #    补齐只插在「同节后继条文头」之前，位置上必落在正文区内，是安全的。
    if True:
        for k, line in enumerate(v1_lines):
            if k in used or not NUMHEAD_RE.match(line):
                continue
            no = artno_of(line)
            if not no or no in got:
                continue
            heads = {artno_of(t): i for i, (kk2, t) in enumerate(out) if kk2 == "L" and artno_of(t)}
            if no in heads:
                continue
            sec, tail = no.rsplit(".", 1)
            tail = int(tail)
            later = [(int(h.rsplit(".", 1)[1]), i) for h, i in heads.items()
                     if h.startswith(sec + ".") and h.rsplit(".", 1)[1].isdigit()
                     and int(h.rsplit(".", 1)[1]) > tail]
            if not later:
                continue                    # 同节无后继条文 → 位置不确定，放弃（留待人工）
            pos = min(later)[1]
            block = [line]
            j = k + 1
            while j < len(v1_lines) and len(block) < 10:
                l2 = v1_lines[j]
                if NUMHEAD_RE.match(l2) or SECHEAD_RE.match(l2) or is_noise(l2):
                    break
                blob_now = "".join(key(t) for kk2, t in out if kk2 == "L")
                if present(key(l2), blob_now):
                    break
                block.append(l2)
                j += 1
            out[pos:pos] = [("L", b) for b in block]
            inserted.extend(block)
            used.update(range(k, j))
            got.add(no)

    # 清掉「孤立编号」插入：号码后面接的是已有内容，说明只是编号错位，
    # 插一个空条文没有意义（切分时也会因正文为空而被丢弃）。
    cleaned = []
    for i, (kind, val) in enumerate(out):
        if kind == "L":
            no = artno_of(val)
            if no and key(val) == no.replace(".", ""):
                nxt = out[i + 1] if i + 1 < len(out) else None
                if nxt is None or (nxt[0] == "L" and artno_of(nxt[1])):
                    continue            # 空条文 → 丢掉
                if nxt and nxt[0] in ("P",):
                    continue
        cleaned.append((kind, val))
    out = cleaned

    # 重组（保留 v2 的分页标记）
    body = []
    for kind, val in out:
        body.append(f"<!-- page {val} -->" if kind == "P" else val)
    text = "\n".join(body) + "\n"
    os.makedirs(DST, exist_ok=True)
    with open(os.path.join(DST, fname), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)

    log.append(f"{fname}")
    log.append(f"  v1 行 {len(v1_lines)}  v2 行 {sum(1 for k,_ in v2_seq if k=='L')}"
               f"  合并后 {sum(1 for k,_ in out if k=='L')}  插入 {len(inserted)} 行")
    for t in inserted[:40]:
        log.append(f"    + {t[:90]}")
    if len(inserted) > 40:
        log.append(f"    ... 另有 {len(inserted)-40} 行")
    log.append("")


def main():
    log = []
    files = sorted(f for f in os.listdir(NEW) if f.endswith(".md"))
    for fn in files:
        if os.path.exists(os.path.join(OLD, fn)):
            merge(fn[:11], fn, log)
        else:
            log.append(f"{fn}: 旧版无对应文件，跳过\n")
    os.makedirs(DST, exist_ok=True)
    report = "\n".join(log)
    with open(os.path.join(DST, "_merge_report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report)
    print(report)


if __name__ == "__main__":
    main()
