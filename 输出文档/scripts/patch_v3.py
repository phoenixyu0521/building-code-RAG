# -*- coding: utf-8 -*-
"""
patch_v3.py —— 用旧版 RapidOCR(v1) 的「行级并集」回填 ocr_v3 丢失的行，产出 ocr_v4。

背景：ocr_v3 = merge_v1_v2.py 的产物，但该合并只插入了「行首是条文号」的行，
导致 v2(PP-OCRv6) 丢失的下列内容没有回填：

  * 行尾续行（如 GB 55019 `2.7.2 ...三级及三级以上的` + `台阶和楼梯应在两侧设置扶手`）
  * 术语定义行（GB 50352 `2.0.1 民用建筑 civil building` 后面的定义）
  * 枚举项编号行（GB 55019 `1 应安装牢固；` 等）
  * 表格行（GB 55037 火灾危险性分类表、JGJ 38 附录 A 等）

做法：对 v1 正文范围内每个「内容行」算 4-gram 在 v3 中的覆盖率；
缺失者以「最近的前置已存在行」为锚点，按 v1 顺序串行回填到 v3 的对应行之后。
锚点定位用 4-gram 最佳匹配 + 单调递增约束（v3 正文顺序 ≈ v1 顺序）。

用法：
  python patch_v3.py --v1 <v1目录> --v3 <ocr_v3目录> --out <ocr_v4目录> [--report X.txt]
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

WM_FRAGS = ["住房城乡建设部信息公开浏览专用", "息公", "部信息", "公信息",
            "信息公开", "浏览专用", "住房城乡建设部", "城乡建设",
            "住房和城乡建设部", "浏览专", "住房城", "住房和", "息浏览"]


def clean(s):
    for f in WM_FRAGS:
        s = s.replace(f, "")
    return s


_NORM = [
    (r"[～~﹏]", "~"),
    (r"[×xX＊*✕]", "×"),
    (r"[入人]", "人"),
    (r"[－—–−一]", "-"),
    (r"[≤≦]", "≤"),
    (r"[≥≧]", "≥"),
    (r"[［【\[]", "["),
    (r"[］】\]]", "]"),
]
_STRIP = r"[\s\u3000，。、；：（）()《》“”\"'·\\\-…\.．;:<>|/]+"


def key(s):
    for p, r in _NORM:
        s = re.sub(p, r, s)
    return re.sub(_STRIP, "", s)


ARTNO = re.compile(r"^\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*(.*)$")


def grams(s, n=4):
    return set(s[i:i + n] for i in range(len(s) - n + 1)) if len(s) >= n else set()


def body_bounds(L):
    st = next((i for i, l in enumerate(L)
               if re.search(r"(?<![\d.])1\s*[.．]\s*0\s*[.．]\s*1(?![\d])", l)), 0)
    en = next((i for i, l in enumerate(L)
               if i > st and re.match(r"(本规范用词说明|引用标准名录|条文说明|本规范用词)", l)), len(L))
    ap_ = next((i for i, l in enumerate(L) if i > st and re.match(r"^附\s*[录彔]", l)), None)
    if ap_ is not None and ap_ < en:
        en = ap_                                   # 附录（表格区）不参与回填，避免制造表格噪声
    return st, en


BAD = re.compile(r"^[\s\d\.．\-—~～%％mMcCiI×xX/*、,，]+$")
STRUCT = re.compile(r"^(<!--|#|>|\d{1,3}$)")
TOCLINE = re.compile(r"^[A-Za-z]")                    # 纯英文目录行
LONELY = re.compile(r"^[\u4e00-\u9fa5]$|^[A-Za-z]{1,2}$|^\d{1,2}$")


def cjk_ratio(s):
    return len(re.findall(r"[\u4e00-\u9fa5]", s)) / max(1, len(re.sub(r"\s", "", s)))


def numeric_tokens(s):
    return len(re.findall(r"\d+(?:\.\d+)*", s))


def insertable(raw):
    """回填前的质量门：中文占比过低（表格数值行）或数值 token 过多（表格行）则丢弃。"""
    if cjk_ratio(raw) < 0.45:
        return False
    if numeric_tokens(raw) > 3:
        return False
    return True


def is_content(k, raw):
    if len(k) < 8:
        return False
    if BAD.match(raw) or STRUCT.match(raw):
        return False
    if not re.search(r"[\u4e00-\u9fa5]", raw):       # 纯英文/纯符号：目录行或表格单位行
        return False
    if LONELY.match(raw.strip()):                    # 孤字水印残留
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="../_backup_v1_20260910/ocr_text")
    ap.add_argument("--v3", default="../ocr_v3")
    ap.add_argument("--out", default="../ocr_v4")
    ap.add_argument("--report", default=None)
    ap.add_argument("--thr-in", type=float, default=0.72, help="全文档 gram 覆盖率高于此值判为已存在")
    ap.add_argument("--thr-dup", type=float, default=0.72, help="与某个 v3 单行相似度高于此值判为近似重复")
    ap.add_argument("--thr-anchor", type=float, default=0.72, help="锚点行匹配阈值")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rep = []

    for name in sorted(os.listdir(a.v3)):
        if not name.endswith(".md"):
            continue
        p1 = os.path.join(a.v1, name)
        if not os.path.exists(p1):
            print(f"!! 无 v1 对应文件：{name}")
            continue
        L1 = [clean(l).strip() for l in open(p1, encoding="utf-8").read().splitlines()]
        L3 = open(os.path.join(a.v3, name), encoding="utf-8").read().splitlines()

        G3 = grams(key("\n".join(L3)), 4)
        K3ALL = key("\n".join(L3)).lower()
        V3NUMS = set(re.findall(r"(?<![\d.])(\d{1,2}\.\d{1,2}\.\d{1,2})(?![\d])", key("\n".join(L3))))
        L3k = [key(l) for l in L3]
        GIDX = defaultdict(set)
        for idx, lk in enumerate(L3k):
            for g in grams(lk, 4):
                GIDX[g].add(idx)
        st, en = body_bounds(L1)

        def cov(k):
            gs = grams(k, 4)
            if not gs:
                return 1.0
            return sum(1 for g in gs if g in G3) / len(gs)

        def best_line_ratio(k):
            """与 v3 中最相似的**单行**的相似度，用于识别"内容其实在、只是 OCR 变体"的假缺失。"""
            gs = grams(k, 4)
            if not gs:
                return 1.0
            cand = set()
            for g in gs:
                cand |= GIDX.get(g, set())
            best = 0.0
            for idx in cand:
                other = L3k[idx]
                if not other:
                    continue
                if abs(len(other) - len(k)) > max(6, int(0.5 * len(k))):
                    continue
                r = SequenceMatcher(None, k, other).ratio()
                if r > best:
                    best = r
                    if best >= 0.96:
                        break
            return best

        def presence(i):
            """内容行是否已存在于 v3。判存在需满足其一：
              ① 全文档 4-gram 覆盖率 ≥ thr_in；
              ② 与某条 v3 单行相似度 ≥ thr_dup（OCR 变体，如 ×/X、蹲/奠）；
              ③ 行首条文号已存在，且其英文术语也已在 v3（v1 的页边距条文号错位）。"""
            raw = L1[i]
            k = key(raw)
            if not is_content(k, raw):
                return None
            v = cov(k)
            if v >= a.thr_in:
                return 1.0
            if best_line_ratio(k) >= a.thr_dup:
                return 1.0
            m = ARTNO.match(raw)
            if m:
                num = "%d.%d.%d" % (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                if num in V3NUMS:
                    latin = re.sub(r"[^A-Za-z]", "", m.group(4))
                    if not (len(latin) >= 4 and latin.lower() not in K3ALL):
                        return 1.0
            return v

        # 1) 判定每行存在性
        exist = {}
        for i in range(st, en):
            exist[i] = presence(i)

        # 2) 归并缺失 run，记录锚点
        runs = []                                # (anchor_i, [line_i,...])
        i = st
        while i < en:
            if exist.get(i) is not None and exist[i] < 1.0:
                j = i
                while j < en and exist.get(j) is not None and exist[j] < 1.0:
                    j += 1
                a_i = i - 1
                while a_i >= st and exist.get(a_i) is None:
                    a_i -= 1
                if a_i >= st and exist.get(a_i) is not None:
                    runs.append((a_i, list(range(i, j))))
                else:
                    rep.append(f"[{name}] 无锚点跳过 {j - i} 行 @ {L1[i][:40]}")
                i = j
            else:
                i += 1

        # 3) 锚点 → v3 行号（4-gram 最佳匹配 + 单调递增）
        L3k = [key(l) for l in L3]
        pos_of = {}
        last = 0
        for a_i, _ in runs:
            ka = key(L1[a_i])
            gs = grams(ka, 4)
            if not gs:
                continue
            best, bestr = -1, 0.0
            for idx in range(last, len(L3k)):
                if not L3k[idx]:
                    continue
                r = sum(1 for g in gs if g in L3k[idx]) / len(gs)
                if r > bestr:
                    bestr, best = r, idx
            if best >= 0 and bestr >= a.thr_anchor:
                pos_of[a_i] = best
                last = best
            else:
                rep.append(f"[{name}] 锚点未定位({bestr:.2f}) @ {L1[a_i][:40]}")

        # 4) 应用插入（过质量门）
        ins = defaultdict(list)
        n_ins = 0
        for a_i, run in runs:
            if a_i not in pos_of:
                continue
            for li in run:
                if not insertable(L1[li]):
                    rep.append(f"[{name}] 丢弃(表格/数值) @ {L1[li][:70]}")
                    continue
                ins[pos_of[a_i]].append(L1[li])
                n_ins += 1
                rep.append(f"[{name}] +@{pos_of[a_i]:5d} | {L1[li][:70]}")

        out = []
        for idx, l in enumerate(L3):
            out.append(l)
            for extra in ins.get(idx, []):
                out.append(extra)

        # 5) 清理：孤字水印残留、页边距重复的裸条文号
        JUNK = re.compile(r"^[\u4e00-\u9fa5]$|^[A-Za-z]{1,2}$|^\d{1,2}$")
        BARENO = re.compile(r"^\s*(\d{1,2}\.\d{1,2}\.\d{1,2})\s*$")
        cleaned = []
        for l in out:
            s = l.strip()
            if JUNK.match(s):
                continue
            m = BARENO.match(s)
            if m:
                prev = "\n".join(cleaned[-3:])
                if m.group(1) in prev:
                    continue
            cleaned.append(l)
        out = cleaned
        open(os.path.join(a.out, name), "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
        print(f"{name[:34]:36s} v1正文行={en - st:5d} 缺失run={len(runs):3d} 回填行={n_ins:4d}")

    if a.report:
        open(a.report, "w", encoding="utf-8", newline="\n").write("\n".join(rep) + "\n")
    print("回填明细 ->", a.report)


if __name__ == "__main__":
    main()
