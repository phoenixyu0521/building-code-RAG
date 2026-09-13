# -*- coding: utf-8 -*-
"""propose_reanchor.py —— 生成「条文号行归位」提案（dry-run，不写文件）。

原理：v4 的**内容行顺序是对的**（v2 骨架按页内自上而下），只是 merge_v1_v2 把
v1 回填回来的**条文号/章节标题行**插到了错误锚点，导致这些行的正文被错配。
本脚本以 v1（RapidOCR，阅读顺序正确）为参照：
  对 v4 中每个条文号行 K，取 v1 中 K 之后的第一行正文作为「签名」，
  在 v4 中定位该签名行，提案把 K 移到签名行之前。
输出提案供人工复核后写入 build_v4.py 的 MOV 白名单。
"""
import argparse
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ART = re.compile(r"^\s*(\d{1,2})\s*[.．]\s*(\d{1,2})\s*[.．]\s*(\d{1,2})\s*(.*)$")
SEC = re.compile(r"^\s*(\d{1,2})\s*[.．]\s*(\d{1,2})\s+(\S.*)$")
JUNK = re.compile(r"^\s*(<!--\s*page|\d{1,3}\s*$)")

STD = {
    "民用建筑设计统一标准": "GB 50352-2019 民用建筑设计统一标准.md",
    "建筑与市政工程无障碍通用规范": "GB 55019-2021 建筑与市政工程无障碍通用规范.md",
    "民用建筑通用规范": "GB 55031-2022 民用建筑通用规范.md",
    "建筑防火通用规范": "GB 55037-2022 建筑防火通用规范.md",
    "图书馆建筑设计规范": "JGJ 38-2015 图书馆建筑设计规范.md",
}


def nz(s):
    return re.sub(r"[\s\u3000]", "", s)


def artkey(line):
    m = ART.match(line.strip())
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def body_end(lines):
    for i, l in enumerate(lines):
        if re.match(r"^\s*(本规范用词说明|引用标准名录|条文说明|本规范用词|附录)", l.strip()):
            return i
    return len(lines)


def v1_signatures(v1):
    """返回 {artkey: 签名文本}（v1 中该条文号之后的第一个正文行）"""
    out = {}
    n = len(v1)
    for i, l in enumerate(v1):
        k = artkey(l)
        if not k or k in out:
            continue
        for j in range(i + 1, min(i + 4, n)):
            s = v1[j].strip()
            if not s or JUNK.match(s) or artkey(s):
                continue
            out[k] = nz(s)
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../ocr_v4")
    ap.add_argument("--v1", default="../_backup_v1_20260910/ocr_text")
    a = ap.parse_args()

    for name in sorted(os.listdir(a.src)):
        if not name.endswith(".md"):
            continue
        lines = open(os.path.join(a.src, name), encoding="utf-8").read().splitlines()
        v1 = open(os.path.join(a.v1, name), encoding="utf-8").read().splitlines()
        sig = v1_signatures(v1)
        be = body_end(lines)
        print("=" * 100)
        print("##", name)
        for i in range(be):
            k = artkey(lines[i])
            if not k:
                continue
            s = sig.get(k)
            if not s:
                continue
            # v4 中紧邻的下一行是否已是签名行？
            if i + 1 < len(lines) and s[:12] and s[:12] in nz(lines[i + 1]):
                continue
            # 在 v4 中向后找签名行
            tgt = None
            for j in range(i + 1, len(lines)):
                if s[:12] and s[:12] in nz(lines[j]):
                    tgt = j
                    break
            if tgt is None:
                print("  [NO-TARGET] %-9s @%4d | %s | sig=%s" %
                      ("%d.%d.%d" % k, i, lines[i][:44], s[:30]))
                continue
            print("  MOVE %-9s @%4d -> before @%4d | line=%s" %
                  ("%d.%d.%d" % k, i, tgt, lines[i][:40]))
            print("        target: %s" % lines[tgt][:70])


if __name__ == "__main__":
    main()
