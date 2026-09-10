# -*- coding: utf-8 -*-
"""
validate_chunks.py —— 对 split_articles_v2.py 产出的 chunks 做完整性校验，
输出人可读的 markdown + csv 报告，回答一个问题：**每条条文是否完整？**

校验项：
  1. 条文号连续性（断号）——「chunk 缺失」最强的信号。按「节」分组比较第三段。
  2. 疑似误识别条文号——孤立出现的超大序号（如 6.7.41 / 2.0.41），多因 OCR 把
     "6.7.4" 后面多粘了一位数字。
  3. 重复条文号。
  4. 不以句末标点结尾的条（疑似被截断）。
  5. 疑似错位：首行系从上一段移入（后置版式重排的副产物）。
  6. 水印残余、关键词残留（信息公开 / 浏览专用 / 公告）。
  7. 超长条（>1000 字）、超短条（<10 字）、表格 chunk。

用法：
  python validate_chunks.py --chunks <chunks_v2目录> [--md 报告.md] [--csv 报告.csv]
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TERMINAL = "。；：！？）)】》”』…．."
WM = "住房城乡建设部信息公开浏览专用"
WM_FRAG = re.compile("|".join(re.escape(WM[i:i + L])
                              for L in (3, 4, 5, 6)
                              for i in range(len(WM) - L + 1)))
KEYWORD_RESIDUE = re.compile(r"信息公开|浏览专用|住房城乡建设部")


def rng(nums):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--md", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--max-ok", type=int, default=1000, help="超长阈值（字）")
    ap.add_argument("--min-ok", type=int, default=10, help="超短阈值（字）")
    args = ap.parse_args()

    jsonl = os.path.join(args.chunks, "chunks.jsonl")
    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8") if l.strip()]

    by_std = defaultdict(list)
    for r in rows:
        by_std[r["standard"]].append(r)

    report = ["# chunks 完整性校验报告", ""]
    report.append(f"- 来源：`{jsonl}`")
    report.append(f"- 总 chunk 数：**{len(rows)}**")
    report.append(f"- 涉及规范：{len(by_std)} 本")
    report.append("")

    csv_rows = []
    total_fail = 0

    for std in sorted(by_std):
        cs = by_std[std]
        name = cs[0].get("standard_name", "")
        arts = [c for c in cs if c["kind"] == "article"]
        tabs = [c for c in cs if c["kind"] == "table"]
        secs = [c for c in cs if c["kind"] == "section"]
        lens = [len(c["text"]) for c in arts] or [0]

        # —— 断号 / 超号 / 重复：按「节」分组
        sec_map = defaultdict(list)
        for c in arts:
            no = c["article_no"]
            if re.match(r"^(?:[A-Z]\.)?\d{1,2}\.\d{1,2}\.\d{1,2}$", no):
                sec_map[".".join(no.split(".")[:2])].append(no)

        gaps, outliers, dups = [], [], []
        for sec, nos in sorted(sec_map.items()):
            cnt = Counter(nos)
            for n, k in cnt.items():
                if k > 1:
                    dups.append(f"{n}×{k}")
            segs = sorted({int(n.split(".")[-1]) for n in nos})
            if len(segs) >= 2 and segs[0] != 1:
                outliers.append(f"{sec}.1 缺失（本节最小为 .{segs[0]}）")
            for a, b in zip(segs, segs[1:]):
                if b - a > 1:
                    missing = list(range(a + 1, b))
                    gaps.append(f"{sec}.{rng(missing)}")
                    # 孤立超大序号 -> 疑 OCR 误识别
                    if b > 30 and b - a > 5:
                        outliers.append(f"{sec}.{b} 疑为 {sec}.{b // 10} 之类误识别"
                                        + (f"（{sec}.{b//10} 亦未出现）"
                                           if (b // 10) not in segs else ""))

        trunc = [c for c in arts if c["text"] and c["text"].rstrip()[-1] not in TERMINAL]
        moved = [c for c in arts if "移入" in c.get("review", "")]
        leak = [c for c in arts if "疑漏" in c.get("review", "")]
        wm = [c for c in cs if WM_FRAG.search(c["text"])]
        kw = [c for c in cs if KEYWORD_RESIDUE.search(c["text"])]
        longc = [c for c in arts if len(c["text"]) > args.max_ok]
        shortc = [c for c in arts if len(c["text"]) < args.min_ok]

        bad = len(gaps) + len(dups) + len(outliers)
        total_fail += bad

        report.append(f"## {std} — {name}")
        report.append("")
        report.append("| 指标 | 值 |")
        report.append("|---|---|")
        report.append(f"| 正文条文 | {len(arts)} |")
        report.append(f"| 章节标题 | {len(secs)} |")
        report.append(f"| 表格 chunk | {len(tabs)} |")
        report.append(f"| 字数 平均/最短/最长 | {sum(lens)//len(lens)} / {min(lens)} / {max(lens)} |")
        report.append(f"| **断号** | **{len(gaps)}** |")
        report.append(f"| **疑误识别条文号** | **{len(outliers)}** |")
        report.append(f"| **疑漏号（正文并入上一条）** | **{len(leak)}** |")
        report.append(f"| 重复条文号 | {len(dups)} |")
        report.append(f"| 不以句末标点结尾 | {len(trunc)} |")
        report.append(f"| 疑似错位（首行移入） | {len(moved)} |")
        report.append(f"| 水印残余 | {len(wm)} |")
        report.append(f"| 关键词残留 | {len(kw)} |")
        report.append(f"| 超长(>{args.max_ok}字) | {len(longc)} |")
        report.append(f"| 超短(<{args.min_ok}字) | {len(shortc)} |")
        report.append("")

        if gaps:
            report.append(f"**断号**（{len(gaps)} 处）：`" + "`, `".join(gaps[:25])
                          + ("` …" if len(gaps) > 25 else "`"))
            report.append("")
        if outliers:
            report.append("**疑误识别条文号**：")
            for o in outliers[:15]:
                report.append(f"- {o}")
            report.append("")
        if dups:
            report.append(f"**重复条文号**：`" + "`, `".join(dups[:15]) + "`")
            report.append("")
        if trunc:
            report.append(f"**疑似截断**（{len(trunc)} 条，抽样 5）：")
            for c in trunc[:5]:
                report.append(f"- `{c['article_no']}` P{c['page_start']}：…{c['text'][-45:]}")
            report.append("")
        if shortc:
            report.append(f"**超短条**（{len(shortc)} 条，抽样 5）：")
            for c in shortc[:5]:
                report.append(f"- `{c['article_no']}` P{c['page_start']}：{c['text']}")
            report.append("")
        if leak:
            report.append(f"**疑漏号**（{len(leak)} 处 —— OCR 漏读了条文号，该条正文被并进上一条）：")
            for c in leak:
                m = re.search(r"疑漏([\d.]+)", c.get("review", ""))
                report.append(f"- `{c['article_no']}` P{c['page_start']}：疑漏 "
                              f"`{m.group(1) if m else '?'}`。本条末尾为：…{c['text'][-75:]}")
            report.append("  处理方式：对照该页原文，在「。」处把末尾那段切出来，"
                          "补上缺失的条文号即可。内容本身未丢失。")
            report.append("")

        for issue, items in [("断号", gaps), ("疑误识别条文号", outliers),
                             ("重复条文号", dups)]:
            for it in items:
                csv_rows.append([std, issue, it, "", "", ""])
        for c in trunc:
            csv_rows.append([std, "疑似截断", c["article_no"], c["page_start"],
                             len(c["text"]), c["text"][-60:]])
            csv_rows.append([std, "疑似错位", c["article_no"], c["page_start"],
                             len(c["text"]), c["text"][:60]])
        for c in moved:
            csv_rows.append([std, "疑似错位", c["article_no"], c["page_start"],
                             len(c["text"]), c["text"][:60]])
        for c in leak:
            csv_rows.append([std, "疑漏号", c["article_no"], c["page_start"],
                             len(c["text"]), c.get("review", "") + " | " + c["text"][-60:]])
        for c in wm:
            csv_rows.append([std, "水印残余", c["article_no"], c["page_start"],
                             len(c["text"]), c["text"][:60]])
        for c in longc:
            csv_rows.append([std, "超长条", c["article_no"], c["page_start"],
                             len(c["text"]), c["text"][:60]])
        for c in shortc:
            csv_rows.append([std, "超短条", c["article_no"], c["page_start"],
                             len(c["text"]), c["text"]])

    # —— 结论
    report.append("## 结论")
    report.append("")
    if total_fail == 0:
        report.append("**通过**：未发现断号、重复号或疑误识别条文号，"
                      "可判定每条条文边界完整。")
    else:
        report.append(f"**需复核**：共 {total_fail} 项结构性问题"
                      f"（断号 / 疑误识别 / 重复号）。")
        report.append("")
        report.append("判读顺序建议：")
        report.append("1. 先看「疑误识别条文号」——孤立超大序号多由 OCR 多粘一位数字造成，"
                      "可对照纸质规范原文直接修正条文号，不影响正文内容。")
        report.append("2. 再看「断号」——若某节的缺号正好是上一条的疑似误识别号，"
                      "则两处为同一个 OCR 问题，改号即闭合。")
        report.append("3. 最后看「疑似截断」——这些条目正文未以句末标点收尾，"
                      "需对照原页确认是否漏行。")
    report.append("")

    md_path = args.md or os.path.join(args.chunks, "校验报告.md")
    with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(report) + "\n")

    csv_path = args.csv or os.path.join(args.chunks, "校验报告.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["standard", "问题类型", "条文号", "页", "字数", "片段"])
        w.writerows(csv_rows)

    print("\n".join(report[-14:]))
    print(f"\n报告已写入:\n  {md_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
