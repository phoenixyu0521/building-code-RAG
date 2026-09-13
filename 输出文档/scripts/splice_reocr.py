# -*- coding: utf-8 -*-
"""把 --dpi 300 重识别得到的页面并回 ocr_v2 的主 md。

替换判据（保守）：新页必须**新增**旧页所缺的条文号，且不得净丢失条文号。
覆盖两个来源目录：_reocr（第一批排队）与 _reocr2（补跑）。

用法:
  python splice_reocr.py --dry-run
  python splice_reocr.py
"""
import argparse
import os
import re
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE = "D:/WorkBuddy/项目/输出文档"
OUT = os.path.join(BASE, "ocr_v2")
DIRS = [os.path.join(BASE, "_reocr"), os.path.join(BASE, "_reocr2")]
JOBS = [
    ("gb55019", "GB 55019-2021 建筑与市政工程无障碍通用规范.md"),
    ("gb55031", "GB 55031-2022 民用建筑通用规范.md"),
    ("gb50352", "GB 50352-2019 民用建筑设计统一标准.md"),
]

ARTNO = re.compile(r"(?:[A-Z]\.)?\d{1,2}\.\d{1,2}\.\d{1,2}")
PAGE_SPLIT = re.compile(r"(<!--\s*page\s+(\d+)\s*-->)")
# 与切分器一致的编号内空格收敛，否则 `2. 9. 6` 会被误判为「丢失」
SPACE_FIX = re.compile(r"([A-Z0-9])\.\s+(?=\d)")


def norm(t):
    return SPACE_FIX.sub(r"\1.", t)


def pages_of(text):
    """-> {page_no: text}；页外内容归 page 0。"""
    parts = PAGE_SPLIT.split(text)
    out = {}
    if parts[0].strip():
        out[0] = parts[0]
    for i in range(1, len(parts), 3):
        out[int(parts[i + 1])] = parts[i + 2]
    return out


def artnums(t):
    return set(ARTNO.findall(norm(t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for tag, fname in JOBS:
        dst = os.path.join(OUT, fname)
        # 收集两个来源目录里该规范的重识别页（后写的优先）
        cand = {}
        for d in DIRS:
            p = os.path.join(d, tag, fname)
            if os.path.exists(p):
                cand.update(pages_of(open(p, encoding="utf-8").read()))
        if not cand:
            print(f"[跳过] {tag}: 无重识别结果")
            continue

        old_pg = pages_of(open(dst, encoding="utf-8").read())
        print("=" * 72)
        print(fname)
        replaced = []
        for pg, new_body in sorted(cand.items()):
            if pg not in old_pg:
                print(f"   p{pg}: 主文件无此页 -> 忽略")
                continue
            gained = artnums(new_body) - artnums(old_pg[pg])
            lost = artnums(old_pg[pg]) - artnums(new_body)
            if gained and len(gained) >= len(lost):
                replaced.append(pg)
                print(f"   p{pg}: 新增 {sorted(gained)}" +
                      (f" / 同时丢失 {sorted(lost)}" if lost else "") + " -> 替换")
                old_pg[pg] = new_body
            elif gained:
                print(f"   p{pg}: 新增 {sorted(gained)} 但丢失 {sorted(lost)}（净损失）-> 保留原页")
            else:
                print(f"   p{pg}: 无新增" + (f"（旧独有 {sorted(lost)}）" if lost else "") + " -> 保留原页")

        if not replaced:
            print("   无页被替换")
            continue

        order = sorted(old_pg)
        merged = "".join((f"<!-- page {p} -->" if p else "") + old_pg[p] for p in order)
        if args.dry_run:
            print(f"   [dry-run] 将替换 {len(replaced)} 页: {replaced}")
        else:
            bak = dst + ".bak"
            if not os.path.exists(bak):
                shutil.copy2(dst, bak)
            with open(dst, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(merged)
            print(f"   已写回，替换 {len(replaced)} 页: {replaced}"
                  f"（备份 {os.path.basename(bak)}）")


if __name__ == "__main__":
    main()
