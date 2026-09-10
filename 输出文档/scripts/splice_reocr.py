# -*- coding: utf-8 -*-
"""把 --dpi 300 重识别得到的页面并回主 md。
判定：仅当新页含「旧页所缺的条文号」时才替换，避免整体回退。
用法: python splice_reocr.py [--dry-run]
"""
import argparse
import os
import re
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")

OUT = "D:/WorkBuddy/项目/输出文档/ocr_v2"
RE = "D:/WorkBuddy/项目/输出文档/_reocr"
JOBS = [
    ("gb55019", "GB 55019-2021 建筑与市政工程无障碍通用规范.md"),
    ("gb55031", "GB 55031-2022 民用建筑通用规范.md"),
    ("gb50352", "GB 50352-2019 民用建筑设计统一标准.md"),
]

ARTNO = re.compile(r"(?:[A-Z]\.)?\d{1,2}\.\d{1,2}\.\d{1,2}")
PAGE_SPLIT = re.compile(r"(<!--\s*page\s+(\d+)\s*-->)")


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
    return set(ARTNO.findall(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for tag, fname in JOBS:
        src_new = os.path.join(RE, tag, fname)
        dst = os.path.join(OUT, fname)
        if not os.path.exists(src_new):
            print(f"[跳过] {tag}: 无重识别结果 {src_new}")
            continue

        old_txt = open(dst, encoding="utf-8").read()
        new_txt = open(src_new, encoding="utf-8").read()
        old_pg, new_pg = pages_of(old_txt), pages_of(new_txt)

        print("=" * 72)
        print(f"{fname}")
        replaced = []
        for pg, new_body in sorted(new_pg.items()):
            if pg not in old_pg:
                continue
            gained = artnums(new_body) - artnums(old_pg[pg])
            if gained:
                replaced.append(pg)
                print(f"   p{pg}: 新增条文号 {sorted(gained)} -> 替换")
                old_pg[pg] = new_body
            else:
                lost = artnums(old_pg[pg]) - artnums(new_body)
                print(f"   p{pg}: 无新增（旧独有 {sorted(lost) if lost else '无'}）-> 保留原页")

        if not replaced:
            print("   本本无页被替换")
            continue

        # 重组
        order = sorted(old_pg)
        merged = "".join(
            (f"<!-- page {p} -->" if p else "") + old_pg[p] for p in order)
        if args.dry_run:
            print(f"   [dry-run] 将替换 {len(replaced)} 页: {replaced}")
        else:
            bak = dst + ".bak"
            if not os.path.exists(bak):
                shutil.copy2(dst, bak)
            with open(dst, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(merged)
            print(f"   已写回，替换 {len(replaced)} 页: {replaced}（原文件备份 {os.path.basename(bak)}）")


if __name__ == "__main__":
    main()
