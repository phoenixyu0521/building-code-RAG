# -*- coding: utf-8 -*-
"""
bench_scale.py —— 对比两档渲染分辨率在「分块图像 PDF」上的耗时与识别置信度。

用途：为「按原生像素渲染」的选择提供实测依据，避免拍脑袋定参数。
只读 PDF，不改动任何既有文件。

用法：
  python bench_scale.py                 # 默认测 GB 55019 第 10-13 页
  python bench_scale.py --pdf X.pdf --pages 10,11,12
"""
import argparse
import importlib.util
import os
import sys
import time

import pymupdf

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SKILL = r"C:\Users\A\.workbuddy\skills\scan-ocr\scripts\scan_ocr.py"
_spec = importlib.util.spec_from_file_location("scan_ocr_base", SKILL)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

DEFAULT_PDF = r"C:\Users\A\Desktop\建筑规范PDF\GB 55019-2021 建筑与市政工程无障碍通用规范.pdf"
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=DEFAULT_PDF)
    ap.add_argument("--pages", default="10,11,12,13", help="1-based 页码")
    ap.add_argument("--scales", default="", help="显式指定缩放比，逗号分隔；留空则对比 原公式/修正后")
    args = ap.parse_args()

    pages = [int(x) - 1 for x in args.pages.split(",") if x.strip()]
    doc = pymupdf.open(args.pdf)
    pg = doc[pages[0]]

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if args.scales:
        combos = [(float(x), "scale=%.2f (%ddpi)" % (float(x), float(x) * 72))
                  for x in args.scales.split(",") if x.strip()]
    else:
        from ocr_v2 import page_scale as fixed_scale
        fixed_scale, src_fixed = fixed_scale(pg)
        old_scale, src_old = base.page_scale(pg)
        combos = [(old_scale, "原公式"), (fixed_scale, "修正后")]

    print("页面 %.1fx%.1f pt" % (pg.rect.width, pg.rect.height))
    for sc, label in combos:
        print("  %-22s -> 渲染 %dx%d" % (label, pg.rect.width * sc, pg.rect.height * sc))
    print()

    t0 = time.time()
    ocr = base.build_ocr()
    print("模型加载 %.1fs\n" % (time.time() - t0), flush=True)

    for scale, label in combos:
        tot, confs, lines = 0.0, [], []
        for i in pages:
            p = doc[i]
            p.get_pixmap(matrix=pymupdf.Matrix(scale, scale)).save(TMP)
            t = time.time()
            items, _ = base.ocr_image(ocr, TMP)
            dt = time.time() - t
            conf = sum(x["s"] for x in items) / max(len(items), 1)
            tot += dt
            confs.append(conf)
            lines.append(len(items))
            print("  [%-22s] p%-3d %6.1fs  %3d 行  置信 %.4f"
                  % (label, i + 1, dt, len(items), conf), flush=True)
        print("  ==> %s: 平均 %.1fs/页  平均置信 %.4f  平均 %.0f 行/页\n"
              % (label, tot / len(pages), sum(confs) / len(confs), sum(lines) / len(lines)),
              flush=True)

    doc.close()
    try:
        os.remove(TMP)
    except OSError:
        pass


if __name__ == "__main__":
    main()
