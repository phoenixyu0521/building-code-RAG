# -*- coding: utf-8 -*-
"""
建筑规范 PDF → 文本（OCR）批量处理脚本

用途：住建部官网公开的规范 PDF 是"信息公开浏览专用"扫描件，无文本层，
      必须先 OCR 才能用于 RAG 建库。

用法：
    python ocr_standards.py                 # 处理全部扫描件
    python ocr_standards.py --dpi 200       # 降分辨率提速
    python ocr_standards.py --only "GB 55031"   # 只处理指定文件

输出：./ocr_text/<规范名>.md
依赖：pip install pymupdf rapidocr-onnxruntime
"""
import os
import re
import sys
import time
import argparse
import pymupdf
from rapidocr_onnxruntime import RapidOCR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "ocr_text")


import re

# 条文号被 OCR 拆成 "3. 1. 6" / "3.1. 6" 时，还原为 "3.1.6"
_ARTICLE_NO = re.compile(r"(?<![\d.])(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})(?![\d])")
# 二级编号 "5. 2" 还原为 "5.2"
_SUB_NO = re.compile(r"(?<![\d.])(\d{1,2})\s*\.\s*(\d{1,2})(?![\d])")


def fix_article_no(text):
    """还原被 OCR 拆散的条文号。数字+点+数字 的模式在中文规范正文里基本只可能是编号。"""
    text = _ARTICLE_NO.sub(r"\1.\2.\3", text)
    return text


def clean_lines(items):
    """OCR 结果按行合并：把同一自然段的碎片拼起来。

    关键：必须同时按 y（行）和 x（行内左右顺序）排序。
    只按 y 排序会导致条文号跑到句末——规范排版里条文号常在行首，
    一旦左右顺序错乱，「按条文定位」这个核心能力就废了。
    """
    lines = []
    for box, text, score in items:
        t = text.strip()
        if not t:
            continue
        # 过滤水印
        if "浏览专用" in t or "信息公开" in t:
            continue
        # (y, x, 文本)
        lines.append((box[0][1], box[0][0], t))

    if not lines:
        return []

    # 先按 y 聚类成行（y 差 < 8 视为同一行）
    lines.sort(key=lambda v: v[0])
    rows_raw = []
    cur_y, cur = lines[0][0], [lines[0]]
    for y, x, t in lines[1:]:
        if abs(y - cur_y) < 8:
            cur.append((y, x, t))
        else:
            rows_raw.append(cur)
            cur = [(y, x, t)]
            cur_y = y
    rows_raw.append(cur)

    # 行内按 x 从左到右排序后拼接
    rows = []
    for r in rows_raw:
        r.sort(key=lambda v: v[1])
        rows.append("".join(v[2] for v in r))

    return [fix_article_no(r) for r in rows]


def ocr_pdf(pdf_path, out_path, dpi=300, ocr=None):
    doc = pymupdf.open(pdf_path)
    n = len(doc)
    pages = []
    t_start = time.time()
    for i in range(n):
        pix = doc[i].get_pixmap(dpi=dpi)
        img = pix.tobytes("png")
        res, _ = ocr(img)
        if res:
            rows = clean_lines(res)
        else:
            rows = []
        pages.append((i + 1, rows))
        el = time.time() - t_start
        print(f"  [{os.path.basename(pdf_path)[:22]}] {i+1}/{n} "
              f"({el/(i+1):.1f}s/页, 剩余约 {(n-i-1)*el/(i+1)/60:.1f}min)", flush=True)
    doc.close()

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"# {os.path.splitext(os.path.basename(pdf_path))[0]}\n\n")
        fh.write(f"> 来源 PDF: {os.path.basename(pdf_path)}\n")
        fh.write(f"> 生成方式: RapidOCR (dpi={dpi})，共 {n} 页\n")
        fh.write("> 注意: OCR 文本可能存在识别误差，关键数值须对照原文核对\n\n---\n\n")
        for pno, rows in pages:
            fh.write(f"\n<!-- page {pno} -->\n\n")
            fh.write("\n".join(rows))
            fh.write("\n")
    return n


def extract_pdf_text(pdf_path, out_path):
    """有文本层的 PDF 直接抽文本，无需 OCR。"""
    doc = pymupdf.open(pdf_path)
    n = len(doc)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"# {os.path.splitext(os.path.basename(pdf_path))[0]}\n\n")
        fh.write(f"> 来源 PDF: {os.path.basename(pdf_path)}\n")
        fh.write(f"> 生成方式: 直接抽取 PDF 文本层（原生，无 OCR 误差），共 {n} 页\n\n---\n\n")
        for i in range(n):
            t = doc[i].get_text().strip()
            fh.write(f"\n<!-- page {i+1} -->\n\n")
            fh.write(t + "\n")
    doc.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--only", default="", help="只处理文件名含此关键字的 PDF")
    ap.add_argument("--force", action="store_true", help="已存在也重新生成")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    ocr = RapidOCR()

    pdfs = sorted(
        os.path.join(HERE, f)
        for f in os.listdir(HERE)
        if f.lower().endswith(".pdf") and not f.startswith("_")
    )

    for pdf in pdfs:
        name = os.path.splitext(os.path.basename(pdf))[0]
        if args.only and args.only not in name:
            continue
        d = pymupdf.open(pdf)
        pages = len(d)
        chars = sum(len(d[i].get_text().strip()) for i in range(min(pages, 12)))
        d.close()
        has_text = chars / max(pages, 1) > 200

        out = os.path.join(OUT_DIR, name + ".md")
        if os.path.exists(out) and os.path.getsize(out) > 5000 and not args.force:
            print(f"[跳过] {name} —— 已生成 {os.path.getsize(out):,} B")
            continue

        t0 = time.time()
        if has_text:
            # 有文本层：直接抽取，无需 OCR（快且零误差）
            print(f"[抽取] {name} ({pages} 页，有文本层)", flush=True)
            extract_pdf_text(pdf, out)
        else:
            print(f"\n[OCR] {name} ({pages} 页，扫描件)", flush=True)
            ocr_pdf(pdf, out, dpi=args.dpi, ocr=ocr)
        print(f"[完成] {name} -> {os.path.getsize(out):,} B，用时 {(time.time()-t0)/60:.1f} 分钟\n", flush=True)


if __name__ == "__main__":
    main()
