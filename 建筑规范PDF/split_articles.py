# -*- coding: utf-8 -*-
"""
把 OCR 出来的规范 Markdown，按「条文编号边界」切分成独立 chunk。

用途：
  1. 输出 chunks.jsonl  -> 本地 RAG（Chroma / FAISS）直接消费
  2. 输出 dify/<规范>.md -> 配合 Dify「自定义分段」分隔符上传，避免条文被拦腰截断
  3. 输出 chunks.csv    -> 人工抽检 OCR 质量 / 后续做元数据标注

用法：
  python split_articles.py             # 处理全部
  python split_articles.py --only 55031  # 只处理文件名含 55031 的
  python split_articles.py --min-len 5   # 丢弃正文短于 5 字的碎片（默认 5）

输出目录： ./chunks/
"""
import os
import re
import csv
import json
import argparse

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE, "ocr_text")
OUT_DIR = os.path.join(BASE, "chunks")

# 条文号：3.1.6 / 10.2.3 / A.0.1  后面要么直接跟中文，要么跟空格/标点
ART_RE = re.compile(r"^\s*([A-Z]?\d{1,2}\.\d{1,2}\.\d{1,2})(?![.\d])\s*(.*)$")
# 章节标题行：2.6无障碍电梯和升降平台 / 3.6防灾避难（只有两段编号，且后接中文标题）
SEC_RE = re.compile(r"^\s*(\d{1,2}\.\d{1,2})(?![.\d])\s*([\u4e00-\u9fa5][\u4e00-\u9fa5A-Za-z]{1,20})\s*$")
# 粘在上一条末尾的章节标题：...。3.6防灾避难
TAIL_SEC_RE = re.compile(r"[。；]\s*(\d{1,2}\.\d{1,2})(?![.\d])\s*([\u4e00-\u9fa5][\u4e00-\u9fa5A-Za-z]{1,20})\s*$")
# 分页标记：<!-- page 12 -->
PAGE_RE = re.compile(r"^<!--\s*page\s+(\d+)\s*-->\s*$", re.I)
# 噪声行：水印残留、纯页码、分隔线
NOISE_RE = re.compile(r"^(\s*[-—=*]{3,}\s*|住房城乡建设部.*浏览专用|.*信息公开.*|\s*\d{1,3}\s*)$")

# 文件名 -> 规范元数据
META = {
    "GB 55031-2022": {"code": "GB 55031-2022", "name": "民用建筑通用规范",
                      "version": "2022", "force": "全文强制", "status": "现行"},
    "GB 55037-2022": {"code": "GB 55037-2022", "name": "建筑防火通用规范",
                      "version": "2022", "force": "全文强制", "status": "现行"},
    "GB 50352-2019": {"code": "GB 50352-2019", "name": "民用建筑设计统一标准",
                      "version": "2019", "force": "强条已废止", "status": "现行(技术参考)"},
    "GB 55019-2021": {"code": "GB 55019-2021", "name": "建筑与市政工程无障碍通用规范",
                      "version": "2021", "force": "全文强制", "status": "现行"},
    "JGJ 38-2015":   {"code": "JGJ 38-2015", "name": "图书馆建筑设计规范",
                      "version": "2015", "force": "部分强条", "status": "现行"},
}


def meta_of(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    for key, m in META.items():
        if key in stem:
            return m
    return {"code": stem, "name": stem, "version": "", "force": "未知", "status": "未知"}


def chapter_of(article_no):
    """3.1.6 -> 3 ；A.0.1 -> A"""
    m = re.match(r"^([A-Z]?\d+)", article_no)
    return m.group(1) if m else ""


# ---- OCR 数值纠错 ----
# 扫描件里 "+"/"-" 常被识别成中文 "十"/"一"，会直接导致数值答案错误
NEG_RE = re.compile(r"(?<![第之其])一(?=\d)")                       # 一10℃ -> -10℃
PLUS_RE = re.compile(r"(?<=[\d)）mMWw%℃])\s*十\s*(?=[\d(（])")       # 0.55m十(0~0.15)m -> +
# 表格混入标记
TABLE_RE = re.compile(r"表\s*\d+\.\d+|\u8868\s?\d")


def fix_ocr_numbers(s):
    s = NEG_RE.sub("-", s)
    s = PLUS_RE.sub("+", s)
    return s


def norm_text(s):
    """清理 OCR 常见噪声：行内多余空格、孤立的编号碎片、数值符号误识别。"""
    s = s.replace("\u3000", " ").strip()
    s = re.sub(r"\s{2,}", " ", s)
    # "3. 1. 6" 这种被拆散的编号还原（正文里偶尔残留）
    s = re.sub(r"(?<=\d)\.\s+(?=\d\.\s*\d)", ".", s)
    return fix_ocr_numbers(s.strip())


def split_file(path, min_len=5):
    meta = meta_of(path)
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    chunks = []
    cur = None
    page = 1
    started = False   # 是否已进入总则（1.0.1）。之前的公告/前言/目录一律丢弃
    # 公告页、前言页特征词（即便出现在正文区也不应收作条文）
    FRONT_WORDS = ("主编单位", "参编单位", "出版发行", "主要起草", "主要审查",
                   "强制性条文，必须严格执行", "现批准", "自2016", "公告")

    def flush(collect=chunks):
        """结束当前 chunk。若末尾粘了下一章节标题，切出来作为独立 section chunk。"""
        nonlocal cur
        if not cur:
            return
        txt = cur["text"]
        m = TAIL_SEC_RE.search(txt)
        if m:
            cur["text"] = txt[:m.start()] + "。"
            collect.append({
                "standard": meta["code"], "standard_name": meta["name"],
                "version": meta["version"], "force_status": meta["force"],
                "status": meta["status"], "article_no": m.group(1),
                "chapter": chapter_of(m.group(1)), "page": cur["page"],
                "kind": "section", "text": m.group(2),
            })
        if len(cur["text"].strip()) >= min_len:
            cur["kind"] = cur.get("kind", "article")
            cur["has_table"] = bool(TABLE_RE.search(cur["text"]))
            collect.append(cur)
        cur = None

    for raw in lines:
        line = raw.rstrip()
        mp = PAGE_RE.match(line)
        if mp:
            page = int(mp.group(1))
            continue
        if NOISE_RE.match(line):
            continue

        if any(w in line for w in FRONT_WORDS):
            continue

        m = ART_RE.match(line)
        if m:
            no, rest = m.group(1), m.group(2)
            # 起始门控：总则 1.0.1 之前是住建部公告、前言、目录
            # 公告里常出现「第6.1.2、6.1.3条为强制性条文」，会被误判成条文起点
            if not started:
                if re.match(r"^1\.0\.[1-9]", no):
                    started = True
                else:
                    continue
            flush()
            cur = {
                "standard": meta["code"],
                "standard_name": meta["name"],
                "version": meta["version"],
                "force_status": meta["force"],
                "status": meta["status"],
                "article_no": no,
                "chapter": chapter_of(no),
                "page": page,
                "text": norm_text(rest),
            }
            continue

        t = norm_text(line)
        if not t:
            continue

        # 独立成行的章节标题：单独存一条，便于检索章节名（如"防灾避难"）
        ms = SEC_RE.match(line)
        if ms and started:
            flush()
            cur = {
                "standard": meta["code"], "standard_name": meta["name"],
                "version": meta["version"], "force_status": meta["force"],
                "status": meta["status"], "article_no": ms.group(1),
                "chapter": chapter_of(ms.group(1)), "page": page,
                "kind": "section", "text": ms.group(2),
            }
            flush()
            continue

        if cur is None:
            # 条文号之前的内容（前言、目录）不收进知识库
            continue
        cur["text"] += t  # 中文换行直接拼接，不加空格

    flush()

    # [v2 2026-09-10] 决策：不再对超长条文二次切分。
    # 原实现会把 >800 字的条文按句号切成 3.1.6#1 / 3.1.6#2，
    # 单段内容不完整，违背「每个 chunk 必须完整」的验收要求。
    # 现在一条 = 一个 chunk；长度控制交给下游（Dify 或向量库）处理。
    return chunks


# ---- OCR 质量可疑度评分（用于生成待校对清单）----
# 数字后面紧跟一个孤立汉字，多半是双栏串行或字符误识别：如 "32m白"、"650m²l"
NOISE_NUM_CN = re.compile(r"\d\s*[a-zA-Z]?\s*[\u4e00-\u9fa5](?![\u4e00-\u9fa5])")
# 句中异常断句：数字/字母被逗号句号切碎，典型版面错序
BROKEN_RE = re.compile(r"[\u4e00-\u9fa5][，、][\u4e00-\u9fa5]{1,2}[，、][\u4e00-\u9fa5]")


def review_score(c):
    s = 0
    if c.get("has_table"):
        s += 3                                   # 表格 OCR 基本是乱的，必查
    if len(c["text"]) > 400:
        s += 2                                   # 长条文易跨栏错序
    s += min(len(NOISE_NUM_CN.findall(c["text"])), 3)   # 数字后孤立汉字
    if BROKEN_RE.search(c["text"]):
        s += 1
    return s


def reason_of(c):
    rs = []
    if c.get("has_table"):
        rs.append("含表格")
    if len(c["text"]) > 400:
        rs.append("长文易错序")
    n = len(NOISE_NUM_CN.findall(c["text"]))
    if n:
        rs.append(f"数值噪声{n}处")
    if BROKEN_RE.search(c["text"]):
        rs.append("断句异常")
    return "+".join(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--dify-sep", default="\n@@@\n",
                    help="Dify 导入文件的分段分隔串。勿用 ==== 或 ---（Markdown 标题语法会被解析器吃掉）")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "dify"), exist_ok=True)

    files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".md")) if os.path.isdir(SRC_DIR) else []
    all_chunks = []

    for fn in files:
        if args.only and args.only not in fn:
            continue
        path = os.path.join(SRC_DIR, fn)
        chunks = split_file(path, min_len=args.min_len)
        all_chunks.extend(chunks)
        lens = [len(c["text"]) for c in chunks] or [0]
        print(f"{os.path.splitext(fn)[0]:<40} 条文 {len(chunks):>4} 条  "
              f"平均 {sum(lens)//len(lens):>3} 字  最长 {max(lens):>4}")

        # Dify 导入用 md：每段带元数据头，便于检索命中与人工核对
        stem = os.path.splitext(fn)[0]
        # newline="\n" 强制 LF：Windows 下默认会把 \n 写成 \r\n，
        # 导致 Dify 里填的分隔符 \n@@@\n 匹配不上
        with open(os.path.join(OUT_DIR, "dify", stem + ".md"), "w",
                  encoding="utf-8", newline="\n") as fh:
            parts = []
            for c in chunks:
                parts.append(
                    f"【{c['standard']}】{c['article_no']}（{c['force_status']}，P{c['page']}）\n"
                    + c["text"] + "\n"
                )
            fh.write(args.dify_sep.join(parts))  # join 避免末尾多出一个空分段

    # 全量 jsonl
    jl = os.path.join(OUT_DIR, "chunks.jsonl")
    with open(jl, "w", encoding="utf-8") as fh:
        for i, c in enumerate(all_chunks):
            c["id"] = f"{c['standard']}_{c['article_no']}"
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 抽检用 csv
    with open(os.path.join(OUT_DIR, "chunks.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["standard", "article_no", "chapter", "page", "force_status", "chars", "text"])
        for c in all_chunks:
            w.writerow([c["standard"], c["article_no"], c["chapter"], c["page"],
                        c["force_status"], len(c["text"]), c["text"]])

    # 优先校对清单：OCR 质量可疑的条文（表格、版面错序、数值噪声）
    rev = [c for c in all_chunks if review_score(c) > 0]
    rev.sort(key=lambda c: -review_score(c))
    with open(os.path.join(OUT_DIR, "待校对优先清单.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["优先级", "疑点", "standard", "article_no", "page", "chars", "text"])
        for c in rev:
            w.writerow([review_score(c), reason_of(c), c["standard"], c["article_no"],
                        c["page"], len(c["text"]), c["text"]])
    print(f"待校对 {len(rev)} 条（占 {len(rev)*100//max(len(all_chunks),1)}%）-> 待校对优先清单.csv")

    print(f"\n合计 {len(all_chunks)} 条 -> {jl}")
    print(f"Dify 导入文件: {os.path.join(OUT_DIR, 'dify')}")


if __name__ == "__main__":
    main()
