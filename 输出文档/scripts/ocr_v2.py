# -*- coding: utf-8 -*-
"""
ocr_v2.py —— 「按图像放置尺寸算原生密度」的 page_scale 变体 + 实测结论存档。

⚠️ 实测结论（2026-09-10）：**本模块的 page_scale 未被采用**，仅保留供 bench_scale.py
   做对照实验。scan-ocr skill 原本的 page_scale 在分块图像 PDF 上反而更好。

背景：住建部通用规范系列（GB 50352 / GB 55019 / GB 55031）每页由 2 列 × 4 行 = 8 张
图块拼成（每块 552×474 px 放在 198.7×170.6 pt 上，真实密度 2.78 = 200 DPI）。
skill 原公式用「图块像素 / 整页矩形」估密度，会低估约 2 倍（得 1.39，再被 min_side
抬到 1.739 → 渲染 690×1000）。本模块改为按图块**实际放置矩形**算，得 2.78 →
渲染 1102×1597，即无损还原原生 200 DPI。

但基准测试（GB 55019 第 10-13 页）显示：

| 渲染 | 耗时 | 平均置信 |
|---|---|---|
| 原公式 690×1000 (125 dpi) | 28.6 s/页 | **0.9923** |
| 本模块 1102×1597 (200 dpi) | 39.8 s/页 | 0.9861 |

**分辨率更高反而更慢且置信度更低** —— PP-OCRv6 内部会把检测到的文本行统一缩放到
固定高度，喂更大的图只是增加计算量；行数基本不变（25 vs 26）。所以 skill 原设置是对的。

真正的过采样问题在**整页单图**的 PDF 上，且要用 `--dpi` 显式压，而不是改密度公式：
  JGJ 38-2015 原生 3384×4824 px（≈604 DPI），原公式撞 max_side=4000 上限后仍渲染
  2802×4013 → 148 s/页 置信 0.9949；改用 `--dpi 360` → 87 s/页 置信 0.9938。
  （实测见 bench_scale.py）

用法（仅用于对照实验）：
  python bench_scale.py --pdf X.pdf --pages 10,11 --scales 1.739,2.78
"""
import importlib.util
import os as _os
import sys

SKILL = r"C:\Users\A\.workbuddy\skills\scan-ocr\scripts\scan_ocr.py"

_spec = importlib.util.spec_from_file_location("scan_ocr_base", SKILL)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


# ---------------------------------------------------------------------------
# 关键修补：禁用每页临时图的「删除」动作（2026-09-10 事故修复）
#
# 事故：scan_ocr.py 每页渲染到 `_tmp_p<N>.png`，识别后 `os.remove(tmp)`。
#   该删除虽被 try/except OSError 包裹，但本机沙箱的 safe-delete 策略在
#   「单轮删除数 ≥ 50」时抛出的是非 OSError 异常 → 直接终止进程，
#   try/except 拦不住。后果：GB 55037 在第 1 页、GB 50352 在第 3 页猝死，
#   而外层 for 循环仍继续下一本，日志里却打出「完成」，极易被误判为成功。
#
# 修法：① 所有临时图统一写同一个文件名 `_ocr_tmp.png`（不再逐页新建）；
#       ② 把该文件的删除变成 no-op。全程「只写不删」，策略无从触发。
# ---------------------------------------------------------------------------
_TMP_NAME = "_ocr_tmp.png"
_orig_join = _os.path.join
_orig_remove = _os.remove


def _join(*parts):
    p = _orig_join(*parts)
    if p.endswith(".png") and "_tmp_p" in _os.path.basename(p):
        return _orig_join(_os.path.dirname(p), _TMP_NAME)
    return p


def _remove(path, *a, **kw):
    if _os.path.basename(str(path)) == _TMP_NAME:
        return None                      # 抑制删除：复用同一文件即可
    return _orig_remove(path, *a, **kw)


_os.path.join = _join
_os.remove = _remove


def page_scale(page, max_side=None, min_side=None):
    """修正版：按每张嵌入图像的实际放置尺寸计算原生密度。"""
    max_side = base.DEFAULT_MAX_SIDE if max_side is None else max_side
    min_side = base.DEFAULT_MIN_SIDE if min_side is None else min_side

    rect = page.rect
    if rect.width <= 0 or rect.height <= 0:
        return base.FALLBACK_DPI / 72.0, "fallback(空页面)"

    best, src = 0.0, ""
    try:
        for img in page.get_images(full=True):
            xref = img[0]
            info = page.parent.extract_image(xref)
            w, h = info.get("width", 0), info.get("height", 0)
            if w <= 0 or h <= 0:
                continue
            for r in page.get_image_rects(xref):
                if r.width <= 0 or r.height <= 0:
                    continue
                s = max(w / r.width, h / r.height)
                if s > best:
                    best, src = s, "原生 %dx%d@%.0fdpi" % (w, h, s * 72)
    except Exception:
        pass

    if best <= 0:
        best, src = base.FALLBACK_DPI / 72.0, "兜底 %ddpi" % base.FALLBACK_DPI

    pw, ph = rect.width * best, rect.height * best
    longest = max(pw, ph)
    if longest > max_side:
        best *= max_side / longest
        src += " -> 限%d" % max_side
    elif longest < min_side:
        best *= min_side / longest
        src += " -> 补%d" % min_side
    return best, src


# ⚠️ 不要启用下面这行！实测本模块的 page_scale 更慢且置信度更低（见文件头表格）。
#    保留函数仅供 bench_scale.py 对照；生产一律沿用 skill 原公式。
# base.page_scale = page_scale

if __name__ == "__main__":
    sys.exit(base.main())
