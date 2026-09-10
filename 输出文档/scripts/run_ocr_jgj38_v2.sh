#!/usr/bin/env bash
# JGJ 38 重跑。源文件已从桌面迁到工作区 D:\WorkBuddy\项目\建筑规范PDF\
# 该版本有 69 页（含真实文本层，但数字识别已损坏，故仍用 PP-OCRv6 统一重跑）。
# --dpi 360：实测压制 604 DPI 过采样，87s/页 置信 0.9938（默认 148s/页 置信 0.9949）。
set -u
PY="C:/Users/A/.workbuddy/binaries/python/envs/paddleocr/Scripts/python.exe"
WRAP="D:/WorkBuddy/项目/输出文档/scripts/ocr_v2.py"
SRC="D:/WorkBuddy/项目/建筑规范PDF"
OUT="D:/WorkBuddy/项目/输出文档/ocr_v2"
FILT='warn|ccache|Creating model|Model files|deprecated'

echo "######## $(date +%H:%M:%S) 开始 JGJ 38（69 页，--dpi 360）########"
"$PY" "$WRAP" "$SRC/JGJ 38-2015 图书馆建筑设计规范.pdf" \
    -o "$OUT" --no-skip-text-layer --overwrite --dpi 360 2>&1 | grep -viE "$FILT"
echo "######## $(date +%H:%M:%S) 完成 JGJ 38 退出码=${PIPESTATUS[0]} ########"
echo "ALL_DONE $(date +%H:%M:%S)"
