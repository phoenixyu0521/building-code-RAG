#!/usr/bin/env bash
# 小实验：对已知丢号的 2 页，比较默认渲染 vs --dpi 200 能否找回条文号。
set -u
PY="C:/Users/A/.workbuddy/binaries/python/envs/paddleocr/Scripts/python.exe"
WRAP="D:/WorkBuddy/项目/输出文档/scripts/ocr_v2.py"
SRC="D:/WorkBuddy/项目/建筑规范PDF"
EXP="D:/WorkBuddy/项目/输出文档/_exp_reocr"
FILT='warn|ccache|Creating model|Model files|deprecated'
mkdir -p "$EXP"

echo "===== A) GB 55019 p11（2.9.5 被识别成 V）默认渲染 ====="
"$PY" "$WRAP" "$SRC/GB 55019-2021 建筑与市政工程无障碍通用规范.pdf" \
    -o "$EXP/a_default" --no-skip-text-layer --overwrite --only-pages 11 2>&1 | grep -viE "$FILT" | tail -3

echo "===== B) GB 55019 p11 --dpi 200 ====="
"$PY" "$WRAP" "$SRC/GB 55019-2021 建筑与市政工程无障碍通用规范.pdf" \
    -o "$EXP/b_dpi200" --no-skip-text-layer --overwrite --only-pages 11 --dpi 200 2>&1 | grep -viE "$FILT" | tail -3

echo "===== C) GB 50352 p12（2.0.26 整行丢失）默认渲染 ====="
"$PY" "$WRAP" "$SRC/GB 50352-2019 民用建筑设计统一标准.pdf" \
    -o "$EXP/c_default" --no-skip-text-layer --overwrite --only-pages 12 2>&1 | grep -viE "$FILT" | tail -3

echo "===== D) GB 50352 p12 --dpi 250 ====="
"$PY" "$WRAP" "$SRC/GB 50352-2019 民用建筑设计统一标准.pdf" \
    -o "$EXP/d_dpi250" --no-skip-text-layer --overwrite --only-pages 12 --dpi 250 2>&1 | grep -viE "$FILT" | tail -3

echo "EXP_DONE $(date +%H:%M:%S)"
