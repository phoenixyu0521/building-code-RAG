#!/usr/bin/env bash
# 正确页上的重识别实验：默认 vs --dpi 300，看能否找回丢失条文号。
set -u
PY="C:/Users/A/.workbuddy/binaries/python/envs/paddleocr/Scripts/python.exe"
WRAP="D:/WorkBuddy/项目/输出文档/scripts/ocr_v2.py"
SRC="D:/WorkBuddy/项目/建筑规范PDF"
EXP="D:/WorkBuddy/项目/输出文档/_exp_reocr2"
FILT='warn|ccache|Creating model|Model files|deprecated'
mkdir -p "$EXP"

echo "== GB 55019 p10（应含 2.9.5，此前识别成 V）默认=="
"$PY" "$WRAP" "$SRC/GB 55019-2021 建筑与市政工程无障碍通用规范.pdf" \
    -o "$EXP/55019_p10_def" --no-skip-text-layer --overwrite --only-pages 10 2>&1 | grep -viE "$FILT" | tail -2
echo "== GB 55019 p10 --dpi 300 =="
"$PY" "$WRAP" "$SRC/GB 55019-2021 建筑与市政工程无障碍通用规范.pdf" \
    -o "$EXP/55019_p10_d300" --no-skip-text-layer --overwrite --only-pages 10 --dpi 300 2>&1 | grep -viE "$FILT" | tail -2

echo "== GB 50352 p11（应含 2.0.26，此前整行丢失）默认=="
"$PY" "$WRAP" "$SRC/GB 50352-2019 民用建筑设计统一标准.pdf" \
    -o "$EXP/50352_p11_def" --no-skip-text-layer --overwrite --only-pages 11 2>&1 | grep -viE "$FILT" | tail -2
echo "== GB 50352 p11 --dpi 300 =="
"$PY" "$WRAP" "$SRC/GB 50352-2019 民用建筑设计统一标准.pdf" \
    -o "$EXP/50352_p11_d300" --no-skip-text-layer --overwrite --only-pages 11 --dpi 300 2>&1 | grep -viE "$FILT" | tail -2

echo "EXP2_DONE $(date +%H:%M:%S)"
