#!/usr/bin/env bash
# 重跑事故中夭折的 3 本规范。
# 用 ocr_v2.py 包装器（已修补：临时图复用单文件 + 禁用删除，
# 规避沙箱 safe-delete 在单轮 ≥50 次删除时的硬中断）。
set -u
PY="C:/Users/A/.workbuddy/binaries/python/envs/paddleocr/Scripts/python.exe"
WRAP="D:/WorkBuddy/项目/输出文档/scripts/ocr_v2.py"
SRC="C:/Users/A/Desktop/建筑规范PDF"
OUT="D:/WorkBuddy/项目/输出文档/ocr_v2"
FILT='warn|ccache|Creating model|Model files|deprecated'

mkdir -p "$OUT"

echo "######## $(date +%H:%M:%S) 开始 GB 50352（68 页）########"
"$PY" "$WRAP" "$SRC/GB 50352-2019 民用建筑设计统一标准.pdf" \
    -o "$OUT" --no-skip-text-layer --overwrite 2>&1 | grep -viE "$FILT"
echo "######## $(date +%H:%M:%S) 完成 GB 50352 退出码=${PIPESTATUS[0]} ########"

echo "######## $(date +%H:%M:%S) 开始 GB 55037（66 页）########"
"$PY" "$WRAP" "$SRC/GB 55037-2022 建筑防火通用规范.pdf" \
    -o "$OUT" --no-skip-text-layer --overwrite 2>&1 | grep -viE "$FILT"
echo "######## $(date +%H:%M:%S) 完成 GB 55037 退出码=${PIPESTATUS[0]} ########"

echo "######## $(date +%H:%M:%S) 开始 JGJ 38（69 页，--dpi 360 抑制过采样）########"
"$PY" "$WRAP" "$SRC/JGJ 38-2015 图书馆建筑设计规范.pdf" \
    -o "$OUT" --no-skip-text-layer --overwrite --dpi 360 2>&1 | grep -viE "$FILT"
echo "######## $(date +%H:%M:%S) 完成 JGJ 38 退出码=${PIPESTATUS[0]} ########"

echo "ALL_DONE $(date +%H:%M:%S)"
