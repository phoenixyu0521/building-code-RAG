#!/usr/bin/env bash
# 等前一批 OCR（4 本）跑完后，再跑 JGJ 38-2015。
# 等待依据：GB 55037 是前一批顺序里的最后一本，其 md 出现即代表前批结束。
# JGJ 38 用 --dpi 360：实测 501dpi(默认上限) 148s/页 置信0.9949，
# 360dpi 87s/页 置信0.9938 —— 用 40% 的时间拿到几乎相同的精度。
set -u
PY="C:/Users/A/.workbuddy/binaries/python/envs/paddleocr/Scripts/python.exe"
SC="C:/Users/A/.workbuddy/skills/scan-ocr/scripts/scan_ocr.py"
SRC="C:/Users/A/Desktop/建筑规范PDF"
OUT="D:/WorkBuddy/项目/输出文档/ocr_v2"
MARK="$OUT/GB 55037-2022 建筑防火通用规范.md"

echo "等待前一批完成（依据 $MARK）..."
for i in $(seq 1 240); do
  [ -f "$MARK" ] && break
  sleep 60
done
sleep 30
echo "######## $(date +%H:%M:%S) 开始 JGJ 38（--dpi 360）########"
"$PY" "$SC" "$SRC/JGJ 38-2015 图书馆建筑设计规范.pdf" -o "$OUT" \
  --no-skip-text-layer --overwrite --dpi 360 2>&1 \
  | grep -v -i "warn\|ccache\|Creating model\|Model files\|deprecated"
echo "JGJ38_DONE $(date +%H:%M:%S)"
