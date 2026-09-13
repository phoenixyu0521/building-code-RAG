#!/usr/bin/env bash
# 第三批：句中截断条文的所在页，--dpi 300 重识别。
# 依据：_plan_reocr4.py（按行首条文号定位真实页码）。
set -u
PY="C:/Users/A/.workbuddy/binaries/python/envs/paddleocr/Scripts/python.exe"
WRAP="D:/WorkBuddy/项目/输出文档/scripts/ocr_v2.py"
SRC="D:/WorkBuddy/项目/建筑规范PDF"
RE="D:/WorkBuddy/项目/输出文档/_reocr3"
FILT='warn|ccache|Creating model|Model files|deprecated'
mkdir -p "$RE"

echo "######## $(date +%H:%M:%S) 第三批重识别开始 ########"

run() {
    echo "===== $(date +%H:%M:%S) $2  pages=$3 ====="
    "$PY" "$WRAP" "$SRC/$1.pdf" -o "$RE/$2" \
        --no-skip-text-layer --overwrite --pages "$3" --dpi 300 2>&1 \
        | grep -viE "$FILT"
}

run "GB 50352-2019 民用建筑设计统一标准" "gb50352" "8,9,10,11,12,13,15,16,17,19,20,21,27,28,29,30,42,43,44,46,47,48,49,50,51,52,53,54,55"
run "GB 55019-2021 建筑与市政工程无障碍通用规范" "gb55019" "7,8,9,10,11,12,19,20,21"
run "GB 55031-2022 民用建筑通用规范" "gb55031" "5,6,7,10,11,12,13,14,16,17,18,19,20,21,22"
run "GB 55037-2022 建筑防火通用规范" "gb55037" "31,32,33,34,35,45,46,47,55,56,57,60,61,62,63"
run "JGJ 38-2015 图书馆建筑设计规范" "jgj38" "28,29,30"

echo "REOCR3_DONE $(date +%H:%M:%S)"
