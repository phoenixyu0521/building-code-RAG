#!/usr/bin/env bash
# 补跑重识别：第一批排队页漏掉的 4 页（GB 50352 p21/26/63、GB 55019 p14）。
# 依据：_plan_reocr3.py 按「前后相邻现存条文号的页码区间」重新定位后的覆盖核对。
set -u
PY="C:/Users/A/.workbuddy/binaries/python/envs/paddleocr/Scripts/python.exe"
WRAP="D:/WorkBuddy/项目/输出文档/scripts/ocr_v2.py"
SRC="D:/WorkBuddy/项目/建筑规范PDF"
RE1="D:/WorkBuddy/项目/输出文档/_reocr"
RE2="D:/WorkBuddy/项目/输出文档/_reocr2"
FILT='warn|ccache|Creating model|Model files|deprecated'
mkdir -p "$RE2"

echo "######## $(date +%H:%M:%S) 等待第一批重识别结束（上限 120 分钟）########"
n=0
while [ ! -f "$RE1/gb50352/GB 50352-2019 民用建筑设计统一标准.md" ] && [ "$n" -lt 120 ]; do
    sleep 60
    n=$((n + 1))
    if [ $((n % 10)) -eq 0 ]; then echo "   ... 已等 $n 分钟，$(date +%H:%M:%S)"; fi
done
echo "######## $(date +%H:%M:%S) 第一批结束（等了 ${n} 分钟），开始补跑 ########"
sleep 15

run() {
    local name="$1" tag="$2" pages="$3"
    echo "===== $(date +%H:%M:%S) 补跑 $tag  pages=$pages ====="
    "$PY" "$WRAP" "$SRC/$name.pdf" -o "$RE2/$tag" \
        --no-skip-text-layer --overwrite --pages "$pages" --dpi 300 2>&1 \
        | grep -viE "$FILT"
}

run "GB 50352-2019 民用建筑设计统一标准" "gb50352" "21,26,63"
run "GB 55019-2021 建筑与市政工程无障碍通用规范" "gb55019" "14"

echo "REOCR2_DONE $(date +%H:%M:%S)"
