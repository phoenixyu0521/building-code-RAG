#!/usr/bin/env bash
# 等 JGJ 38 OCR 结束后，对 14 处 OCR 丢号所在的 30 页做 --dpi 300 重识别。
# 实验依据（scripts/exp_reocr2.sh）：默认渲染下 GB 55019 p10 的 2.9.5 会丢失，
# --dpi 300 能找回；GB 50352 p11 的 2.0.26 同样找回。
set -u
PY="C:/Users/A/.workbuddy/binaries/python/envs/paddleocr/Scripts/python.exe"
WRAP="D:/WorkBuddy/项目/输出文档/scripts/ocr_v2.py"
SRC="D:/WorkBuddy/项目/建筑规范PDF"
OUT="D:/WorkBuddy/项目/输出文档/ocr_v2"
RE="D:/WorkBuddy/项目/输出文档/_reocr"
FILT='warn|ccache|Creating model|Model files|deprecated'
J38="$OUT/JGJ 38-2015 图书馆建筑设计规范.md"
mkdir -p "$RE"

echo "######## $(date +%H:%M:%S) 等待 JGJ 38 完成（上限 240 分钟）########"
n=0
while [ ! -f "$J38" ] && [ "$n" -lt 240 ]; do
    sleep 60
    n=$((n + 1))
    if [ $((n % 15)) -eq 0 ]; then
        echo "   ... 已等 $n 分钟，$(date +%H:%M:%S)"
    fi
done
if [ -f "$J38" ]; then
    echo "######## $(date +%H:%M:%S) JGJ 38 已完成，开始重识别 ########"
else
    echo "######## $(date +%H:%M:%S) 等待超时（${n} 分钟），仍继续执行 ########"
fi
sleep 20

run() {
    local name="$1" tag="$2" pages="$3"
    echo "===== $(date +%H:%M:%S) $tag  pages=$pages ====="
    "$PY" "$WRAP" "$SRC/$name.pdf" -o "$RE/$tag" \
        --no-skip-text-layer --overwrite --pages "$pages" --dpi 300 2>&1 \
        | grep -viE "$FILT"
}

run "GB 55019-2021 建筑与市政工程无障碍通用规范" "gb55019" "10,11,12,13"
run "GB 55031-2022 民用建筑通用规范"             "gb55031" "6,7,12,13"
run "GB 50352-2019 民用建筑设计统一标准"          "gb50352" "11,12,19,20,23,24,25,27,28,29,30,37,38,39,41,42,47,48,49,50,61,62"

echo "REOCR_DONE $(date +%H:%M:%S)"
