# -*- coding: utf-8 -*-
"""生成并校验「检索评测集」（评测集.jsonl）。

评测集只管一件事：这条 query **应该召回哪条条文**。
它不判「答案写得对不对」—— 那是生成阶段的评测（见 知识库测试题.md）。

用法：  python build_evalset.py
产出：  D:\\WorkBuddy\\项目\\输出文档\\检索评测\\评测集.jsonl
校验：  ① 每个 gold 条文号必须在 chunks_v5 里真实存在
        ② 每条 query 去重、qid 唯一、长度 ≤250 字符（Dify hit-testing 硬限制）
"""
import json
import os
import re
import sys
import collections

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJ = r"D:\WorkBuddy\项目"
CHUNKS = os.path.join(PROJ, r"输出文档\chunks_v5\chunks.jsonl")
DIFY_DIR = os.path.join(PROJ, r"输出文档\chunks_v5\dify")
OUT = os.path.join(PROJ, r"输出文档\检索评测\评测集.jsonl")

JGJ = "JGJ 38-2015"
GB52 = "GB 50352-2019"
GB37 = "GB 55037-2022"
GB31 = "GB 55031-2022"
GB19 = "GB 55019-2021"

# (qid, query, standard, gold, type, difficulty, note)
#   type: keyword=术语直给 / semantic=换词口语化 / numeric=问指标数值
#         table=表格 / unanswerable=库里不该有的
#   gold: 可接受的金标条文号列表（命中任意一条即算召回成功）
P = [
    # ================= JGJ 38-2015 图书馆（27）=================
    ("JGJ-01", "图书馆书库的净高要求是多少？", JGJ, ["4.2.8"], "numeric", "easy", "净高≥2.40m，有梁管线处≥2.30m"),
    ("JGJ-02", "二层到五层的书库需要设置什么提升设备？", JGJ, ["4.2.10"], "keyword", "medium", "二层至五层设书刊提升设备，六层以上设专用货梯"),
    ("JGJ-03", "特藏书库的入口需要做什么构造处理？", JGJ, ["4.2.6"], "semantic", "medium", "设缓冲间 + 两侧密闭门"),
    ("JGJ-04", "卫生间和开水间能不能布置在书库正上方？", JGJ, ["4.2.7"], "semantic", "medium", "经常积水的场所不应设在书库内部及其直接上方"),
    ("JGJ-05", "图书馆四层以上设阅览室时，电梯有什么要求？", JGJ, ["4.1.4"], "numeric", "medium", "应设读者电梯，至少一台无障碍电梯"),
    ("JGJ-06", "新建公共图书馆的建筑密度上限是多少？", JGJ, ["3.2.4"], "numeric", "easy", "不宜大于40%"),
    ("JGJ-07", "图书馆基地的绿地率要求是多少？", JGJ, ["3.2.6"], "numeric", "easy", "不宜小于30%"),
    ("JGJ-08", "开架阅览桌之间的主通道净宽是多少？", JGJ, ["4.3.5-表", "4.3.5"], "table", "hard", "表4.3.5：开架1.50m，闭架1.20m"),
    ("JGJ-09", "普通阅览室每个座位需要多少使用面积？", JGJ, ["附录B", "4.3.14"], "table", "medium", "表B.0.1：1.8~2.3 m²/座"),
    ("JGJ-10", "少年儿童阅览区要不要单独设出入口？", JGJ, ["3.2.3"], "semantic", "easy", "宜设单独对外出入口和室外活动场地"),
    ("JGJ-11", "视障阅览室应该和哪个空间连通？", JGJ, ["4.3.12"], "semantic", "medium", "应与盲文书库相连通"),
    ("JGJ-12", "基本书库、特藏书库与其他部位之间该怎么分隔？", JGJ, ["6.2.1"], "keyword", "medium", "防火墙 + 甲级防火门"),
    ("JGJ-13", "图书馆每层至少要有几个安全出口？", JGJ, ["6.4.1"], "numeric", "easy", "不应少于两个，并分散布置"),
    ("JGJ-14", "公共阅览室只设一个疏散门时，净宽不能小于多少？", JGJ, ["6.4.4"], "numeric", "easy", "1.20m"),
    ("JGJ-15", "藏书量超过100万册的高层图书馆，耐火等级是几级？", JGJ, ["6.1.2"], "numeric", "easy", "一级"),
    ("JGJ-16", "基本书库的温度和相对湿度控制在什么范围？", JGJ, ["8.2.3"], "numeric", "medium", "5~30℃，相对湿度 30%~65%"),
    ("JGJ-17", "特藏书库24小时内温湿度变化不能超过多少？", JGJ, ["8.2.4"], "numeric", "medium", "温度±2℃，相对湿度±5%"),
    ("JGJ-18", "书库机械通风时空气流速的上限是多少？", JGJ, ["8.2.14"], "numeric", "medium", "不应大于0.5m/s"),
    ("JGJ-19", "书库照明灯具和书刊资料之间要保持多大距离？", JGJ, ["8.3.7"], "numeric", "medium", "垂直距离不应小于0.50m"),
    ("JGJ-20", "珍善本书库里可以穿给排水管道吗？", JGJ, ["8.1.2"], "semantic", "hard", "不应有水管进入；其他书库除消防管道外不应有给排水管穿过"),
    ("JGJ-21", "超过300座的报告厅应该怎么布置？", JGJ, ["4.5.5"], "numeric", "medium", "应独立设置并与阅览区隔离"),
    ("JGJ-22", "寄存处的存物柜数量按什么比例确定？", JGJ, ["4.5.3"], "numeric", "hard", "按阅览座位的25%"),
    ("JGJ-23", "书库工作人员专用楼梯的梯段净宽和坡度有什么要求？", JGJ, ["4.2.9"], "numeric", "hard", "净宽不宜小于0.80m，坡度不应大于45°"),
    ("JGJ-24", "图书馆的疏散门装门禁可以吗？有什么条件？", JGJ, ["6.4.6"], "semantic", "hard", "可设门禁，但紧急时应易于从内部开启"),
    ("JGJ-25", "图书馆主要出入口和特藏书库要不要设安防装置？", JGJ, ["5.8.1"], "semantic", "medium", "应设安全防范装置"),
    ("JGJ-26", "书库外窗的开启扇怎么防蚊蝇？", JGJ, ["5.7.2"], "semantic", "hard", "应采取防蚊蝇措施"),
    ("JGJ-27", "目录检索空间里计算机检索台每台占多少使用面积？", JGJ, ["4.4.5"], "numeric", "hard", "按2m²计算"),

    # ================= GB 50352-2019（12）=================
    ("GB52-01", "半地下室是怎么定义的？", GB52, ["2.0.16"], "keyword", "easy", "超过净高1/3且不超过1/2"),
    ("GB52-02", "地下室和半地下室的区分标准是什么？", GB52, ["2.0.15", "2.0.16"], "semantic", "medium", "1/2 为界；易混题"),
    ("GB52-03", "层高和室内净高的定义有什么区别？", GB52, ["2.0.13", "2.0.14"], "semantic", "medium", "易混题"),
    ("GB52-04", "建筑基地的机动车出入口位置有什么规定？", GB52, ["4.2.4"], "keyword", "medium", "主干路交叉口距离等"),
    ("GB52-05", "大型人员密集的建筑基地要满足哪些要求？", GB52, ["4.2.5"], "keyword", "medium", "交通、文化、体育、商业等"),
    ("GB52-06", "建筑连接体的净宽限制是多少？", GB52, ["4.4.4"], "numeric", "medium", "交通功能的不宜大于9.0m，地上≥3.0m，地下≥4.0m"),
    ("GB52-07", "建筑物的哪些部分可以突出建筑控制线？", GB52, ["4.3.3", "4.2.2"], "semantic", "hard", "地下室、窗井、台阶、坡道雨篷等除外"),
    ("GB52-08", "民用建筑的设计使用年限分成哪几类？", GB52, ["3.2.1-表", "3.2"], "table", "medium", "表3.2.1：5/25/50/100年"),
    ("GB52-09", "严寒地区对建筑的基本要求是什么？", GB52, ["3.3.1-表", "3.3"], "table", "hard", "库内仅存第1条主要指标，另两项缺失"),
    ("GB52-10", "建筑平面定位线尺寸要符合什么模数要求？", GB52, ["3.5.2"], "keyword", "hard", "基本模数倍数"),
    ("GB52-11", "防灾避难场所和设施有什么要求？", GB52, ["3.6.4"], "semantic", "medium", "应保障安全、长期备用、便于管理"),
    ("GB52-12", "采光系数的定义是什么？", GB52, ["2.0.33", "2.0.34"], "keyword", "medium", "术语定义题"),

    # ================= GB 55037-2022 防火（14）=================
    ("GB37-01", "消防救援口有什么设置要求？", GB37, ["2.2.3"], "keyword", "easy", "5 款，含 2 个、1.0m/0.8m、永久性标志"),
    ("GB37-02", "哪些建筑必须设置消防电梯？", GB37, ["2.2.6"], "keyword", "medium", "例外条款多，易漏"),
    ("GB37-03", "消防电梯的载重量要求是多少？", GB37, ["2.2.10"], "numeric", "easy", "不应小于800kg"),
    ("GB37-04", "消防电梯前室有什么要求？", GB37, ["2.2.8"], "keyword", "medium", "除三种例外均应设前室"),
    ("GB37-05", "消防车登高操作场地对应的范围内要设什么？", GB37, ["2.2.2"], "keyword", "medium", "直通室外的楼梯或直通楼梯间的入口"),
    ("GB37-06", "靠外墙的封闭楼梯间顶部要不要设固定窗？", GB37, ["2.2.4"], "semantic", "hard", "设机械加压送风系统时，顶部或最上层外墙设常闭式应急排烟窗"),
    ("GB37-07", "消防水泵房布置有什么要求？", GB37, ["4.1.7"], "keyword", "medium", "耐火等级与防火分隔"),
    ("GB37-08", "消防控制室的防火分隔要求是什么？", GB37, ["4.1.8"], "keyword", "medium", "与消防水泵房同章，易混"),
    ("GB37-09", "甲、乙类物品运输车的汽车库与人员密集场所的防火间距是多少？", GB37, ["3.1.3"], "numeric", "medium", "不应小于50m"),
    ("GB37-10", "建筑周围必须设置什么消防设施？", GB37, ["3.4.1"], "semantic", "hard", "消防车道、消防车登高操作场地等"),
    ("GB37-11", "建筑高度大于100m的民用建筑，防火间距怎么确定？", GB37, ["3.3.1"], "numeric", "medium", "易与 3.1.2 混"),
    ("GB37-12", "埋深大于15m的地铁车站公共区要设什么？", GB37, ["2.2.7"], "numeric", "hard", "消防专用通道"),
    ("GB37-13", "公共建筑在什么情况下可以只设一个安全出口？", GB37, ["7.4.1-表", "7.4"], "table", "hard", "表7.4.1 按层数/面积"),
    ("GB37-14", "疏散楼梯每100人需要多宽的净宽度？", GB37, ["7.4.7-表", "7.4"], "table", "medium", "表7.4.7"),

    # ================= GB 55031-2022（12）=================
    ("GB31-01", "哪些建筑空间不计算建筑面积？", GB31, ["3.1.6"], "keyword", "medium", "枚举 5 款"),
    ("GB31-02", "建筑空间计算建筑面积的层高门槛是多少？", GB31, ["3.1.4"], "numeric", "medium", "2.20m"),
    ("GB31-03", "室内净高的最低要求在哪一条？", GB31, ["3.2.7"], "keyword", "medium", "各功能场所最低净高"),
    ("GB31-04", "公共楼梯梯段的最小净宽怎么确定？", GB31, ["5.3.2"], "numeric", "medium", "按人流股数，每股0.55m"),
    ("GB31-05", "公共楼梯休息平台的宽度要求是什么？", GB31, ["5.3.5"], "numeric", "easy", "≥梯段净宽且≥1.20m"),
    ("GB31-06", "公共楼梯梯段和休息平台过道处的净高要求是多少？", GB31, ["5.3.7"], "numeric", "easy", "过道≥2.00m，梯段≥2.20m"),
    ("GB31-07", "公共楼梯一个梯段最多几级踏步？", GB31, ["5.3.8"], "numeric", "easy", "不应少于2级、不应超过18级"),
    ("GB31-08", "台阶踏步不足2级时应该怎么处理？", GB31, ["5.2.3"], "semantic", "medium", "按人行坡道设置"),
    ("GB31-09", "台阶和人行坡道总高度超过多少要设防护措施？", GB31, ["5.2.1"], "numeric", "medium", "0.70m"),
    ("GB31-10", "公共楼梯什么情况下需要两侧设扶手？", GB31, ["5.3.4"], "numeric", "medium", "梯段净宽达3股人流宽度"),
    ("GB31-11", "建筑出入口的设置要考虑哪些要求？", GB31, ["5.1.1"], "semantic", "hard", "场地条件、使用功能、交通组织、安全疏散"),
    ("GB31-12", "建筑基地内的道路系统要满足什么要求？", GB31, ["4.3.1"], "semantic", "medium", "顺畅便捷、消防与无障碍通行"),

    # ================= GB 55019-2021 无障碍（12）=================
    ("GB19-01", "无障碍通道的通行净宽要求是多少？", GB19, ["2.2.2"], "numeric", "easy", "≥1.20m，人员密集公共场所≥1.80m"),
    ("GB19-02", "轮椅坡道的通行净宽是多少？", GB19, ["2.3.2"], "numeric", "easy", "≥1.20m"),
    ("GB19-03", "轮椅坡道的纵向坡度最大是多少？", GB19, ["2.3.1"], "numeric", "medium", "不应大于1:12"),
    ("GB19-04", "轮椅坡道什么情况下需要在两侧设扶手？", GB19, ["2.3.4"], "numeric", "medium", "高度>300mm 且纵坡>1:20"),
    ("GB19-05", "无障碍出入口有哪几种形式？", GB19, ["2.4.1"], "keyword", "medium", "3 种"),
    ("GB19-06", "无障碍出入口的门前平台净深度要求是多少？", GB19, ["2.4.2"], "numeric", "medium", "不应小于1.50m"),
    ("GB19-07", "无障碍门有高差时该怎么处理？", GB19, ["2.5.3"], "numeric", "hard", "高差≤15mm，斜面过渡"),
    ("GB19-08", "无障碍通道上能用旋转门吗？", GB19, ["2.5.2"], "semantic", "easy", "不应使用"),
    ("GB19-09", "无障碍自动门开启后的通行净宽是多少？", GB19, ["2.5.5"], "numeric", "medium", "不应小于1.00m"),
    ("GB19-10", "装有闭门器的无障碍门，闭门时间有什么要求？", GB19, ["2.5.8"], "numeric", "hard", "不应小于3s"),
    ("GB19-11", "无障碍通道上有井盖时，孔洞尺寸限制是多少？", GB19, ["2.2.4"], "numeric", "hard", "宽度或直径不应大于13mm"),
    ("GB19-12", "自动扶梯和楼梯下方的低矮空间怎么处理？", GB19, ["2.2.5"], "semantic", "hard", "净高≤2.00m 处采取安全阻挡措施"),

    # ================= 拒答负例（9）=================
    ("NEG-01", "《图书馆建筑设计规范》对智慧图书馆的自助借还设备有什么技术要求？", None, [], "unanswerable", "hard", "「智慧图书馆」「自助借还」全库 0 命中"),
    ("NEG-02", "图书馆人脸识别门禁系统应该怎么设计？", None, [], "unanswerable", "hard", "「人脸识别」全库 0 命中"),
    ("NEG-03", "《民用建筑设计统一标准》第9.9.9条规定了什么？", None, [], "unanswerable", "easy", "该标准章号最大为 8，不存在第 9 章"),
    ("NEG-04", "图书馆建筑的碳排放量应该怎么计算？", None, [], "unanswerable", "medium", "「碳排放」全库 0 命中"),
    ("NEG-05", "图书馆用无人机配送图书时，通道净宽要求是多少？", None, [], "unanswerable", "hard", "「无人机」全库 0 命中"),
    ("NEG-06", "图书馆漂流书屋的设置要求有哪些？", None, [], "unanswerable", "hard", "「漂流书屋」全库 0 命中"),
    ("NEG-07", "图书馆元宇宙阅读空间的声学指标是多少？", None, [], "unanswerable", "hard", "「元宇宙」全库 0 命中"),
    ("NEG-08", "佛山市图书馆的总建筑面积是多少？", None, [], "unanswerable", "easy", "具体项目数据，规范库里不可能有"),
    ("NEG-09", "图书馆停车场要配建多少个新能源汽车充电桩？", None, [], "unanswerable", "medium", "「充电桩」全库 0 命中"),
]


def load_corpus():
    with open(CHUNKS, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    seen = collections.defaultdict(set)
    for r in rows:
        seen[r["standard"]].add(r["article_no"])
    return rows, seen


def main():
    rows, have = load_corpus()
    print(f"语料：{len(rows)} 条 / {len(have)} 本规范")

    errs, warns = [], []
    qids, queries = collections.Counter(), collections.Counter()

    for it in P:
        qid, q, std, gold, typ, diff, note = it
        qids[qid] += 1
        queries[q] += 1
        if len(q) > 250:
            errs.append(f"{qid} query 超过 250 字符（Dify hit-testing 硬限制）")
        if typ != "unanswerable":
            if not gold:
                errs.append(f"{qid} 缺金标")
            for g in gold:
                if g not in have[std]:
                    errs.append(f"{qid} 金标不存在：{std} / {g}")
        else:
            if gold:
                errs.append(f"{qid} 负例不应有金标")

    for k, v in qids.items():
        if v > 1:
            errs.append(f"qid 重复：{k}")
    for k, v in queries.items():
        if v > 1:
            warns.append(f"query 重复：{k}")

    items = [{"qid": q, "query": t, "standard": s, "gold": g,
              "type": ty, "difficulty": d, "note": n}
             for (q, t, s, g, ty, d, n) in P]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    pos = sum(1 for i in items if i["type"] != "unanswerable")
    neg = len(items) - pos
    print(f"写出：{OUT}")
    print(f"  正例 {pos} 条 / 负例 {neg} 条 / 合计 {len(items)} 条")
    print("  按规范：", dict(collections.Counter(i["standard"] for i in items if i["standard"])))
    print("  按类型：", dict(collections.Counter(i["type"] for i in items)))

    if warns:
        print("\n[提示]")
        for w in warns:
            print("  -", w)
    if errs:
        print("\n[错误]")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print("\n校验通过：全部金标在 chunks_v5 中存在。")


if __name__ == "__main__":
    main()
