# -*- coding: utf-8 -*-
"""force_rules.py —— 条文级强制状态规则（唯一事实来源）。

为什么需要这张表
----------------
此前 split_articles_v2.py 把「整本标准」的 force 原样写给每一个 chunk，
于是 GB 50352 的 303 条全标成「强条已废止」、JGJ 38 的 222 条全标成
「部分强条」。强制状态是**条文级**属性，必须按条文号判定，否则
「只看现行强制条文」这个核心过滤功能不可信。

两种模式
--------
mode="all"   全文强制性工程建设规范：每一条都是 status。
mode="list"  只有公告/前言明确列出的强条取 status，其余取 default。
             清单必须逐条可查，禁止把整本标准的状态套到每条上。

依据来源（逐条可核）
--------------------
- GB 55031-2022 / GB 55037-2022 / GB 55019-2021：
  公告明示「本规范为强制性工程建设规范，全部条文必须严格执行」。
- JGJ 38-2015：住房和城乡建设部公告第 907 号（规范正文第 3 页）：
  「其中，第 6.1.2、6.1.3、6.2.1、6.2.2 条为强制性条文，必须严格执行」；
  前言另有「以黑体字标志的条文为强制性条文」。
- GB 50352-2019：住房和城乡建设部公告 2019 年第 57 号：
  「其中，第 4.3.1、6.7.4、6.8.6、6.8.9 条为强制性条文」；
  GB 55031-2022 公告「同时废止下列现行工程建设标准相关强制性条文」
  逐项列出上述 4 条，故这 4 条标「原强条已废止」，其余条文为「非强条」。

维护约定
--------
新增规范必须在本表登记；validate_chunks.py 把「未登记标准」判为门禁失败，
不允许悄悄落到「非强条」。正文没有明确依据的条文一律保守标 default，
并在 UNCERTAIN 中登记待核事项。
"""
from __future__ import annotations

RULES_VERSION = 1

# 未登记标准的返回值：显眼、可被门禁捕获，不能被当成「非强条」
UNKNOWN = "未定义规则"

# 强制过滤（force_only）认可的两种状态
FORCE_OK = ("全文强制", "部分强条")


STANDARDS = {
    "GB 55031-2022": {
        "name": "民用建筑通用规范",
        "mode": "all",
        "status": "全文强制",
        "default": "全文强制",
        "articles": (),
        "basis": "公告：本规范为强制性工程建设规范，全部条文必须严格执行",
        "source": "GB 55031-2022 公告；规范前言「关于规范实施」"
                  "（ocr_v4 第 2 页，第 48-53 行）",
    },
    "GB 55037-2022": {
        "name": "建筑防火通用规范",
        "mode": "all",
        "status": "全文强制",
        "default": "全文强制",
        "articles": (),
        "basis": "公告：本规范为强制性工程建设规范，全部条文必须严格执行",
        "source": "GB 55037-2022 公告；规范前言「关于规范实施」"
                  "（ocr_v4 第 2 页，第 48-53 行）",
    },
    "GB 55019-2021": {
        "name": "建筑与市政工程无障碍通用规范",
        "mode": "all",
        "status": "全文强制",
        "default": "全文强制",
        "articles": (),
        "basis": "公告：本规范为强制性工程建设规范，全部条文必须严格执行",
        "source": "GB 55019-2021 公告；规范前言「关于规范实施」"
                  "（ocr_v4 第 2 页，第 49-52 行）",
    },
    "JGJ 38-2015": {
        "name": "图书馆建筑设计规范",
        "mode": "list",
        "status": "部分强条",
        "default": "非强条",
        # 公告第 907 号逐条列出的 4 条，不许多也不许少
        "articles": ("6.1.2", "6.1.3", "6.2.1", "6.2.2"),
        "basis": "住建部公告第 907 号：第 6.1.2、6.1.3、6.2.1、6.2.2 条"
                 "为强制性条文，必须严格执行",
        "source": "ocr_v4/JGJ 38-2015 图书馆建筑设计规范.md 第 3 页（公告）、"
                  "第 4 页（前言「黑体字标志的条文为强制性条文」）",
    },
    "GB 50352-2019": {
        "name": "民用建筑设计统一标准",
        "mode": "list",
        "status": "原强条已废止",
        "default": "非强条",
        "articles": ("4.3.1", "6.7.4", "6.8.6", "6.8.9"),
        "basis": "住建部公告 2019 年第 57 号列上述 4 条为强制性条文；"
                 "GB 55031-2022 公告「同时废止下列现行工程建设标准相关"
                 "强制性条文」逐项废止这 4 条",
        "source": "住建部 2019 年第 57 号公告（mohurd.gov.cn/gongkai/zc/wjk/"
                  "art/2024/art_17339_240715.html）；GB 55031-2022 公告",
    },
}


# 需要业务侧复核、但本次不改口径的条目。写进校验报告，不隐藏。
UNCERTAIN = (
    {
        "standard": "JGJ 38-2015",
        "articles": ("6.1.2", "6.1.3", "6.2.1", "6.2.2"),
        "question": "GB 55031-2022 前言有「强制性工程建设规范实施后，现行…"
                    "行业标准中的强制性条文同时废止」的总括表述，但其公告的"
                    "逐项废止清单并未列出 JGJ 38。本次按公告清单保留"
                    "「部分强条」，现行效力建议业务侧复核。",
        "decision": "部分强条",
    },
)


def rule_of(standard):
    """返回某本标准的规则；未登记返回 None。"""
    return STANDARDS.get(standard)


def listed_articles(standard):
    """清单模式标准的强条清单（保持表内顺序）。"""
    rule = STANDARDS.get(standard)
    if not rule:
        return ()
    return tuple(rule["articles"])


def normalize_article_no(article_no):
    """`6.1.2-表` -> `6.1.2`；其余原样返回。"""
    no = (article_no or "").strip()
    if no.endswith("-表"):
        no = no[:-2]
    return no


def resolve(standard, article_no):
    """标准号 + 条文号 -> 强制状态。未登记标准返回 UNKNOWN。"""
    rule = STANDARDS.get(standard)
    if rule is None:
        return UNKNOWN
    if rule["mode"] == "all":
        return rule["status"]
    if normalize_article_no(article_no) in rule["articles"]:
        return rule["status"]
    return rule["default"]


def uncertain_for(standard=None):
    """待核事项；不传 standard 时返回全部。"""
    if standard is None:
        return list(UNCERTAIN)
    return [u for u in UNCERTAIN if u["standard"] == standard]


def _self_check():
    """规则表自检：清单可解析、无重复、依据齐全。"""
    import re

    errors = []
    for std, rule in STANDARDS.items():
        for key in ("name", "mode", "status", "basis", "source"):
            if not rule.get(key):
                errors.append(f"{std}: 缺少 {key}")
        if rule["mode"] not in ("all", "list"):
            errors.append(f"{std}: 未知 mode={rule['mode']}")
        if rule["mode"] == "all":
            if rule["default"] != rule["status"]:
                errors.append(f"{std}: 全文强制标准的 default 必须等于 status")
            continue
        arts = rule["articles"]
        if len(arts) != len(set(arts)):
            errors.append(f"{std}: 清单有重复条文号")
        for no in arts:
            if not re.match(r"^(?:[A-Z]\.)?\d{1,2}\.\d{1,2}\.\d{1,2}$", no):
                errors.append(f"{std}: 条文号格式异常 {no!r}")
        if resolve(std, arts[0]) != rule["status"]:
            errors.append(f"{std}: 清单首条未解析为 {rule['status']}")
    if errors:
        raise SystemExit("force_rules 自检失败：\n  " + "\n  ".join(errors))

    print(f"force_rules v{RULES_VERSION}：{len(STANDARDS)} 本规范，"
          f"{len(UNCERTAIN)} 条待核事项")
    for std, rule in STANDARDS.items():
        if rule["mode"] == "all":
            desc = f"全部条文 → {rule['status']}"
        else:
            desc = (f"{len(rule['articles'])} 条 → {rule['status']}，"
                    f"其余 → {rule['default']}（{','.join(rule['articles'])}）")
        print(f"  {std:<16} {desc}")


if __name__ == "__main__":
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _self_check()
