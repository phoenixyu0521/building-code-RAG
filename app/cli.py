# -*- coding: utf-8 -*-
"""命令行入口 —— 本项目「可演示」的最小形态。

在**项目根目录**下运行（因为要用 `-m` 把 app 当包导入）：
    cd "D:\\WorkBuddy\\项目"
    python -m app.cli "图书馆阅览室到最近疏散门的最大距离是多少"
    python -m app.cli --no-llm "半地下室的定义"          # 只检索，不花钱
    python -m app.cli --force-only "防火分区的最大面积"   # 只看强制条文
    python -m app.cli --standards                        # 看看库里有哪些规范
    python -m app.cli                                    # 不带问题 = 交互模式

（也可以 `python app\\cli.py`，文件顶部有一段引导代码会自己修好导入路径。）
"""
import argparse
import collections
import os
import sys

# 直接 `python app\cli.py` 时，把项目根目录塞进 sys.path，让 `import app.xxx` 能找到
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config                                    # noqa: E402
from app.pipeline import ask, format_result               # noqa: E402

# 常用的元数据过滤预设（Chroma 的 where 语法）
PRESETS = {
    "force-only": ("只看强制条文", {"force_status": {"$in": ["全文强制", "部分强条"]}}),
    "exclude-repealed": ("排除已废止强条", {"force_status": {"$ne": "强条已废止(由GB55031替代)"}}),
    "article-only": ("只要正文，不要表格/节标题", {"kind": "article"}),
    "current": ("只看现行", {"status": {"$in": ["现行", "现行(技术参考)"]}}),
}


def combine(*conds):
    """把多个过滤条件合成一个。

    ⚠️ Chroma 有个反直觉的规则：**多个字段必须显式写 $and**。
       直接写 {"a": 1, "b": 2} 会被当成「同一个字段的复合条件」而报错。
    """
    cs = [c for c in conds if c]
    if not cs:
        return None
    if len(cs) == 1:
        return cs[0]
    return {"$and": cs}


def build_where(args):
    conds = []
    if args.standard:
        conds.append({"standard": args.standard})
    for name in (args.preset or []):
        conds.append(PRESETS[name][1])
    if args.kind:
        conds.append({"kind": args.kind})
    return combine(*conds)


def show_standards():
    """列出库里的规范与条数 —— 用来确认「库里有哪几本」。"""
    from app.retriever import Retriever
    r = Retriever()
    col = r._get_col()
    got = col.get(include=["metadatas"])
    cnt = collections.Counter(
        (m.get("standard"), m.get("standard_name"), m.get("force_status"))
        for m in got["metadatas"]
    )
    print(f"库内共 {col.count()} 条，覆盖 {len({k[0] for k in cnt})} 本规范：\n")
    print(f"  {'规范号':<16}{'名称':<22}{'强制状态':<24}条数")
    print("  " + "-" * 70)
    for (std, name, fs), n in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"  {std or '?':<16}{name or '?':<22}{fs or '?':<24}{n}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")           # Windows 控制台默认 GBK
    except Exception:
        pass

    p = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="建筑规范问答（本地向量库 + 通义千问）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="提示：条文号必须能核对 —— 答案下面会列出全部检索来源。",
    )
    p.add_argument("question", nargs="*", help="问题（不带则进入交互模式）")
    p.add_argument("--top-k", type=int, default=None, help="检索条数（默认取配置 TOP_K）")
    p.add_argument("--no-llm", action="store_true", help="只检索、不调用大模型（不花钱）")
    p.add_argument("--standard", default=None,
                   help='只查某一本规范，如 "GB 50352-2019"')
    p.add_argument("--kind", default=None,
                   help="只查某类片段：article / section / table / appendix-table")
    p.add_argument("--preset", action="append", choices=sorted(PRESETS),
                   help="过滤预设，可叠加：" +
                        "；".join(f"{k}={v[0]}" for k, v in PRESETS.items()))
    p.add_argument("--standards", action="store_true", help="列出库里的规范与条数后退出")
    p.add_argument("--show-config", action="store_true", help="打印当前配置后退出")
    args = p.parse_args()

    if args.show_config:
        for k, v in config.summary():
            print(f"  {k:<20} {v}")
        return 0

    if args.standards:
        show_standards()
        return 0

    where = build_where(args)
    if where:
        print(f"（元数据过滤：{where}）\n")

    if args.question:
        q = " ".join(args.question)
        print(format_result(ask(q, top_k=args.top_k, where=where,
                                allow_llm=not args.no_llm)))
        return 0

    # 交互模式
    print("交互模式。直接回车退出。输入 :standards 看规范清单。\n")
    while True:
        try:
            q = input("❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        if q == ":standards":
            show_standards()
            continue
        try:
            print(format_result(ask(q, top_k=args.top_k, where=where,
                                   allow_llm=not args.no_llm)))
        except Exception as e:                              # 单条出错不中断会话
            print(f"❌ {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
