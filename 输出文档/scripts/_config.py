# -*- coding: utf-8 -*-
"""脚本侧配置薄壳 —— 让 ``输出文档/scripts/`` 与 ``输出文档/检索评测/`` 也能用 app/config.py。

为什么要有这一层
----------------
脚本是按「双击 / 在任意目录下 ``python build_index.py``」的方式跑的，工作目录不一定
是项目根。这里只做两件事：

1. 把项目根（含 ``app/config.py`` 的那一级）加进 ``sys.path``：
   从当前文件往上逐级找 ``app/config.py``；找不到再用环境变量 ``BCR_PROJECT_ROOT``。
2. 重新导出 ``app.config`` 里的常量与 ``load_config()``。

脚本里只准这样取配置：
    from _config import load_config
    cfg = load_config()
    cfg.CHROMA_DIR / cfg.COLLECTION / cfg.EMBED_MODEL / cfg.CHUNKS_FILE …

真正的默认值与 ``.env`` 读取都在 ``app/config.py``，这里不写任何业务默认值。

为什么文件名不以 ``_`` 开头
--------------------------
``.gitignore`` 里有 ``输出文档/scripts/_*``（一次性探针脚本不入库），
``_bcr_config.py`` 会被一并忽略 —— 提交不进去，合并后直接 ImportError。
所以本文件叫 ``_config.py``（含下划线但**不以下划线开头**）。新增脚本请避开
``_`` 开头。
"""
import importlib
import os
import sys

_MODULE = None


def _candidate_roots():
    here = os.path.dirname(os.path.abspath(__file__))
    seen = set()
    d = here
    for _ in range(8):                       # 逐级往上：scripts → 输出文档 → 项目根
        if d and d not in seen:
            seen.add(d)
            yield d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    env = os.environ.get("BCR_PROJECT_ROOT")
    if env and env not in seen:
        yield os.path.normpath(env)


def project_root():
    """项目根（含 app/config.py 的那一级）。"""
    for d in _candidate_roots():
        if os.path.isfile(os.path.join(d, "app", "config.py")):
            return d
    raise SystemExit(
        "找不到项目根（需要含 app/config.py 的目录）。\n"
        "  已尝试：" + "；".join(_candidate_roots()) + "\n"
        "  请用环境变量 BCR_PROJECT_ROOT 指定项目根，例如：\n"
        "    $env:BCR_PROJECT_ROOT = 'D:\\WorkBuddy\\项目'")


def _load():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    root = project_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    _MODULE = importlib.import_module("app.config")
    return _MODULE


def load_config():
    """当前生效配置（app.config.Config 快照）。"""
    return _load().current()


def __getattr__(name):
    """``from _config import CHROMA_DIR`` 之类也能用。"""
    return getattr(_load(), name)
