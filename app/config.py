# -*- coding: utf-8 -*-
"""统一配置入口 —— 所有路径、模型名、API Key 都从这里取，别处不许硬编码。

为什么要有这个文件
------------------
1. **换机可复现**：路径只在这一个文件里出现，改一行就全部生效。
2. **防 Key 泄露**：Key 只从 ``.env`` 读，不写进代码；``.env`` 已在 ``.gitignore`` 里
   → 永远不会被 push 到 GitHub。
3. **守住中文路径陷阱**：``CHROMA_DIR`` 在这里做一次 ASCII 断言，任何脚本都绕不过去。
   （chromadb 1.5.9 在含中文的路径下**不写 HNSW 索引**且不报错，详见
   ``输出文档/scripts/build_index.py`` 文件头）
4. **脚本与 Web 同源（2026-11 整改）**：``输出文档/scripts/`` 与 ``输出文档/检索评测/``
   下的脚本一律 ``from _bcr_config import load_config``，而 ``_bcr_config`` 只是本文件的
   薄壳 —— 库目录 / 集合名 / 模型名 / 语料路径只在这里定义一次，不再各脚本各写一份。

用法
----
    from app.config import CHROMA_DIR, COLLECTION, EMBED_MODEL   # 直接取常量
    from app.config import Config, current                       # 需要快照时

    脚本里（工作目录不在项目根也没关系）：
        from _bcr_config import load_config
        cfg = load_config()

自检（打印当前生效的全部配置，Key 只显示掩码）：
    python -m app.config
"""
import os

# ---------------------------------------------------------------------------
# 项目根目录
# PROJECT_ROOT 环境变量仅用于测试/换机；不设时按本文件位置自动推算
# （本文件在 <ROOT>\app\config.py，所以往上一级就是 ROOT）
# ---------------------------------------------------------------------------
ROOT = os.path.normpath(
    os.environ.get("PROJECT_ROOT")
    or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# ---------------------------------------------------------------------------
# 读 .env（装了 python-dotenv 才生效；系统环境变量优先，不会被 .env 覆盖）
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"), override=False)
except ImportError:
    pass


def _env(key, default):
    """读环境变量：未设或为空串则用默认值。"""
    v = os.environ.get(key)
    return default if v is None or not str(v).strip() else v.strip()


# ---------------------------------------------------------------------------
# 向量库（库目录必须全 ASCII）
# ---------------------------------------------------------------------------
CHROMA_DIR = _env("CHROMA_DIR", r"D:\WorkBuddy\chroma_db")
COLLECTION = _env("CHROMA_COLLECTION", "standards")

# ---------------------------------------------------------------------------
# 语料与向量模型：必须与当初建库时**完全一致**。
# 换模型 = 换向量空间 = 旧库作废，只能重新向量化。
# ---------------------------------------------------------------------------
CHUNKS_FILE = _env("CHUNKS_FILE",
                   os.path.join(ROOT, "输出文档", "chunks_v6", "chunks.jsonl"))
EMBED_MODEL = _env("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_DIM = int(_env("EMBED_DIM", "1024"))

# 向量缓存：建库脚本写、入库脚本读。默认放在 chunks 文件同级的 cache/。
CACHE_DIR = _env("EMB_CACHE_DIR",
                 os.path.join(os.path.dirname(CHUNKS_FILE), "cache"))
NPY_NAME = "emb_qwen3-embedding-0.6b.npy"
META_NAME = "emb_meta.json"
NPY_FILE = os.path.join(CACHE_DIR, NPY_NAME)
META_FILE = os.path.join(CACHE_DIR, META_NAME)

HF_HOME = _env("HF_HOME", r"D:\WorkBuddy\models\hf")

# ---------------------------------------------------------------------------
# 生成模型（通义千问 / 阿里云百炼 · OpenAI 兼容接口）
# ⚠️ Key 与地域绑定：北京（dashscope.aliyuncs.com）的 Key 不能用于新加坡域名。
# ⚠️ 必须用**按量付费**的 Key，且用**默认业务空间**的 Key。
# ---------------------------------------------------------------------------
DASHSCOPE_API_KEY = _env("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = _env("DASHSCOPE_BASE_URL",
                          "https://dashscope.aliyuncs.com/compatible-mode/v1")
CHAT_MODEL = _env("CHAT_MODEL", "qwen-plus")
TEMPERATURE = float(_env("TEMPERATURE", "0.2"))       # 铁律要求 0 ~ 0.2

# ---------------------------------------------------------------------------
# 检索参数
# ---------------------------------------------------------------------------
TOP_K = int(_env("TOP_K", "5"))


# ---------------------------------------------------------------------------
# 守卫：库目录一旦含非 ASCII 字符，宁可直接报错退出，也不要做出一个打不开的库
# （只对 CHROMA_DIR 断言 —— 项目根目录是中文没关系，chromadb 只在乎库目录）
# ---------------------------------------------------------------------------
def assert_ascii_path(path, name="CHROMA_DIR"):
    bad = sorted({c for c in path if ord(c) > 127})
    if bad:
        raise SystemExit(
            f"\n❌ {name} 含非 ASCII 字符 {bad}：\n   {path}\n"
            "   chromadb 1.5.9 的 Rust HNSW 在中文路径下**静默不写索引文件**，\n"
            "   做出的库打不开（Error loading hnsw index）。\n"
            "   请把库放到纯 ASCII 路径，例如 D:\\WorkBuddy\\chroma_db\n"
        )
    return path


assert_ascii_path(CHROMA_DIR)


class Config(object):
    """当前生效配置的快照。

    显式传入的字段优先（测试里用它构造假配置）；不传则取本模块加载时算好的值。
    """

    def __init__(self, chroma_dir=None, collection=None, embed_model=None,
                 embed_dim=None, chunks_file=None, cache_dir=None,
                 npy_file=None, meta_file=None, hf_home=None, top_k=None):
        self.ROOT = ROOT
        self.CHROMA_DIR = CHROMA_DIR if chroma_dir is None else chroma_dir
        self.COLLECTION = COLLECTION if collection is None else collection
        self.EMBED_MODEL = EMBED_MODEL if embed_model is None else embed_model
        self.EMBED_DIM = EMBED_DIM if embed_dim is None else int(embed_dim)
        self.CHUNKS_FILE = CHUNKS_FILE if chunks_file is None else chunks_file
        self.CACHE_DIR = CACHE_DIR if cache_dir is None else cache_dir
        self.NPY_FILE = (NPY_FILE if npy_file is None
                         else npy_file)
        self.META_FILE = (META_FILE if meta_file is None
                          else meta_file)
        self.HF_HOME = HF_HOME if hf_home is None else hf_home
        self.TOP_K = TOP_K if top_k is None else int(top_k)

    def as_dict(self):
        return {"ROOT": self.ROOT, "CHROMA_DIR": self.CHROMA_DIR,
                "COLLECTION": self.COLLECTION, "EMBED_MODEL": self.EMBED_MODEL,
                "EMBED_DIM": self.EMBED_DIM, "CHUNKS_FILE": self.CHUNKS_FILE,
                "CACHE_DIR": self.CACHE_DIR, "NPY_FILE": self.NPY_FILE,
                "META_FILE": self.META_FILE, "HF_HOME": self.HF_HOME,
                "TOP_K": self.TOP_K}

    def __repr__(self):
        return (f"Config(CHROMA_DIR={self.CHROMA_DIR!r}, "
                f"COLLECTION={self.COLLECTION!r}, "
                f"EMBED_MODEL={self.EMBED_MODEL!r})")


def current():
    """返回一份当前生效配置的快照。"""
    return Config()


def mask(secret):
    """Key 掩码：只留前 3 位与后 4 位，够确认是哪把 Key，又不会泄到截图里。"""
    if not secret:
        return "(未设置)"
    if len(secret) <= 10:
        return secret[:2] + "*" * (len(secret) - 2)
    return f"{secret[:3]}…{secret[-4:]}（长度 {len(secret)}）"


def summary():
    """返回当前生效配置的清单（list of (名称, 值)），供打印/日志用。"""
    return [
        ("ROOT", ROOT),
        ("CHROMA_DIR", CHROMA_DIR),
        ("COLLECTION", COLLECTION),
        ("CHUNKS_FILE", CHUNKS_FILE),
        ("CACHE_DIR", CACHE_DIR),
        ("EMBED_MODEL", EMBED_MODEL),
        ("EMBED_DIM", EMBED_DIM),
        ("HF_HOME", HF_HOME),
        ("DASHSCOPE_BASE_URL", DASHSCOPE_BASE_URL),
        ("CHAT_MODEL", CHAT_MODEL),
        ("TEMPERATURE", TEMPERATURE),
        ("TOP_K", TOP_K),
        ("DASHSCOPE_API_KEY", mask(DASHSCOPE_API_KEY)),
    ]


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")       # Windows 控制台默认 GBK
    except Exception:
        pass
    print("当前生效配置：")
    for k, v in summary():
        print(f"  {k:<20} {v}")
    for p in (CHROMA_DIR, CHUNKS_FILE):
        print(f"  {'存在' if os.path.exists(p) else '不存在':<18} {p}")
