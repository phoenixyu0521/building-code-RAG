# -*- coding: utf-8 -*-
"""检索层 —— 输入「问题」，输出「条文片段」。这一层**不碰任何大模型**。

为什么单独一层
--------------
1. 检索错了，后面提示词写得再好也救不回来。分层之后可以单独量它（就是 86 条评测集做的事）。
2. 不依赖 API Key / 不花钱，跑得快，调试时先把它调对。
3. 以后换生成模型（比如换 Ollama 离线），这一层一行都不用改。

自检（只检索，不调大模型，不需要 API Key）：
    python -m app.retriever
    python -m app.retriever "半地下室的定义是什么"
"""
import sys

from app.config import CHROMA_DIR, COLLECTION, EMBED_MODEL, TOP_K
# ↑ 必须放在 chromadb / sentence_transformers 之前：
#   import app.config 会先执行 app/__init__.py，把 HF_HOME 设好
import chromadb                                          # noqa: E402
from sentence_transformers import SentenceTransformer     # noqa: E402


class Retriever:
    """Chroma 检索的薄封装。

    模型与集合用**类级缓存**，同一个进程里只加载一次 —— 载模型要十几秒到几分钟，
    每问一句就重载一次是不可接受的。
    """

    _model = None
    _col = None

    def __init__(self, top_k=TOP_K):
        self.top_k = top_k

    # ---------------- 懒加载 ----------------
    @classmethod
    def _get_model(cls):
        if cls._model is None:
            print(f"[retriever] 载入向量模型 {EMBED_MODEL} …（首次较慢）",
                  file=sys.stderr)
            cls._model = SentenceTransformer(EMBED_MODEL)
        return cls._model

    @classmethod
    def _get_col(cls):
        if cls._col is None:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            cls._col = client.get_collection(COLLECTION)
        return cls._col

    # ---------------- 主体 ----------------
    def search(self, question, top_k=None, where=None, min_score=None):
        """检索。

        参数
        ----
        question  : 自然语言问题
        top_k     : 返回条数，默认取配置里的 TOP_K
        where     : Chroma 元数据过滤，例如
                    只看现行强制条文  {"force_status": {"$in": ["全文强制", "部分强条"]}}
                    排除已废止强条    {"force_status": {"$ne": "强条已废止(由GB55031替代)"}}
                    只查某一本规范    {"standard": "GB 50352-2019"}
                    只要正文不要表格  {"kind": "article"}
        min_score : 相似度下限（0~1）。默认 None 不过滤 ——
                    ⚠️ 实测在本项目数据上**不存在能同时压住负例、留住正例的阈值**
                    （负例最高 0.727 > 正例 5% 分位 0.675），所以拒答靠提示词，不靠这里。

        返回
        ----
        [{"text": 正文, "meta": 元数据 dict, "score": 相似度}, ...]
        """
        k = int(top_k or self.top_k)
        col = self._get_col()

        # ⚠️ 查询侧必须写 prompt_name="query"（模型自带的检索指令）。
        #    漏掉不会报错，只会让效果悄悄变差 —— 本项目最隐蔽的坑。
        #    （建库/文档侧**不能**加这个参数，所以两边本来就不对称。）
        vec = self._get_model().encode(
            [question], prompt_name="query", normalize_embeddings=True
        ).tolist()

        res = col.query(
            query_embeddings=vec,
            n_results=k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(res["documents"][0],
                                   res["metadatas"][0],
                                   res["distances"][0]):
            score = 1.0 - float(dist)          # hnsw:space=cosine → 距离转相似度
            if min_score is not None and score < min_score:
                continue
            hits.append({"text": doc, "meta": meta, "score": score})
        return hits

    # ---------------- 展示 ----------------
    def count(self):
        return self._get_col().count()

    # ---------------- 给 Web 层用的公开入口 ----------------
    # 说明：api.py 需要「预热」和「取全部元数据」两件事。与其让它去碰 _get_model /
    # _get_col 这些下划线成员，不如在这里开两个有名有姓的公开方法 —— 职责归属清楚。
    @classmethod
    def warmup(cls):
        """预热：提前把向量模型与集合加载好，返回库内条数。

        Web 服务启动时调用。不预热的话，第一次提问要干等十几秒载模型，
        前端看起来就像卡死了。
        """
        cls._get_model()
        return cls._get_col().count()

    @classmethod
    def metadatas(cls):
        """取出全部元数据（只读，**不加载向量模型**）。

        给前端做「按规范筛选」的下拉用 —— 规范清单应该从库里现取，
        而不是在代码里写死，否则以后加一本规范就得改代码。
        """
        return cls._get_col().get(include=["metadatas"])["metadatas"]


def format_hits(hits):
    """把检索结果打印成人看的样式。"""
    lines = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        lines.append(f"[{i}] 相似度 {h['score']:.4f}")
        lines.append(f"    《{m.get('standard_name', '?')}》{m.get('standard', '?')}"
                     f"  第 {m.get('article_no', '?')} 条"
                     f"  ［{m.get('status', '')}；{m.get('force_status', '')}］")
        text = h["text"].strip().replace("\n", " ")
        lines.append(f"    {text[:200]}{'…' if len(text) > 200 else ''}")
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")           # Windows 控制台默认 GBK
    except Exception:
        pass

    questions = sys.argv[1:] or [
        "图书馆阅览室到最近疏散门的最大距离是多少",
        "半地下室的定义是什么",
    ]
    print(f"库：{CHROMA_DIR}  集合：{COLLECTION}")
    r = Retriever()
    print(f"库内 {r.count()} 条\n")
    for q in questions:
        print("=" * 66)
        print(f"❓ {q}")
        print("=" * 66)
        print(format_hits(r.search(q)))
        print()
