# -*- coding: utf-8 -*-
"""Web 接口层 —— 把 app/pipeline.py 的能力暴露成 HTTP 接口。

为什么单独一层
--------------
命令行只有自己会用；**接口是给别人用的**。接口定下来之后，前端可以随便换
（今天是单页 HTML，明天想换 React / 小程序），后端一行都不用改。

接口一览
--------
GET  /api/health     服务是否活着 + 库内条数
GET  /api/standards  库内规范清单（供前端做筛选，不写死）
POST /api/ask        问一个问题 → 答案 + 引用条文

起服务
------
    python -m uvicorn app.api:app --port 8000
然后浏览器打开 http://127.0.0.1:8000

自检（只打印路由，不起服务）：
    python -m app.api
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

# ⚠️ 顺序要紧：import app.config 会先执行 app/__init__.py（把 HF_HOME 设好），
#    必须发生在任何会用向量模型的东西之前。
from app import config                                        # noqa: F401
from app import pipeline
from app.prompts import NO_HIT_REPLY
from app.retriever import Retriever

import numpy as np                                            # noqa: E402
from fastapi import FastAPI, HTTPException                    # noqa: E402
from fastapi.responses import JSONResponse                    # noqa: E402
from fastapi.staticfiles import StaticFiles                   # noqa: E402
from pydantic import BaseModel, Field                         # noqa: E402

WEB_DIR = os.path.join(config.ROOT, "web")

# 「只看现行强制条文」= 排除那 303 条「强条已废止(由GB55031替代)」
# 取值来自实库（1083 条）：全文强制 558 / 部分强条 222 / 已废止 303
FORCE_OK = ["全文强制", "部分强条"]


# ---------------------------------------------------------------------------
# 启动钩子：预热模型
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app):
    # 🔑 预热是必须的。向量模型加载要十几秒，若等第一次提问才加载，
    #    前端会以为服务卡死（而且这段时间 uvicorn 不响应任何请求）。
    try:
        n = Retriever.warmup()
        print(f"[api] 预热完成：库内 {n} 条", file=sys.stderr)
    except Exception as e:                                    # noqa: BLE001
        # 预热失败不直接退出 —— 让 /api/health 把错误报给前端，比闷死在启动日志里好排查
        print(f"[api] 预热失败：{type(e).__name__}: {e}", file=sys.stderr)
    yield


app = FastAPI(title="建筑规范智能问答", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500,
                          description="自然语言问题")
    top_k: int = Field(0, ge=0, le=20, description="返回条数；0 = 用配置默认值")
    force_only: bool = Field(False, description="只看现行强制条文")
    standards: list[str] = Field(default_factory=list,
                                 description="限定规范号，如 ['GB 50352-2019']；空 = 全部")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _clean(obj):
    """把 numpy 标量递归转成 python 原生类型。

    🔑 不转的话 JSON 序列化会失败 → 接口 500，而且报错信息很难看懂。
    """
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):        # np.floating / np.integer / np.bool_ …
        return obj.item()
    return obj


def _build_where(force_only, standards):
    """拼 Chroma 的元数据过滤条件。

    ⚠️ 多个字段必须**显式**用 $and 包起来 —— 并排写两个 key 不报错但行为不对。
    """
    conds = []
    if force_only:
        conds.append({"force_status": {"$in": FORCE_OK}})
    if standards:
        conds.append({"standard": {"$in": list(standards)}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _hit_to_dict(h):
    """把一条检索结果整理成前端好用的扁平结构。"""
    m = h["meta"]
    return {
        "article_no": str(m.get("article_no", "")),
        "standard": str(m.get("standard", "")),
        "standard_name": str(m.get("standard_name", "")),
        "version": str(m.get("version", "")),
        "status": str(m.get("status", "")),
        "force_status": str(m.get("force_status", "")),
        "kind": str(m.get("kind", "")),
        "chapter": str(m.get("chapter", "")),
        "page": str(m.get("page_start", "")),
        "score": round(float(h["score"]), 4),
        "text": h["text"],
    }


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    """服务是否活着 + 库内条数 + 当前模型。前端右上角那个状态点就是它。"""
    try:
        n = Retriever().count()
    except Exception as e:                                    # noqa: BLE001
        return JSONResponse(status_code=503,
                            content={"ok": False,
                                     "error": f"{type(e).__name__}: {e}"})
    return {
        "ok": True,
        "count": n,
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
        # 前端据此提示「未配置 Key」—— Key 本身永远不返回
        "llm_enabled": bool(config.DASHSCOPE_API_KEY),
    }


@app.get("/api/standards")
def list_standards():
    """库内规范清单（规范号 / 名称 / 版本 / 条数）。"""
    tally = {}
    for m in Retriever.metadatas():
        k = (m.get("standard") or "", m.get("standard_name") or "",
             m.get("version") or "")
        tally[k] = tally.get(k, 0) + 1
    items = [{"standard": s, "standard_name": nm, "version": v, "count": c}
             for (s, nm, v), c in sorted(tally.items())]
    return {"ok": True, "items": items, "total": sum(tally.values())}


@app.post("/api/ask")
def ask_question(req: AskRequest):
    """提问 → 答案 + 引用条文。"""
    # ⚠️ 这里刻意写 def 而不是 async def：
    #    pipeline.ask() 是同步阻塞的（等大模型要几秒），FastAPI 会把普通 def 端点
    #    丢进线程池执行；写成 async def 反而会堵死事件循环，第二个人进来就得排队。
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    where = _build_where(req.force_only, req.standards)
    top_k = req.top_k or config.TOP_K

    try:
        r = pipeline.ask(question, top_k=top_k, where=where)
    except Exception as e:                                    # noqa: BLE001
        raise HTTPException(status_code=500,
                            detail=f"{type(e).__name__}: {e}") from e

    hits = [_hit_to_dict(h) for h in r["hits"]]

    # 两种拒答：① 检索 0 条（pipeline 短路，没调大模型）；② 大模型按提示词第 4 条拒答
    refused = len(hits) == 0 or r["answer"].strip() == NO_HIT_REPLY.strip()

    return _clean({
        "question": r["question"],
        "answer": r["answer"],
        "used_llm": r["used_llm"],
        "refused": refused,
        "top_k": top_k,
        "force_only": req.force_only,
        "standards": req.standards,
        "hits": hits,
    })


# ---------------------------------------------------------------------------
# 托管前端（必须放在所有 /api 路由**之后**注册，否则 "/" 会把它们全吃掉）
# ---------------------------------------------------------------------------
if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
else:
    print(f"[api] 未找到前端目录 {WEB_DIR}，本次只提供 /api/* 接口", file=sys.stderr)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")              # Windows 控制台默认 GBK
    except Exception:                                         # noqa: BLE001
        pass
    print("已注册的路由：")
    for r in app.routes:
        path = getattr(r, "path", "")
        if path.startswith("/api"):
            print(f"  {sorted(getattr(r, 'methods', []) or [])}  {path}")
    print(f"\n前端目录：{WEB_DIR}  "
          f"{'存在' if os.path.isdir(WEB_DIR) else '不存在'}")
    print("\n起服务（不要把 --reload 用在这里，改代码会重载模型，要等十几秒）：")
    print("  python -m uvicorn app.api:app --port 8000")
    print("  然后浏览器打开 http://127.0.0.1:8000")
