# -*- coding: utf-8 -*-
"""用 RAGAS 评「检索层」：context_recall（≈召回率）+ context_precision（≈准确率）。

为什么能直接用、不用重新标注
  · RAGAS 的这两个指标不需要「生成的答案」，只要 检索结果 + 标准答案 就能算 → 纯检索评测
  · 标准答案（reference）直接用 gold 条文号的规范原文
    （已实测：评测集 86 条的金标 100% 能在 chunks_v5 里取到原文，零人工）

两阶段（两个 venv，互不污染）
  ① export  在 chroma venv 跑：评测集 + 本地 Chroma → ragas_input.jsonl（零成本，不调 LLM）
  ② score   在 ragas venv 跑：ragas_input.jsonl → 判官 qwen-plus → 结果_ragas_<tag>.csv

为什么分两个 venv
  ragas 会拉 langchain-core / datasets / pydantic，有覆盖 chromadb 依赖的风险；
  而 score 阶段完全不需要 torch 和 embedding 模型（本脚本用的两个 RAGAS 指标只用 LLM 判官）。

用法
  $PY_CHROMA = C:\\Users\\A\\.workbuddy\\binaries\\python\\envs\\chroma\\Scripts\\python.exe
  $PY_RAGAS  = C:\\Users\\A\\.workbuddy\\binaries\\python\\envs\\ragas\\Scripts\\python.exe

  # ① 导出：先 dry 看样例，确认链路；再去掉 --dry 跑全量
  & $PY_CHROMA ragas_eval.py --stage export --dry --limit 2
  & $PY_CHROMA ragas_eval.py --stage export --top-k 5

  # ② 建 RAGAS 专用 venv（一次性；装软件请你自己执行）
  & C:\\Users\\A\\.workbuddy\\binaries\\python\\versions\\3.13.12\\python.exe -m venv C:\\Users\\A\\.workbuddy\\binaries\\python\\envs\\ragas
  & $PY_RAGAS -m pip install -U ragas datasets openai -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn

  # ③ 体检（装完包先跑这个，只花 1 条的钱，验证判官链路通不通）
  & $PY_RAGAS ragas_eval.py --stage doctor

  # ④ 打分（会调 qwen-plus，产生少量费用；已算过的会走缓存）
  & $PY_RAGAS ragas_eval.py --stage score --tag k5

已核实（对齐 ragas 0.4.3 源码，2026-09-20）
  · 指标：from ragas.metrics.collections import ContextRecall, ContextPrecision
      构造 ContextRecall(llm=llm) / ContextPrecision(llm=llm)
      ascore(user_input=, retrieved_contexts=List[str], reference=) → MetricResult.value
  · 判官：ragas.llms.llm_factory(model, provider="openai", client=AsyncOpenAI(...), **kwargs)
  · 一条 query 的 LLM 调用数约 5~15 次（context_recall 逐句归因、context_precision 逐条 context 判定），
    2026-09-20 实测 77 条 ≈ 千次量级调用，qwen-plus 约几毛钱

坑（已核实）
  · AsyncOpenAI 的连接池绑定 event loop → 必须在同一个 loop 跑完全部；
    逐条 asyncio.run() 会在第 2 条起报 `Event loop is closed`（本脚本已用 _score_loop 修掉）
  · max_tokens 默认只有 1024，context_recall 的「逐句 + reason」JSON 会被截断 → 静默变 NaN；
    本脚本已显式提到 4096
  · 判官 temperature 必须 0；分数有噪声，看趋势不看绝对值
  · 判官返回解析失败会静默变 NaN，不是报错 → 报告会打印 NaN 条数
  · contexts 必须是 List[str]，传 Document 对象会 ValueError
  · ragas 0.4.3 起 evaluate() 与 ragas.metrics 旧导入均已废弃，官方推 ragas.metrics.collections
    → 本脚本两条路都试，自动选通的那条（报告里会打印实际用的 API）
  · 换判官模型 = 历史分数全部不可比，必须重新基线
  · 9 条拒答负例不参与：RAGAS 需要标准答案，负例没有答案可对照；拒答仍用 run_eval.py 判定
"""
import argparse
import asyncio
import csv as _csv
import inspect
import json
import os
import sys
import time
import types

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(BASE))
EVAL_PATH = os.path.join(BASE, "评测集.jsonl")
CHUNKS = os.path.join(PROJ, r"输出文档\chunks_v5\chunks.jsonl")
ENV_PATH = os.path.join(PROJ, ".env")
IN_JSONL = os.path.join(BASE, "ragas_input.jsonl")
IN_DRY = os.path.join(BASE, "ragas_input_dry.jsonl")
CACHE = os.path.join(BASE, "ragas_cache.jsonl")

CHROMA_DIR = r"D:\WorkBuddy\chroma_db"      # 必须纯 ASCII，见 build_index.py
COLL = "standards"
EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"
JUDGE = "qwen-plus"


def load_env():
    """读项目根 .env（已有 DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL）。"""
    if not os.path.exists(ENV_PATH):
        return
    for line in open(ENV_PATH, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def install_langchain_shims():
    """★ 修 ragas 0.4.3 与 langchain-community>=0.4 的硬冲突（必须在 import ragas 之前调用）。

    实测根因（2026-09-21）：
      ragas/llms/base.py:12  `from langchain_community.chat_models.vertexai import ChatVertexAI`
      · 该模块自 langchain-community 0.4 起已被**移除**（社区包已官宣 sunset）
      · 而 ragas 对 langchain 系的依赖是「裸写无上界」（`Requires-Dist: langchain-community`）
        → pip 必然拉最新版 → 必然踩这个坑，且 ragas 已停更、上游不会修
      · 症状：`import ragas` 直接抛 ModuleNotFoundError，连 doctor 都进不去
      · 该名字只出现在 base.py:43 的 MULTIPLE_COMPLETION_SUPPORTED 列表里做 isinstance，
        **不会被实例化** → 塞一个空类即可，零副作用
      · 注意 `langchain_community/llms/vertexai.py` **仍存在** → base.py:13 的 VertexAI 无需处理

    做法：只往 sys.modules 注入占位模块，**不改 site-packages、不降级任何包**。
    若你的 langchain-community 是 0.3.x（模块仍在），此处自动 no-op。
    """
    TARGET = "langchain_community.chat_models.vertexai"
    if TARGET in sys.modules:
        return False
    try:
        __import__(TARGET)
        return False
    except ModuleNotFoundError:
        pass
    for pkg in ("langchain_community", "langchain_community.chat_models"):
        try:
            __import__(pkg)
        except Exception:
            return False
    mod = types.ModuleType(TARGET)
    mod.ChatVertexAI = type("ChatVertexAI", (), {
        "__doc__": "ragas 占位桩：真模块已从 langchain-community 移除，仅用于 isinstance 白名单"})
    mod.__wb_shim__ = True
    sys.modules[TARGET] = mod
    try:
        setattr(sys.modules["langchain_community.chat_models"], "vertexai", mod)
    except Exception:
        pass
    return True


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


# ============================== 阶段 ① export ==============================
def load_corpus_text():
    """(standard, article_no) -> 原文。把 gold 条文号翻成标准答案文本。"""
    idx = {}
    for r in read_jsonl(CHUNKS):
        idx[(r["standard"], r["article_no"])] = r["text"]
    return idx


def make_reference(it, corpus):
    """标准答案 = gold 条文原文拼接。表 chunk 号带 `-表` 后缀，去掉再试一次。"""
    parts = []
    for g in it.get("gold") or []:
        for key in (g, g.replace("-表", "")):
            t = corpus.get((it["standard"], key))
            if t:
                parts.append(t)
                break
    if not parts:                            # 兜底：评测集自带的答案要点
        parts = [it.get("note") or ""]
    return "\n".join(p for p in parts if p).strip()


def stage_export(a):
    os.environ.setdefault("HF_HOME", r"D:\WorkBuddy\models\hf")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    import chromadb
    from sentence_transformers import SentenceTransformer

    corpus = load_corpus_text()
    all_items = read_jsonl(EVAL_PATH)
    items = [i for i in all_items if (i.get("gold") or [])]
    if a.only_standard:
        items = [i for i in items if i["standard"] == a.only_standard]
    if a.limit:
        items = items[: a.limit]
    print(f"评测集 {len(all_items)} 条 → 取正例 {len(items)} 条"
          f"（{len(all_items) - len([i for i in all_items if i.get('gold')])} 条负例无标准答案，不喂 RAGAS）")

    print("载入本地 embedding 模型…")
    t = time.time()
    model = SentenceTransformer(EMB_MODEL)
    print(f"  就绪 {time.time() - t:.1f}s")
    col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLL)

    rows = []
    for n, it in enumerate(items, 1):
        try:
            emb = model.encode([it["query"]], prompt_name="query",
                               normalize_embeddings=True)[0]
        except Exception:
            emb = model.encode([it["query"]], normalize_embeddings=True)[0]
        res = col.query(query_embeddings=[emb.tolist()], n_results=a.top_k,
                        include=["documents"])
        ctx = [d for d in (res["documents"][0] or []) if d]
        rows.append({"qid": it["qid"], "question": it["query"],
                     "reference": make_reference(it, corpus), "contexts": ctx,
                     "gold": it.get("gold"), "type": it.get("type"),
                     "difficulty": it.get("difficulty"),
                     "standard": it.get("standard")})
        sys.stdout.write(f"\r  检索 {n}/{len(items)}")
        sys.stdout.flush()
    print()

    out = IN_DRY if a.dry else IN_JSONL
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    empty_ref = [r["qid"] for r in rows if not r["reference"]]
    empty_ctx = [r["qid"] for r in rows if not r["contexts"]]
    print(f"→ {out}")
    if empty_ref:
        print(f"  ⚠ 标准答案为空 {len(empty_ref)} 条（会导致 context_recall 恒为 0）：{empty_ref[:5]}")
    if empty_ctx:
        print(f"  ⚠ 检索结果为空 {len(empty_ctx)} 条：{empty_ctx[:5]}")

    if a.dry:
        for r in rows:
            print("-" * 62)
            print(f"  qid      : {r['qid']}")
            print(f"  question : {r['question']}")
            print(f"  gold     : {r['gold']}")
            print(f"  reference: {r['reference'][:110]}…")
            first = r["contexts"][0][:80] if r["contexts"] else "（空）"
            print(f"  contexts : {len(r['contexts'])} 条 | 首条 {first}…")
        print(f"\n链路 OK（结果写到 {os.path.basename(out)}）。"
              f"全量跑：去掉 --dry；然后 & $PY_RAGAS ragas_eval.py --stage score")


# ============================== 阶段 ② score ==============================
def build_judge():
    """判官 = qwen-plus（OpenAI 兼容端点）+ instructor 结构化输出。

    已核实 ragas 0.4.3 的 llm_factory 签名：
        llm_factory(model, provider="openai", client=AsyncOpenAI(...), **kwargs)
      · provider="openai" 时走 _map_openai_params，只对 o-series / gpt-5 改名
        max_tokens → max_completion_tokens；模型名 "qwen-plus" 不匹配任何 pattern
        → 参数原样透传，通义兼容，安全
      · InstructorModelArgs 默认 temperature=0.01 / top_p=0.1 / max_tokens=1024
        → max_tokens=1024 对 context_recall 的「逐句归因 + reason」JSON 太短，会
          静默截断成 NaN，故显式提到 4096；temperature 归零（判官要求确定性）
    """
    key = os.environ.get("DASHSCOPE_API_KEY")
    base = os.environ.get("DASHSCOPE_BASE_URL") or \
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not key:
        raise SystemExit(f"缺 DASHSCOPE_API_KEY，请写进 {ENV_PATH}")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=key, base_url=base)
    mod = None
    err = None
    for mod_name in ("ragas.llms.base", "ragas.llms"):
        try:
            mod = __import__(mod_name, fromlist=["llm_factory"])
            break
        except Exception as e:
            err = e
    if mod is None:
        raise SystemExit(f"没找到 llm_factory（{err}）→ 请 pip install -U ragas")
    try:
        return mod.llm_factory(JUDGE, provider="openai", client=client,
                               temperature=0.0, max_tokens=4096)
    except Exception as e:
        print(f"  （temperature/max_tokens 不被接受，退回默认：{type(e).__name__}）")
        return mod.llm_factory(JUDGE, provider="openai", client=client)


def load_metrics():
    """兼容 ragas 新旧两版 API，返回 (api_kind, [(名称, 指标对象), ...])。"""
    try:
        from ragas.metrics.collections import ContextPrecision, ContextRecall
        return "collections", [("context_recall", ContextRecall),
                               ("context_precision", ContextPrecision)]
    except Exception:
        pass
    try:
        from ragas.metrics import context_recall, context_precision
        return "legacy", [("context_recall", context_recall),
                          ("context_precision", context_precision)]
    except Exception as e:
        raise SystemExit(f"加载 RAGAS 指标失败：{e}\n→ pip install -U ragas")


async def _ascore(metric, **kw):
    """按实际签名裁剪参数，避免不同 ragas 版本的参数名差异直接炸掉。"""
    try:
        params = set(inspect.signature(metric.ascore).parameters)
        kw = {k: v for k, v in kw.items() if k in params}
    except Exception:
        pass
    r = await metric.ascore(**kw)
    return float(getattr(r, "value", r))


async def score_one(kind, metrics, judge, row):
    q, ctx, ref = row["question"], row["contexts"], row["reference"]
    if kind == "collections":
        out = {}
        for name, cls in metrics:
            out[name] = await _ascore(cls(llm=judge), user_input=q,
                                      retrieved_contexts=ctx, reference=ref)
        return out

    from datasets import Dataset
    from ragas import evaluate
    ds = Dataset.from_dict({"question": [q], "contexts": [ctx],
                            "ground_truth": [ref], "answer": [""]})
    res = evaluate(ds, metrics=[m for _, m in metrics], llm=judge,
                   raise_exceptions=False)
    d = res.to_pandas().iloc[0]
    return {name: float(d.get(name, float("nan"))) for name, _ in metrics}


def _mean(xs):
    xs = [x for x in xs if x == x]          # 去 NaN
    return sum(xs) / len(xs) if xs else float("nan")


def pick_input(a):
    """选输入文件：显式 dry → dry；否则全量；再退 dry。"""
    if a.dry and os.path.exists(IN_DRY):
        return IN_DRY
    if os.path.exists(IN_JSONL):
        return IN_JSONL
    if os.path.exists(IN_DRY):
        return IN_DRY
    return None


async def _score_loop(kind, metrics, judge, todo, on_done=None, quiet=False,
                      concurrency=1):
    """★ 必须在**同一个 event loop** 里跑完全部：AsyncOpenAI 的 httpx 连接池
    绑定创建它的 loop，若每条都 asyncio.run() 新建 loop，第 2 条起必报
    `Event loop is closed` / `attached to a different loop`。

    concurrency=1（默认）= 串行；>1 时用信号量限流并发（实测单条约 15~20s，
    77 条串行 20~25 分钟 → 并发 3 可压到 8 分钟左右）。
    注意 on_done 在锁内调用，缓存文件不会被并发写坏。
    """
    out = [None] * len(todo)
    if not todo:
        return out
    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    state = {"n": 0}

    async def one(i, r):
        async with sem:
            try:
                s = await score_one(kind, metrics, judge, r)
            except Exception as e:
                s = {"context_recall": float("nan"), "context_precision": float("nan")}
                print(f"\n  [{r['qid']}] 判官出错：{type(e).__name__}: {str(e)[:220]}")
        out[i] = s
        async with lock:
            state["n"] += 1
            if on_done:
                on_done(r, s, state["n"])
            if not quiet:
                v = s["context_recall"]
                sys.stdout.write(f"\r  评判 {state['n']}/{len(todo)}  {r['qid']:10s} "
                                 f"{'NaN' if v != v else format(v, '.2f')}")
                sys.stdout.flush()

    await asyncio.gather(*(one(i, r) for i, r in enumerate(todo)))
    if not quiet:
        print()
    return out


# ============================== 阶段 ①-b doctor ==============================
def stage_doctor(a):
    """装完包 30 秒内验证「判官链路 + 指标 API」是否可用，只花 1 条的钱。
    通不过就别跑全量，会白烧 77 条。"""
    load_env()
    install_langchain_shims()   # main 已调过；这里幂等，保证单独调用也安全
    shimmed = bool(getattr(sys.modules.get("langchain_community.chat_models.vertexai"),
                           "__wb_shim__", False))
    import platform
    print("=" * 62)
    print("  RAGAS 链路体检")
    print("=" * 62)
    try:
        import ragas
        print(f"  python        {platform.python_version()}")
        print(f"  ragas         {ragas.__version__}")
    except ModuleNotFoundError as e:
        miss = str(e).split("'")[1] if "'" in str(e) else str(e)
        if miss.startswith("ragas"):
            raise SystemExit(f"  ✗ ragas 未安装 → 用 ragas venv 的 pip 装：\n"
                             f"      pip install -U ragas datasets openai")
        raise SystemExit(f"  ✗ 依赖缺失：{miss}\n"
                         f"    → pip install -U ragas 补齐依赖")
    except Exception as e:
        raise SystemExit(f"  ✗ import ragas 失败：{type(e).__name__}: {e}")
    if shimmed:
        print("  已打桩        ChatVertexAI（langchain-community>=0.4 删了该模块，"
              "ragas 0.4.3 仍硬 import）")
    try:
        import langchain_community as _lc
        print(f"  langchain-com {getattr(_lc, '__version__', '?')}")
    except Exception:
        pass
    try:
        kind, metrics = load_metrics()
        print(f"  指标 API      {kind}  →  "
              f"{', '.join(n for n, _ in metrics)}")
    except SystemExit as e:
        raise
    print(f"  判官模型      {JUDGE}  ({os.environ.get('DASHSCOPE_BASE_URL', '')[:48]})")
    print(f"  Key           {'已读到' if os.environ.get('DASHSCOPE_API_KEY') else '✗ 缺失'}")

    if a.no_call:
        print("-" * 62)
        print("  --no-call：跳过真实调用，只验证 import 与对象构造（零成本）")
        judge = build_judge()
        print(f"  判官对象      {type(judge).__name__}  "
              f"model_args={getattr(judge, 'model_args', None)}")
        for name, cls in metrics:
            print(f"  指标对象      {name} -> {cls(llm=judge).name}")
        print("\n  ✓ import 与构造全通（网络与 Key 未验证）")
        return

    src = pick_input(a)
    if src is None:
        print("  ⚠ 没有 ragas_input.jsonl → 用内置样例跑（不影响判定链路）")
        row = {"qid": "DOCTOR", "question": "图书馆书库的净高要求是多少？",
               "reference": "书库的净高不应小于2.40m；当有梁或管线时，其底面至地面的净高不应小于2.20m。",
               "contexts": ["5.2.1 书库、非书资料库的净高应符合下列规定："
                            "1 采用地面直接承重存放书籍时，其净高不应小于2.40m；"
                            "2 有梁或管线时，其底面至地面的净高不应小于2.20m。"],
               "gold": ["4.2.8"], "type": "doctor"}
    else:
        rows = read_jsonl(src)
        if a.limit:
            rows = rows[: a.limit]
        if not rows:
            raise SystemExit(f"  ✗ {os.path.basename(src)} 是空的")
        row = rows[0]
        print(f"  输入          {os.path.basename(src)} 的第 1 条 [{row['qid']}]")

    print(f"  问题          {row['question'][:56]}")
    print(f"  标准答案      {row['reference'][:56]}…")
    print(f"  检索结果      {len(row['contexts'])} 条")
    print("-" * 62)
    print("  真实调用判官（这是唯一一次花钱的验证，约几秒）…")
    judge = build_judge()
    t = time.time()
    try:
        s = asyncio.run(_score_loop(kind, metrics, judge, [row], quiet=True))
        s = s[0]
    except Exception as e:
        raise SystemExit(f"  ✗ 判官链路失败：{type(e).__name__}: {str(e)[:400]}\n"
                         f"    常见原因：Key 无效 / 端点不通 / instructor 结构化输出被拒\n"
                         f"    → 换 CHAT_MODEL 或自查 dashscope 控制台")
    print(f"  context_recall    {s['context_recall']}")
    print(f"  context_precision {s['context_precision']}")
    print(f"  耗时              {time.time() - t:.1f}s")
    nan = s["context_recall"] != s["context_recall"]
    if nan:
        print("\n  ⚠ 分数是 NaN：判官返回没法解析（多半被 max_tokens 截断或拒答）")
        print("    → 已默认 max_tokens=4096；仍 NaN 就换更大的判官模型重试")
    else:
        print("\n  ✓ 链路 OK。可以跑全量了：")
        print("    & $PY_CHROMA ragas_eval.py --stage export --top-k 5")
        print("    & $PY_RAGAS  ragas_eval.py --stage score  --tag k5")


# ============================== 阶段 ② score ==============================
def stage_score(a):
    load_env()
    src = pick_input(a)
    if not os.path.exists(IN_JSONL) and src == IN_DRY:
        print("⚠ 没找到 ragas_input.jsonl，用 _dry 文件评分（只有 2 条）")
    if src is None:
        raise SystemExit(f"缺 {os.path.basename(IN_JSONL)} → 先用 chroma venv 跑 --stage export")
    rows = read_jsonl(src)
    if a.limit is not None:
        rows = rows[: a.limit]

    judge = build_judge()
    kind, metrics = load_metrics()
    n_call = "约 5~15 次/条" if kind == "collections" else "1 次/条"
    print(f"输入 {os.path.basename(src)} · {len(rows)} 条 · 判官 {JUDGE} · "
          f"指标 API {kind}（{n_call} LLM 调用）")

    cache = {}
    for o in read_jsonl(CACHE):
        cache[o["qid"]] = o
    score_map, todo = {}, []
    for r in rows:
        c = cache.get(r["qid"])
        if c and "context_recall" in c:
            score_map[r["qid"]] = {"context_recall": c["context_recall"],
                                   "context_precision": c["context_precision"]}
        else:
            todo.append(r)
    print(f"  缓存命中 {len(score_map)} 条 · 待评 {len(todo)} 条")

    def on_done(r, s, n):
        score_map[r["qid"]] = s
        with open(CACHE, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"qid": r["qid"], "judge": JUDGE,
                                "context_recall": s["context_recall"],
                                "context_precision": s["context_precision"]},
                               ensure_ascii=False) + "\n")

    if todo:
        asyncio.run(_score_loop(kind, metrics, judge, todo, on_done=on_done,
                                concurrency=a.concurrency))

    done = [{**r, **score_map.get(r["qid"], {})} for r in rows]

    tag = a.tag or "base"
    out_csv = os.path.join(BASE, f"结果_ragas_{tag}.csv")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["qid", "type", "difficulty", "standard", "query", "gold",
                    "context_recall", "context_precision", "n_contexts"])
        for r in done:
            w.writerow([r["qid"], r["type"], r["difficulty"], r["standard"],
                        r["question"], "/".join(r["gold"] or []),
                        f"{r['context_recall']:.3f}", f"{r['context_precision']:.3f}",
                        len(r["contexts"])])

    nan = sum(1 for r in done if r["context_recall"] != r["context_recall"])
    print("=" * 62)
    print(f"  context_recall    {_mean([r['context_recall'] for r in done]):.3f}   ← 召回率")
    print(f"  context_precision {_mean([r['context_precision'] for r in done]):.3f}   ← 准确率")
    print(f"  判官解析失败(NaN) {nan} 条"
          + ("   ⚠ 判官可能拒答或超 max_tokens，去掉缓存重跑该几条" if nan else ""))
    print("=" * 62)

    for key, title in (("type", "按题型"), ("standard", "按规范"), ("difficulty", "按难度")):
        g = {}
        for r in done:
            g.setdefault(r[key], []).append(r["context_recall"])
        print(f"\n{title}（context_recall 升序）")
        for k, v in sorted(g.items(), key=lambda kv: _mean(kv[1])):
            print(f"  {str(k):24s} {_mean(v):.3f}  (n={len(v)})")

    bad = sorted([r for r in done if r["context_recall"] == r["context_recall"]],
                 key=lambda r: (r["context_recall"], r["context_precision"]))[: a.low_n]
    print(f"\n最差 {len(bad)} 条 —— Bad Case 候选（召回率升序）")
    print("  RAGAS 只给分数、不给原因；原因分类交给 run_eval.py 的「未命中明细」")
    for r in bad:
        print(f"  {r['qid']:10s} R={r['context_recall']:.2f} P={r['context_precision']:.2f} "
              f"{str(r['type']):6s} {str(r['question'])[:24]:26s} "
              f"gold={'/'.join(r['gold'] or [])}")
    print(f"\n→ {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["export", "doctor", "score"], required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-standard", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--low-n", type=int, default=12, help="报告里列出最差多少条")
    ap.add_argument("--dry", action="store_true",
                    help="export：只跑 2 条、写 _dry 文件、打印样例；score：读 _dry 文件")
    ap.add_argument("--no-call", action="store_true",
                    help="doctor：跳过真实判官调用，只验证 import 与对象构造（零成本）")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="score：并发评判条数（默认 1 串行；3 约可快 3 倍，"
                         "过高可能触发 DashScope 限流）")
    a = ap.parse_args()
    install_langchain_shims()   # ★ 必须在任何 ragas import 之前（export 阶段用不到也无害）
    if a.stage == "export":
        if a.dry and not a.limit:
            a.limit = 2
        stage_export(a)
    elif a.stage == "doctor":
        stage_doctor(a)
    else:
        stage_score(a)


if __name__ == "__main__":
    main()
