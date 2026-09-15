# AGENT.md —— 协作与提交规范

本文件约束**在本仓库中工作的人与 AI Agent**。每次开始改动前先读一遍。

---

## 注意事项

1. 每次改动完成后，都必须创建一个对应的 Git commit，以便后续追踪和回滚。
2. 每次改动后，都必须编写或更新相关测试，并且在交付给用户之前，确保所有测试和验证全部通过。

---

## 落地细则

上面两条是原则，下面是可执行的判定标准。

### 一、提交（对应注意事项 1）

- **一个逻辑改动 = 一个 commit**。修 bug 与改文档不要混在同一个 commit 里。
- message 用 Conventional Commits 前缀：`feat:` / `fix:` / `docs:` / `refactor:` / `test:` / `chore:`。
- 提交前必须先跑 `git status --porcelain`，确认**没有**把下列内容带进暂存区：
  - `.env`（含 API Key）、`config.json`
  - `建筑规范PDF/*.pdf`（受版权保护）
  - 向量库、`*.npy`、`*.sqlite3`
  - `输出文档/scripts/_*`（一次性探针脚本）、`__pycache__/`
- ⚠️ **`git commit` 只写本地**。要同步到 GitHub 必须再执行 `git push`，两者不是一回事。
- ⚠️ 本仓库**禁止**执行 `git gc` 与 `git reset --hard`（历史上曾因中断的 `gc` 丢失 blob）。

### 二、测试与验证（对应注意事项 2）

本仓库**不使用 pytest 等测试框架**。验证手段是「模块内置自检 + 独立校验脚本」，按下表选择必跑的项。

| 改动范围 | 必须执行的验证 |
|---|---|
| 切分脚本 `split_articles_v2.py` | `validate_chunks.py` + `检索评测/build_evalset.py`（金标绑定当前文本，必须重跑） |
| `build_v4.py`（OCR 修复白名单） | 重跑全链路 → `validate_chunks.py` |
| 建库 / 向量化 `build_index.py` | `verify_index.py`（7 项，不加载模型） |
| `app/` 任一模块 | 该模块自检：`python -m app.<模块名>` |
| 提示词 `app/prompts.py` | `python -m app.prompts`（与《Dify导入说明》第三节**逐字 diff**） |
| 检索逻辑 `app/retriever.py` | `检索评测/run_eval.py --backend chroma --profile local` |
| 配置 / 路径 `app/config.py` | `python -m app.config` |
| 提示词或拒答策略 | 跑 `输出文档/知识库测试题.md`，**D 组必须全部拒答** |

命令用法见 `README.md`「本地版快速开始」与 `输出文档/本地部署操作指引.md`。

**交付前的检查清单：**

- [ ] 上表中相关项的自检**全部通过**，且无新增失败
- [ ] 已确认输出无 `[safe-delete][SAFE_DELETE_BULK_CONFIRM_REQUIRED]` 之类的中断标记
- [ ] 没有新增临时文件、探针脚本、缓存目录进入版本控制
- [ ] 已创建对应 commit（注意事项 1）

> 若某项验证因缺少条件无法执行（如无 API Key、无 Dify 访问权限），**必须在交付说明中明确指出是哪一项、为什么没跑**，不得默认通过。

### 三、硬性技术约束（勿破坏）

这些是让"验证全部通过"有意义的前提，改动前务必确认没有踩到：

- **向量库目录必须纯 ASCII** —— chromadb 1.5.9 的 Rust HNSW 在含中文的路径下**静默不写索引且不报错**，做出的库打不开。判定依据是 4 个 `.bin` 文件都在。
- **`HF_HOME` 必须在 `import sentence_transformers` 之前设定** —— 由 `app/__init__.py` 专责，不得挪位置。
- **检索侧必须带 `prompt_name="query"`** —— 漏掉不报错，只让效果悄悄变差。
- **Chroma 多字段过滤必须显式写 `$and`** —— 直接并列多个字段会被当成同一字段的复合条件而报错。
- **7 条铁律与《Dify导入说明》第三节保持逐字一致** —— 已实测有效（条文号 3/3、拒答 2/2），不可随手润色。
- **一条正文 = 一个 chunk，绝不切分** —— 切分粒度是本项目准确性的根基。
