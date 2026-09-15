# -*- coding: utf-8 -*-
"""本地规范问答应用包。

⚠️ 这个文件里做的唯一一件事，必须发生在**任何第三方库被 import 之前**：
   设定向量模型的缓存位置（HF_HOME）。

为什么必须放这里
----------------
huggingface_hub 在**被 import 的那一刻**就把 HF_HOME 读走并锁死了，之后再改环境变量
无效。而 sentence-transformers / transformers 都会 import 它。所以只要顺序错了，
程序就会去联网重下 1.1 GB 模型（或者直接失败）。

把 os.environ 设置写在包的 ``__init__`` 里 = 只要有人 `from app.xxx import yyy`，
这个包会先被初始化，环境变量必定最先设好。这是最省心的顺序保障。
"""
import os

# setdefault：你自己在系统里设过 HF_HOME / HF_ENDPOINT 就以你的为准
os.environ.setdefault("HF_HOME", r"D:\WorkBuddy\models\hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

__all__ = ["config", "retriever", "prompts", "llm", "pipeline"]
