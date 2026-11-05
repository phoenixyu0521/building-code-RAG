# -*- coding: utf-8 -*-
"""tests/ 包。

这些测试**不依赖** torch / chromadb / 网络 / API Key：
guardrails 是纯函数，pipeline 用 monkeypatch 打桩；
索引相关用例只用临时目录和假客户端。
所以 `python -m unittest discover -s tests -v` 在哪台机器上都能直接跑。
"""
