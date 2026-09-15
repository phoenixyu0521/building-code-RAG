# -*- coding: utf-8 -*-
"""生成层 —— 调用通义千问（阿里云百炼 · OpenAI 兼容接口）。

为什么用 requests 手写而不用 openai SDK
----------------------------------------
这个 venv 里已经装了 requests（chromadb 的依赖），但**没有装 openai**。
手写一个 POST 只有 10 行，还能避免多装一个包、避免 SDK 版本变动。
接口本身是 OpenAI 兼容的，随时想换回 SDK 也只改这一个文件。

为什么用 OpenAI 兼容模式而不是 DashScope 原生接口
--------------------------------------------------
兼容模式的响应结构和 OpenAI 一致（choices[0].message.content），
以后想换 DeepSeek / Kimi / Ollama，只改 base_url + model 两个配置即可。

自检（要 API Key）：
    python -m app.llm "用一句话介绍你自己"
"""
import json
import sys

import requests

from app.config import (DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, CHAT_MODEL,
                        TEMPERATURE)

TIMEOUT = 90          # 秒。规范条文长，回答可能较慢


class LLMError(RuntimeError):
    """调用失败时抛出，message 里带**可直接照做的修法**。"""


def chat(messages, temperature=None, model=None, timeout=TIMEOUT):
    """发一轮对话，返回 (回答文本, usage dict)。

    messages 形如 [{"role": "system", "content": ...}, {"role": "user", ...}]
    """
    if not DASHSCOPE_API_KEY:
        raise LLMError(
            "没有读到 API Key。请检查：\n"
            "  1) 项目根目录下有没有 .env 文件（可复制 .env.example 改名而来）\n"
            "  2) .env 里有没有这一行：DASHSCOPE_API_KEY=sk-xxxxxxxx\n"
            "  Key 在阿里云百炼控制台 → API-KEY 页面创建（必须**按量付费**、"
            "用**默认业务空间**）"
        )

    url = DASHSCOPE_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": model or CHAT_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE if temperature is None else temperature,
        "stream": False,          # 命令行演示不需要流式；要打字机效果再改这里
    }
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, headers=headers,
                             data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                             timeout=timeout)
    except requests.exceptions.Timeout:
        raise LLMError(f"请求超时（{timeout} 秒）。稍后重试，或把 TIMEOUT 调大。")
    except requests.exceptions.ConnectionError as e:
        raise LLMError(f"连不上 {url}\n  原始错误：{e}")

    if resp.status_code != 200:
        raise LLMError(_explain_http_error(resp))

    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise LLMError(f"响应结构异常：{json.dumps(data, ensure_ascii=False)[:400]}")
    return text, data.get("usage", {})


def _explain_http_error(resp):
    """把 HTTP 错误翻译成中文的「下一步怎么办」。"""
    raw = resp.text[:300]
    code = resp.status_code
    tips = {
        400: "常见原因：模型名写错，或消息格式不对。当前模型见 app/config.py 的 CHAT_MODEL。",
        401: "Key 无效或与地域不匹配。注意：百炼 Key 与地域绑定 —— 北京 Key 只能配\n"
             "     https://dashscope.aliyuncs.com/compatible-mode/v1；\n"
             "     新加坡 Key 要改成 https://dashscope-intl.aliyuncs.com/compatible-mode/v1。\n"
             "     另外务必用**默认业务空间**的 Key（子空间 Key 会报这个错）。",
        403: "权限不足。确认已开通百炼模型服务，且 Key 是按量付费类型。",
        429: "触发限流或欠费。检查账户余额；或稍后重试。",
        500: "服务端错误，稍后重试。",
        503: "服务暂时不可用，稍后重试。",
    }
    extra = tips.get(code, "")
    return f"HTTP {code}\n  {extra}\n  服务端返回：{raw}"


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    q = " ".join(sys.argv[1:]) or "用一句话介绍你自己"
    print(f"模型：{CHAT_MODEL}  接口：{DASHSCOPE_BASE_URL}")
    print(f"提问：{q}\n")
    try:
        text, usage = chat([{"role": "user", "content": q}])
    except LLMError as e:
        print("❌ 调用失败：\n" + str(e))
        raise SystemExit(1)
    print("✅ 回答：")
    print(text)
    if usage:
        print(f"\n（用量：{usage}）")
