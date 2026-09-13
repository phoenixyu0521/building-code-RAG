# -*- coding: utf-8 -*-
"""极简 CDP 客户端：手写 WebSocket 握手（不带 Origin → 绕过 Chrome 的 origin 校验）。"""
import base64
import json
import os
import socket
import struct
import time
import urllib.request


class WSError(Exception):
    pass


class WS:
    def __init__(self, url, timeout=30):
        assert url.startswith("ws://"), url
        rest = url[5:]
        hostport, _, path = rest.partition("/")
        path = "/" + path
        host, _, port = hostport.partition(":")
        self.s = socket.create_connection((host, int(port or 80)), timeout)
        self.s.settimeout(timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = ("GET %s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\n"
               "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
               "Sec-WebSocket-Version: 13\r\n\r\n") % (path, hostport, key)
        self.s.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            d = self.s.recv(4096)
            if not d:
                raise WSError("连接被关闭")
            buf += d
        head, _, self._buf = buf.partition(b"\r\n\r\n")
        status = head.split(b"\r\n")[0].decode("latin1")
        if "101" not in status:
            raise WSError("握手失败: " + status)
        self._mid = 0

    # ---- 底层帧 ----
    def _read(self, n):
        while len(self._buf) < n:
            d = self.s.recv(65536)
            if not d:
                raise WSError("连接中断")
            self._buf += d
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _send_frame(self, opcode, payload):
        b = bytearray([0x80 | opcode])
        n = len(payload)
        if n < 126:
            b.append(0x80 | n)
        elif n < 65536:
            b.append(0x80 | 126)
            b += struct.pack(">H", n)
        else:
            b.append(0x80 | 127)
            b += struct.pack(">Q", n)
        mask = os.urandom(4)
        b += mask
        b += bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
        self.s.sendall(bytes(b))

    def recv_message(self, timeout=60):
        self.s.settimeout(timeout)
        frags = []
        op = None
        while True:
            h = self._read(2)
            fin = h[0] & 0x80
            opcode = h[0] & 0x0F
            masked = h[1] & 0x80
            ln = h[1] & 0x7F
            if ln == 126:
                ln = struct.unpack(">H", self._read(2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if masked else None
            data = self._read(ln) if ln else b""
            if mask:
                data = bytes(c ^ mask[i % 4] for i, c in enumerate(data))
            if opcode == 0x9:                      # ping
                self._send_frame(0xA, data)
                continue
            if opcode == 0xA:                      # pong
                continue
            if opcode == 0x8:
                raise WSError("服务端关闭")
            if opcode != 0:
                op = opcode
            frags.append(data)
            if fin:
                break
        return b"".join(frags).decode("utf-8", "ignore")

    # ---- CDP ----
    def call(self, method, params=None, timeout=90, _id=None):
        self._mid += 1
        mid = _id or self._mid
        self._send_frame(0x1, json.dumps(
            {"id": mid, "method": method, "params": params or {}}).encode())
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = json.loads(self.recv_message(timeout=timeout))
            if msg.get("id") == mid:
                if "error" in msg:
                    return {"__error__": msg["error"]}
                return msg.get("result", {})
        raise WSError("CDP 超时: " + method)

    def close(self):
        try:
            self.s.close()
        except Exception:
            pass


def http_json(url, timeout=10):
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return json.loads(op.open(url, timeout=timeout).read().decode("utf-8"))


def pick_page(port=9222, prefer="about:blank"):
    lst = http_json("http://127.0.0.1:%d/json/list" % port)
    pages = [t for t in lst if t.get("type") == "page"]
    for t in pages:
        if t.get("url") == prefer:
            return t
    return pages[0] if pages else None


def connect_page(port=9222):
    t = pick_page(port)
    if not t:
        raise WSError("没有可用的 page target")
    return WS(t["webSocketDebuggerUrl"]), t["id"]


def ev(ws, expr, timeout=60, await_promise=False):
    r = ws.call("Runtime.evaluate", {
        "expression": expr, "returnByValue": True,
        "awaitPromise": await_promise, "userGesture": True}, timeout=timeout)
    if "__error__" in r:
        return "__CDP_ERR__ " + json.dumps(r["__error__"], ensure_ascii=False)
    res = r.get("result", {})
    if r.get("exceptionDetails"):
        return "__JS_ERR__ " + json.dumps(r["exceptionDetails"], ensure_ascii=False)[:400]
    return res.get("value")
