# -*- coding: utf-8 -*-
"""访问 Dify demo 并抓取结构 / 问答复现。整段必须同一条命令内跑完。"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdp_min import WS, WSError, connect_page, ev, http_json  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
DBG = 9222
DEMO = "https://udify.app/chat/pshniTIvIfA2nqbA"
OUT = r"D:\WorkBuddy\项目\输出文档\demo体验"
_profile = os.path.join(tempfile.gettempdir(),
                        os.environ.get("CDP_PROFILE", "cdp_profile_9222"))


def log(*a):
    print(*a, flush=True)


def launch():
    os.makedirs(_profile, exist_ok=True)
    a = [CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
         "--no-default-browser-check", "--remote-allow-origins=*",
         "--remote-debugging-port=%d" % DBG, "--user-data-dir=" + _profile,
         "--window-size=1440,1400", "--hide-scrollbars", "--lang=zh-CN",
         "about:blank"]
    p = subprocess.Popen(a, creationflags=0x8 | 0x200,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        try:
            http_json("http://127.0.0.1:%d/json/version" % DBG)
            log("chrome: up")
            return p
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("chrome 未就绪")


def wait_ready(ws, need_text=True, timeout=45):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = ev(ws, "document.readyState + '|' + (document.body?document.body.innerText.length:0)")
        if isinstance(st, str) and st.startswith("complete"):
            if not need_text or int(st.split("|")[1] or 0) > 20:
                return st
        time.sleep(1)
    return st


def shot(ws, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = ws.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
    if "data" in r:
        open(path, "wb").write(base64.b64decode(r["data"]))
        return path
    return "__no data__"


def new_page():
    p = launch()
    ws, tid = connect_page(DBG)
    ws.call("Page.enable")
    ws.call("Runtime.enable")
    ws.call("Page.navigate", {"url": DEMO})
    log("readyState:", wait_ready(ws))
    time.sleep(5)
    return p, ws, tid


def kill_all(p):
    try:
        p.kill()
    except Exception:
        pass
    try:
        subprocess.run(
            ["wmic", "process", "where",
             "name='chrome.exe' and CommandLine like '%cdp_profile_9222%'",
             "call", "terminate"], capture_output=True, timeout=15)
    except Exception:
        pass


def recon():
    p, ws, tid = new_page()
    try:
        log("\nURL  :", ev(ws, "location.href"))
        log("标题 :", ev(ws, "document.title"))
        log("\n=== 页面文本（前 1200 字）===")
        log(ev(ws, "(document.body.innerText||'').slice(0,1200)"))

        log("\n=== 输入/按钮元素 ===")
        log(ev(ws, r"""
[...document.querySelectorAll('textarea,input,button,[contenteditable=true],[role=button]')]
 .map((e,i) => i+' '+e.tagName+' id='+(e.id||'-')+' ph='+(e.getAttribute('placeholder')||'-')
   +' aria='+(e.getAttribute('aria-label')||'-')+' txt='+((e.innerText||'').trim().slice(0,25))
   +' cls='+((typeof e.className==='string'?e.className:'').slice(0,60))).join('\n')
"""))

        log("\n=== 关键 class 名 ===")
        log(ev(ws, r"""
(() => { const s=new Set();
 document.querySelectorAll('*').forEach(e=>{const c=(typeof e.className==='string')?e.className:'';
  c.split(/\s+/).forEach(x=>{if(/chat|message|markdown|query|answer|send/i.test(x))s.add(x);});});
 return [...s].slice(0,100).join(' | ');})()
"""))
        log("\n=== 消息容器计数 ===")
        log(ev(ws, r"""
['.chat-message-container','.chat-message','.markdown-body','textarea','[class*=send]','button']
 .map(s=>s+' -> '+document.querySelectorAll(s).length).join('\n')
"""))
        log("\nscreenshot ->", shot(ws, os.path.join(OUT, "demo_初始页.png")))
    finally:
        try:
            ws.close()
        except Exception:
            pass
        kill_all(p)


def ask(questions, tag="run"):
    os.makedirs(OUT, exist_ok=True)
    p, ws, tid = new_page()
    results = []
    try:
        log("=== 页面文本 ===")
        log(ev(ws, "(document.body.innerText||'').slice(0,600)"))
        for i, q in enumerate(questions, 1):
            log("\n########## 第 %d 问：%s" % (i, q))
            r = ev(ws, r"""
(() => {
  const ta = document.querySelector('textarea');
  if (!ta) return 'NO_TEXTAREA';
  const set = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value').set;
  set.call(ta, %s);
  ta.dispatchEvent(new Event('input', {bubbles:true}));
  ta.focus();
  return 'filled';
})()
""" % json.dumps(q))
            log("fill ->", r)
            time.sleep(1.2)
            r = ev(ws, r"""
(() => {
  const btns=[...document.querySelectorAll('button')];
  let b = btns.find(x=>/send|发送/i.test((x.getAttribute('aria-label')||'')+x.className));
  if(!b) b = btns.filter(x=>x.type==='submit').pop();
  if(!b) b = btns[btns.length-1];
  if(!b) return 'NO_BTN';
  b.click();
  return 'clicked: '+(b.getAttribute('aria-label')||b.className||b.type);
})()
""")
            log("send ->", r)
            prev, stable, t0 = None, 0, time.time()
            while time.time() - t0 < 180:
                time.sleep(3)
                txt = ev(ws, "(document.body.innerText||'')")
                if txt == prev:
                    stable += 1
                    if stable >= 5:
                        break
                else:
                    prev, stable = txt, 0
            log("回答:\n", (prev or "")[-2500:])
            results.append({"q": q, "page_text": prev})
            shot(ws, os.path.join(OUT, "demo_%s_q%d.png" % (tag, i)))
    finally:
        with open(os.path.join(OUT, "demo问答_%s.json" % tag), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        try:
            ws.close()
        except Exception:
            pass
        kill_all(p)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "recon"
    if mode == "recon":
        recon()
    elif mode == "ask":
        src = sys.argv[2]
        if os.path.exists(src):
            with open(src, encoding="utf-8") as f:
                qs = json.load(f)
        else:
            qs = json.loads(src)
        ask(qs, sys.argv[3] if len(sys.argv) > 3 else "run")
