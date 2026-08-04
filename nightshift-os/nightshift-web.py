#!/usr/bin/env python3
"""nightshift-web — the browser console for NightShift OS.

A single-file, stdlib-only web GUI for operating an Arm Linux server in natural language.
You type intent; the local Llama agent streams its reasoning, runs read-only diagnostics
automatically, and asks for approval (in the browser) before any state change. A live health
strip shows disk/memory/load. Everything stays on the box.

Serves on 0.0.0.0:8088 and talks to the local model server (nightshift-llm) on 127.0.0.1:8080.
"""
import json
import os
import queue
import re
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LLM_URL = os.environ.get("NIGHTSHIFT_LLM_URL", "http://127.0.0.1:8080").rstrip("/")
MODEL = os.environ.get("NIGHTSHIFT_MODEL", "nightshift")
PORT = int(os.environ.get("NIGHTSHIFT_WEB_PORT", "8088"))
AUDIT = os.environ.get("NIGHTSHIFT_AUDIT", "/var/log/nightshift-web.audit")
CMD_TIMEOUT = 30
MAX_STEPS = 12

READ_ONLY = {"ls", "cat", "head", "tail", "wc", "stat", "file", "find", "grep", "egrep", "awk",
    "sed", "cut", "sort", "uniq", "tr", "df", "du", "free", "ps", "top", "uptime", "who", "w",
    "id", "uname", "hostname", "hostnamectl", "date", "env", "printenv", "lscpu", "lsblk", "lsmem",
    "lsof", "ip", "ss", "netstat", "journalctl", "dmesg", "vmstat", "nproc", "getent", "sysctl",
    "readlink", "realpath", "echo", "pwd", "cmp", "diff", "md5sum", "sha256sum", "numfmt",
    "systemd-analyze", "tree", "cat"}
SAFE_SUB = {"systemctl": {"status", "list-units", "list-unit-files", "is-active", "is-enabled",
    "show", "cat", "list-timers"}, "apt": {"list", "show", "policy", "search"},
    "docker": {"ps", "images", "logs", "inspect"}, "git": {"status", "log", "diff", "show"}}
DENY = [r"\brm\s+-rf\s+/(?!\w)", r"\bmkfs", r"\bdd\b.*\bof=/dev/", r":\(\)\s*\{", r"\bshutdown\b",
        r"\breboot\b", r">\s*/dev/sd", r"\bchmod\s+-R\s+777\s+/"]


def classify(cmd):
    c = cmd.strip()
    for p in DENY:
        if re.search(p, c):
            return "deny"
    first = re.split(r"[|;&]", c)[0].strip().split()
    if not first:
        return "mutate"
    exe = os.path.basename(first[0])
    if exe in SAFE_SUB:
        return "read" if (len(first) > 1 and first[1] in SAFE_SUB[exe]) else "mutate"
    if re.search(r"(?<![0-9])>>?[^&]", c) or re.search(r"\btee\b", c):
        return "mutate"
    return "read" if exe in READ_ONLY else "mutate"


def audit(kind, detail):
    try:
        with open(AUDIT, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{kind}\t{detail}\n")
    except OSError:
        pass


def run_shell(cmd):
    try:
        p = subprocess.run(["/bin/bash", "-lc", cmd], capture_output=True, text=True, timeout=CMD_TIMEOUT)
        out = (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        out = f"[timed out after {CMD_TIMEOUT}s]"
    except Exception as e:  # noqa: BLE001
        out = f"[error: {e}]"
    return (out[:4000] + "\n...[truncated]") if len(out) > 4000 else (out or "[no output]")


SYSTEM = """You are NightShift, an agent operating an Arm Linux server. Accomplish the task by \
reasoning and running shell commands. Respond with ONE JSON object and nothing else:
  {"thought": "...", "action": "shell", "command": "<one command>"}
  {"thought": "...", "action": "final", "answer": "<answer>"}
Prefer read-only diagnostics first (df, free, ps, journalctl, systemctl status). Only propose a
state-changing command when the task requires it; the admin approves it in the browser. Cite the
numbers you observed. Finish with action "final"."""


def llm(messages):
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": 0,
                       "max_tokens": 512}).encode()
    req = urllib.request.Request(LLM_URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse_action(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    s = m.group(0)
    depth = 0
    for i, ch in enumerate(s):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            try:
                return json.loads(s[:i + 1])
            except json.JSONDecodeError:
                return None
    return None


# pending browser approvals: token -> (Event, {"decision": bool})
PENDING = {}


def agent_stream(task, emit):
    audit("task", task)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    for step in range(1, MAX_STEPS + 1):
        try:
            reply = llm(messages)
        except Exception as e:  # noqa: BLE001
            emit("error", f"model server error: {e}")
            return
        act = parse_action(reply)
        if not act:
            emit("error", "could not parse agent action")
            return
        messages.append({"role": "assistant", "content": json.dumps(act)})
        if act.get("action") == "final":
            emit("final", act.get("answer", "").strip())
            audit("final", act.get("answer", "")[:300])
            return
        cmd = (act.get("command") or "").strip()
        kind = classify(cmd)
        emit("step", {"n": step, "thought": act.get("thought", ""), "command": cmd, "kind": kind})
        if kind == "deny":
            obs = "[BLOCKED by NightShift safety policy — hard deny-list]"
            audit("deny", cmd)
        elif kind == "mutate":
            tok = f"{step}-{int(time.time()*1000)%100000}"
            ev = threading.Event()
            PENDING[tok] = (ev, {})
            emit("approval", {"token": tok, "command": cmd})
            ok = ev.wait(timeout=120)
            decided = PENDING.pop(tok, (None, {}))[1].get("decision", False)
            if not ok or not decided:
                obs = "[admin declined / timed out — not executed]"
                audit("declined", cmd)
            else:
                audit("mutate", cmd)
                obs = run_shell(cmd)
        else:
            audit("read", cmd)
            obs = run_shell(cmd)
        emit("observation", {"n": step, "text": obs})
        messages.append({"role": "user", "content": "OBSERVATION:\n" + obs})
    emit("error", f"reached max steps ({MAX_STEPS})")


PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>NightShift OS</title><style>
:root{--bg:#0d0e10;--pan:#15171b;--fg:#dee0e4;--dim:#8a8f98;--grn:#5ed682;--cyn:#56b6e0;--acc:#e87846;--yel:#e8c24a;--red:#e05a5a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
header{padding:14px 20px;border-bottom:1px solid #23262c;display:flex;align-items:center;gap:14px}
.logo{font-weight:700;letter-spacing:.5px}.logo b{color:var(--acc)}
.health{margin-left:auto;font:12px ui-monospace,Menlo,monospace;color:var(--dim);white-space:pre}
#log{max-width:900px;margin:0 auto;padding:20px}
.msg{margin:14px 0}.user{color:var(--fg);background:#1b1e24;padding:10px 14px;border-radius:10px;border:1px solid #262a31}
.thought{color:var(--dim);font-style:italic;margin:8px 0 2px}
.cmd{font:13px ui-monospace,Menlo,monospace;padding:8px 12px;border-radius:8px;margin:4px 0;white-space:pre-wrap}
.read{background:#0f2027;border-left:3px solid var(--cyn)}.mutate{background:#2a2410;border-left:3px solid var(--yel)}
.deny{background:#2a1414;border-left:3px solid var(--red)}
.obs{font:12px ui-monospace,Menlo,monospace;color:var(--dim);background:#101216;padding:8px 12px;border-radius:8px;white-space:pre-wrap;max-height:260px;overflow:auto}
.answer{background:#0f2417;border:1px solid #1e5e38;border-radius:10px;padding:12px 14px;color:#c8f5d8;margin-top:8px}
.appr{background:#2a2410;border:1px solid var(--yel);border-radius:10px;padding:12px 14px;margin:6px 0}
.appr button{font-size:14px;padding:6px 16px;margin-right:8px;border-radius:7px;border:0;cursor:pointer}
.ok{background:var(--grn);color:#06210f}.no{background:#33373e;color:var(--fg)}
footer{position:sticky;bottom:0;background:var(--bg);border-top:1px solid #23262c;padding:14px 20px}
.bar{max-width:900px;margin:0 auto;display:flex;gap:10px}
input{flex:1;background:var(--pan);border:1px solid #2a2e35;color:var(--fg);border-radius:10px;padding:12px 14px;font-size:15px}
button.send{background:var(--acc);color:#160a04;border:0;border-radius:10px;padding:0 20px;font-weight:600;cursor:pointer}
.hint{max-width:900px;margin:0 auto;color:var(--dim);font-size:12px;padding-top:6px}
</style></head><body>
<header><span class=logo>NIGHT<b>SHIFT</b> OS</span><span class=dim style="color:var(--dim);font-size:12px">agentic · Arm · GPU-free</span><span class=health id=health>loading…</span></header>
<div id=log></div>
<footer><div class=bar><input id=q placeholder="Tell the OS what to do — e.g. 'why is the disk filling up?'" autofocus>
<button class=send onclick=go()>Run</button></div>
<div class=hint>Read-only diagnostics run automatically. State changes ask for your approval here. Everything stays on this machine.</div></footer>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),H=document.getElementById('health');
function el(c,h){const d=document.createElement('div');d.className=c;if(h!=null)d.innerHTML=h;return d}
function esc(s){return (s||'').replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]))}
async function health(){try{const r=await fetch('/api/health');H.textContent=await r.text()}catch(e){}}
setInterval(health,5000);health();
function go(){const t=q.value.trim();if(!t)return;q.value='';
 log.appendChild(el('msg','<div class=user>'+esc(t)+'</div>'));window.scrollTo(0,1e9);
 const es=new EventSource('/api/stream?task='+encodeURIComponent(t));
 es.addEventListener('step',e=>{const d=JSON.parse(e.data);
   if(d.thought)log.appendChild(el('msg','<div class=thought>'+esc(d.thought)+'</div>'));
   log.appendChild(el('cmd '+d.kind,'$ '+esc(d.command)));window.scrollTo(0,1e9)});
 es.addEventListener('observation',e=>{const d=JSON.parse(e.data);
   log.appendChild(el('obs',esc(d.text)));window.scrollTo(0,1e9)});
 es.addEventListener('approval',e=>{const d=JSON.parse(e.data);
   const box=el('appr','<div>Approve this state-changing command?</div><div class=cmd style="background:#000">$ '+esc(d.command)+'</div>');
   const ok=el('');ok.innerHTML='<button class=ok>Approve & run</button><button class=no>Deny</button>';
   ok.querySelector('.ok').onclick=()=>{fetch('/api/approve',{method:'POST',body:JSON.stringify({token:d.token,decision:true})});box.remove()};
   ok.querySelector('.no').onclick=()=>{fetch('/api/approve',{method:'POST',body:JSON.stringify({token:d.token,decision:false})});box.remove()};
   box.appendChild(ok);log.appendChild(box);window.scrollTo(0,1e9)});
 es.addEventListener('final',e=>{log.appendChild(el('msg','<div class=answer>'+esc(e.data)+'</div>'));es.close();window.scrollTo(0,1e9)});
 es.addEventListener('error',e=>{if(e.data)log.appendChild(el('msg','<div class=answer style="background:#2a1414;border-color:#5e1e1e;color:#f5c8c8">'+esc(e.data)+'</div>'));es.close()});
}
q.addEventListener('keydown',e=>{if(e.key==='Enter')go()});
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            return self._send(200, "text/html; charset=utf-8", PAGE)
        if self.path == "/api/health":
            out = run_shell("printf 'host %s | ' \"$(hostname)\"; "
                            "free -g | awk 'NR==2{printf \"mem %d/%dG | \",$3,$2}'; "
                            "df -h / | awk 'NR==2{printf \"root %s used | \",$5}'; "
                            "uptime | sed 's/.*load average/load/'")
            return self._send(200, "text/plain; charset=utf-8", out)
        if self.path.startswith("/api/stream"):
            from urllib.parse import urlparse, parse_qs
            task = parse_qs(urlparse(self.path).query).get("task", [""])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            qout = queue.Queue()
            done = object()

            def emit(ev, data):
                qout.put((ev, data))
                if ev in ("final", "error"):
                    qout.put(done)
            threading.Thread(target=agent_stream, args=(task, emit), daemon=True).start()
            while True:
                item = qout.get()
                if item is done:
                    break
                ev, data = item
                payload = data if isinstance(data, str) else json.dumps(data)
                try:
                    self.wfile.write(f"event: {ev}\ndata: {payload}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
            return
        self._send(404, "text/plain", "not found")

    def do_POST(self):
        if self.path == "/api/approve":
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n) or b"{}")
            tok = d.get("token")
            if tok in PENDING:
                PENDING[tok][1]["decision"] = bool(d.get("decision"))
                PENDING[tok][0].set()
            return self._send(200, "application/json", b'{"ok":true}')
        self._send(404, "text/plain", "not found")


if __name__ == "__main__":
    print(f"nightshift-web on :{PORT} -> model {LLM_URL}")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
