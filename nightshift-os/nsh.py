#!/usr/bin/env python3
"""nsh — NightShift Shell: a natural-language agent that operates an Arm Linux server.

An on-prem, GPU-free SRE/sysadmin agent backed by a local llama.cpp server (flagship
Llama by default). You describe intent; nsh investigates and acts by running shell
commands through a ReAct loop. Safe by default:
  - read-only commands (ps, df, journalctl, systemctl status, ...) run automatically
  - anything that mutates state asks for confirmation
  - a hard deny-list is never run
  - every action is appended to an audit log

stdlib only — no pip install. Point it at any OpenAI-compatible endpoint.

Usage:
  nsh "why is the root disk filling up?"        # one task
  nsh                                            # interactive
Env:
  NIGHTSHIFT_LLM_URL   (default http://127.0.0.1:8080)
  NIGHTSHIFT_MODEL     model alias (default "nightshift")
  NIGHTSHIFT_YES=1     auto-approve mutations (CI/headless; use with care)
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

LLM_URL = os.environ.get("NIGHTSHIFT_LLM_URL", "http://127.0.0.1:8080").rstrip("/")
MODEL = os.environ.get("NIGHTSHIFT_MODEL", "nightshift")
AUTO_YES = os.environ.get("NIGHTSHIFT_YES") == "1"
AUDIT = os.environ.get("NIGHTSHIFT_AUDIT", os.path.expanduser("~/.nightshift/audit.log"))
MAX_STEPS = int(os.environ.get("NIGHTSHIFT_MAX_STEPS", "12"))
CMD_TIMEOUT = int(os.environ.get("NIGHTSHIFT_CMD_TIMEOUT", "30"))

# --- safety classification -------------------------------------------------
# first token (basename) that only reads state -> auto-approved
READ_ONLY = {
    "ls", "cat", "head", "tail", "less", "wc", "stat", "file", "find", "grep", "egrep",
    "awk", "sed", "cut", "sort", "uniq", "tr", "df", "du", "free", "ps", "top", "htop",
    "uptime", "who", "w", "id", "uname", "hostname", "date", "env", "printenv", "lscpu",
    "lsblk", "lsmem", "lsof", "ip", "ss", "netstat", "ping", "dig", "nslookup", "journalctl",
    "dmesg", "vmstat", "iostat", "mpstat", "sar", "nproc", "getent", "sysctl", "readlink",
    "realpath", "basename", "dirname", "echo", "true", "pwd", "cmp", "diff", "md5sum",
    "sha256sum", "column", "numfmt", "systemd-analyze", "tree",
}
# systemctl/apt subcommands that only read
SAFE_SUBCMD = {"systemctl": {"status", "list-units", "list-unit-files", "is-active",
                             "is-enabled", "show", "cat", "list-timers", "list-dependencies"},
               "apt": {"list", "show", "policy", "search"},
               "apt-get": {"-s"}, "docker": {"ps", "images", "logs", "inspect", "stats"},
               "git": {"status", "log", "diff", "show", "branch", "remote"}}
# never run, no matter what
DENY = [r"\brm\s+-rf\s+/(?!\w)", r"\bmkfs", r"\bdd\b.*\bof=/dev/", r":\(\)\s*\{", r"\bshutdown\b",
        r"\breboot\b", r">\s*/dev/sda", r"\bchmod\s+-R\s+777\s+/", r"\bchown\s+-R\b.*\s/\s"]


def classify(cmd):
    c = cmd.strip()
    for pat in DENY:
        if re.search(pat, c):
            return "deny"
    # take the first pipeline stage's leading executable
    first = re.split(r"[|;&]", c)[0].strip()
    toks = first.split()
    if not toks:
        return "mutate"
    exe = os.path.basename(toks[0])
    if exe in SAFE_SUBCMD:
        sub = toks[1] if len(toks) > 1 else ""
        return "read" if sub in SAFE_SUBCMD[exe] else "mutate"
    # a redirection that writes a file is a mutation
    if re.search(r"(?<![0-9])>>?[^&]", c) or re.search(r"\btee\b", c):
        return "mutate"
    return "read" if exe in READ_ONLY else "mutate"


def audit(kind, detail):
    try:
        os.makedirs(os.path.dirname(AUDIT), exist_ok=True)
        with open(AUDIT, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{kind}\t{detail}\n")
    except OSError:
        pass


def run_shell(cmd):
    try:
        p = subprocess.run(["/bin/bash", "-lc", cmd], capture_output=True, text=True,
                           timeout=CMD_TIMEOUT)
        out = (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        out = f"[timed out after {CMD_TIMEOUT}s]"
    except Exception as e:  # noqa: BLE001
        out = f"[error: {e}]"
    if len(out) > 4000:
        out = out[:4000] + "\n...[truncated]"
    return out or "[no output]"


# --- LLM ---------------------------------------------------------------------
SYSTEM = """You are NightShift, an autonomous agent that operates an Arm Linux server on behalf \
of an administrator. You accomplish tasks by reasoning and running shell commands.

Respond with a SINGLE JSON object and nothing else, in one of two forms:
  {"thought": "<brief reasoning>", "action": "shell", "command": "<one shell command>"}
  {"thought": "<brief reasoning>", "action": "final", "answer": "<answer for the admin>"}

Rules:
- One command at a time. Prefer read-only diagnostics first (df, du, ps, journalctl, systemctl status).
- Only propose a state-changing command when the task clearly requires it; the admin will be asked
  to confirm it, so explain why in "thought".
- After each command you will receive an OBSERVATION with its output. Use it, then continue.
- When you have enough to answer, use action "final" with a concise, correct answer.
- Be precise and concrete; cite the numbers you observed."""


def llm(messages):
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": 0,
                       "max_tokens": 512, "stream": False}).encode()
    req = urllib.request.Request(LLM_URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]


def parse_action(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # tolerate trailing prose / minor issues by grabbing the first balanced object
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


C = {"g": "\033[92m", "c": "\033[96m", "y": "\033[93m", "r": "\033[91m",
     "d": "\033[90m", "b": "\033[1m", "x": "\033[0m"} if sys.stdout.isatty() else \
    {k: "" for k in "gcyrdbx"}


def confirm(cmd):
    if AUTO_YES:
        print(f"{C['y']}  [auto-approved mutation]{C['x']}")
        return True
    try:
        ans = input(f"{C['y']}  run this mutating command? [y/N] {C['x']}").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def solve(task):
    print(f"{C['b']}NightShift{C['x']} {C['d']}· {LLM_URL} · model={MODEL}{C['x']}")
    print(f"{C['b']}task:{C['x']} {task}\n")
    audit("task", task)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    for step in range(1, MAX_STEPS + 1):
        try:
            reply = llm(messages)
        except Exception as e:  # noqa: BLE001
            print(f"{C['r']}LLM error: {e}{C['x']}  (is the model server up at {LLM_URL}?)")
            return 1
        act = parse_action(reply)
        if not act:
            print(f"{C['r']}could not parse a valid action; raw:{C['x']} {reply[:300]}")
            return 1
        messages.append({"role": "assistant", "content": json.dumps(act)})
        thought = act.get("thought", "")
        if act.get("action") == "final":
            print(f"{C['g']}{C['b']}answer:{C['x']} {act.get('answer', '').strip()}")
            audit("final", act.get("answer", "")[:400])
            return 0
        cmd = (act.get("command") or "").strip()
        if not cmd:
            messages.append({"role": "user", "content": "OBSERVATION: no command given; retry."})
            continue
        kind = classify(cmd)
        if thought:
            print(f"{C['d']}[{step}] {thought}{C['x']}")
        tag = {"read": C['c'] + 'read', "mutate": C['y'] + 'MUTATE', "deny": C['r'] + 'DENIED'}[kind]
        print(f"    {tag}{C['x']} $ {cmd}")
        if kind == "deny":
            obs = "[BLOCKED by NightShift safety policy — command is on the hard deny-list]"
            audit("deny", cmd)
        elif kind == "mutate" and not confirm(cmd):
            obs = "[admin declined to run this command]"
            audit("declined", cmd)
        else:
            audit(kind, cmd)
            obs = run_shell(cmd)
            print(f"{C['d']}" + "\n".join("    " + l for l in obs.splitlines()[:12]) + C['x'])
            if len(obs.splitlines()) > 12:
                print(f"{C['d']}    ...{C['x']}")
        messages.append({"role": "user", "content": "OBSERVATION:\n" + obs})
    print(f"{C['y']}reached max steps ({MAX_STEPS}) without a final answer.{C['x']}")
    return 2


def main():
    if len(sys.argv) > 1:
        return solve(" ".join(sys.argv[1:]))
    print(f"{C['b']}NightShift Shell{C['x']} — natural-language ops for this server. Ctrl-C to exit.")
    while True:
        try:
            task = input(f"{C['g']}nsh>{C['x']} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if task in ("exit", "quit"):
            return 0
        if task:
            solve(task)
            print()


if __name__ == "__main__":
    sys.exit(main())
