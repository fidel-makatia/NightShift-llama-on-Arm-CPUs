# NightShift OS — architecture

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Arm64 Linux server (Cobalt 100 / Graviton / Ampere) — no GPU     │
  │                                                                   │
  │   admin ── nsh "why is disk full?" ─┐                             │
  │                                     │  ReAct loop                 │
  │                          ┌──────────▼───────────┐                 │
  │                          │  nsh  (stdlib Python) │                 │
  │                          │  • prompt → action    │                 │
  │                          │  • classify(cmd)      │                 │
  │                          │  • run / confirm / deny                │
  │                          │  • audit log          │                 │
  │                          └───┬───────────────▲───┘                 │
  │            /v1/chat/completions│              │ OBSERVATION         │
  │                          ┌─────▼──────────────┴───┐                 │
  │                          │ nightshift-llm (systemd)│                │
  │                          │ llama.cpp server        │                │
  │                          │ flagship Llama, KleidiAI│                │
  │                          │ i8mm · loopback :8080   │                │
  │                          └─────────────────────────┘                │
  │                                     ▲                               │
  │                          shell commands (bash -lc)                  │
  │                          run against the real system                │
  └─────────────────────────────────────────────────────────────────┘
        nothing leaves the box · model + prompts + data stay local
```

## The agent loop (`nsh`)

A **ReAct** loop with a strict JSON protocol, so it works with any instruct model (no dependence on
a server-side tool-calling template):

1. `nsh` sends the system prompt + task to the local model.
2. The model replies with one JSON object:
   `{"thought": ..., "action": "shell", "command": ...}` or `{"action": "final", "answer": ...}`.
3. `nsh` **classifies** the command, runs it (or asks / blocks), captures output, and appends it as an
   `OBSERVATION`.
4. Repeat until `final` (or a max-steps cap). Everything is appended to an audit log.

Temperature 0 for determinism; output truncation + per-command timeout keep a runaway model bounded.

## The safety classifier

Every proposed command is bucketed before it can run:

| Class | Rule | Behavior |
|---|---|---|
| **read** | leading executable in a read-only allow-list (`df`, `ps`, `journalctl`, `systemctl status`, …); safe sub-commands for `systemctl`/`apt`/`docker`/`git` | runs automatically |
| **mutate** | anything else, or a redirect / `tee` that writes | requires interactive `y` (or `NIGHTSHIFT_YES=1` in headless/CI) |
| **deny** | regex hard-list: `rm -rf /`, `mkfs`, `dd of=/dev/…`, fork-bombs, `shutdown`/`reboot`, `chmod -R 777 /`, … | never runs |

Conservative by design: unknown ⇒ mutate ⇒ confirmation. The classifier inspects the first pipeline
stage's leading executable and scans the whole line for writes and deny patterns.

## Why Arm + this model choice

- **CPU-only, bandwidth-bound.** Generation speed on CPU tracks *active* parameters, not total size.
  A 3B-active MoE (Qwen3-30B) answers at ~44 tok/s on Cobalt 100; the 400B/17B-active Maverick
  flagship trades speed for quality. `install.sh --model` picks the tier.
- **KleidiAI i8mm** accelerates Q4_0 on Neoverse, which is why the flagship default is Maverick **Q4_0**
  (fast *and* higher quality than a 2-bit quant).
- **Loopback only.** The model server binds `127.0.0.1`; the agent is the only client. No inbound
  surface, no egress.

## Extending it (production hardening — not yet built)

- Replace the shell classifier with a policy engine (RBAC per operator, seccomp-sandboxed execution,
  per-command dry-run simulation before apply).
- Structured tools (typed `read_file`, `service_action`, `pkg_query`) instead of raw shell.
- A REST daemon + web console on top of the same loop for multi-user ops.
- Fleet mode: one control node fanning tasks to many servers, each running its own local model.
