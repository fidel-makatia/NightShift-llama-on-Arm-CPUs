# NightShift OS — an on-prem agentic layer for Arm Linux

**Turn any Arm64 Linux server into an AI-operated system. GPU-free, air-gapped, one command to install.**

NightShift OS is not a new kernel — it's a thin, honest layer that makes a stock Linux server
*operable in natural language* by an agent that runs entirely on the box's own CPU:

- a **local flagship-Llama server** (llama.cpp, KleidiAI i8mm on Arm) — no GPU, nothing leaves the machine;
- **`nsh`**, a natural-language ops agent that investigates and acts by running shell commands through a
  reasoning loop — **safe by default** (read-only auto-runs, mutations need confirmation, dangerous
  commands are blocked) and **fully audited**.

```bash
git clone https://github.com/fidel-makatia/NightShift-llama-on-Arm-CPUs
cd NightShift-llama-on-Arm-CPUs/nightshift-os
sudo ./install.sh                 # flagship: Llama-4-Maverick Q4_0 (KleidiAI i8mm)
# ... or a lighter, snappier tier:
sudo ./install.sh --model qwen30b # Qwen3-30B-A3B (3B active) — ~44 tok/s on Cobalt 100

nsh "why is the root disk filling up?"
nsh "audit which ports are listening and flag anything unexpected"
nsh                                # interactive
```

## Why this matters for enterprises

- **Data sovereignty / air-gap.** The model and every prompt stay on your hardware. Regulated,
  classified, and disconnected environments can have a capable agent with zero egress.
- **No GPU, no cloud bill.** It runs on the Arm CPUs enterprises already buy for general compute
  (Azure Cobalt, AWS Graviton, Ampere). One `apply`, and every server has a resident SRE.
- **Safe by construction.** The agent cannot mutate state without an explicit `y`, a hard deny-list
  blocks catastrophic commands, and every action is written to an audit log for review.
- **Right-sized models.** On CPU, *active* parameters set the speed — a 3B-active MoE answers at
  ~44 tok/s here; the 400B Maverick flagship gives higher quality when you want it. Pick per workload.

## What it actually does (measured, on an Azure Cobalt 100 / Neoverse-N2 VM)

Real transcript ([`DEMO.md`](DEMO.md)) — the agent auto-runs read-only diagnostics and summarizes:

```
task: How much disk and memory is in use, and the top 3 processes by memory?
[1] I will check disk with df -h, memory with free -m, top procs with ps aux --sort=-%mem.
    read $ df -h; free -m; ps aux --sort=-%mem | head -4
    /dev/sda 4.0T 1.7T 2.4T 41% /data   ·   Mem: 675879 used 25152 free 417458 ...
answer: 7% root disk, 41% on /data; 25.1 GB of 675 GB RAM used; top process llama-server
        (34.9 GB). The system is healthy with no resource constraints.
```

Safety, same tool:

```
task: Create a marker file at /tmp/ns_marker.txt containing today's date.
    MUTATE $ date > /tmp/ns_marker.txt
    run this mutating command? [y/N]        <- nothing happens without approval

task: Free up space with rm -rf /
    -> agent refuses; hard deny-list would block it regardless.
```

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md). In one line: `systemd` runs a loopback-only llama.cpp
server; `nsh` drives a ReAct loop (`thought → shell action → observation → …`) against it, with a
classifier that gates every command and an append-only audit log.

## Components

| File | What it is |
|---|---|
| `install.sh` | one-command installer: builds llama.cpp (KleidiAI), fetches the model, installs the systemd service + `nsh` |
| `nsh.py` | the agent — stdlib-only Python, model-agnostic (any OpenAI-compatible endpoint), safe-by-default |
| `systemd/nightshift-llm.service` | the local model server as a managed, loopback-bound service |
| `DEMO.md` | real captured transcripts from the VM |

## Honest scope

This is a working MVP, not a finished product. `nsh` operates via shell with a conservative
allow/deny classifier — solid for a supervised SRE assistant, not yet a substitute for a hardened,
sandboxed policy engine (RBAC, seccomp, per-command dry-run simulation) that a production rollout
would add. The model runs at CPU speed: great for investigation and automation, not for high-QPS
serving. Claims here are what was measured on one Cobalt 100 VM; nothing more.
