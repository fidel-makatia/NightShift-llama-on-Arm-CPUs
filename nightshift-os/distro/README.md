# NightShift OS — an agentic Linux distribution (Arm, GPU-free)

A bootable **aarch64 Linux distribution with a local Llama and an operating agent baked in as
first-class system services.** You don't administer NightShift OS with a shell first — you talk to
it. The model runs on the machine's own CPU; nothing leaves the box.

Built and **proven to boot** on an Azure Cobalt 100 VM (Neoverse-N2). This directory is the
reproducible builder, not a prebuilt blob.

## What's baked in

| Layer | What it is |
|---|---|
| Base | Ubuntu `noble` arm64, `--variant=minbase` (glibc matches modern Arm build hosts) |
| Engine | `llama.cpp` server at `/opt/nightshift`, loopback-only, KleidiAI i8mm |
| Model | a real Llama (default **Llama-3.2-3B Q4_0**) baked to `/opt/nightshift/models/llama.gguf` — works offline on first boot |
| Agent | `nsh` in `PATH` — natural-language ops, safe-by-default, audited |
| Identity | `/etc/os-release` = *NightShift OS*, hostname `nightshift`, agentic MOTD |
| Services | `nightshift-llm` (the model) · `nightshift-web` (browser console :8088) · `nightshift-boot-health` (agent writes a live health line into the MOTD at boot) |
| GUI | web console at `http://<host>:8088` — chat the OS, watch diagnostics stream live, approve/deny state changes in-browser, live disk/mem/load strip |
| Login | interactive shells are greeted and told how to drive the box in English |

## Build it

```bash
sudo ./build-nightshift-os.sh          # -> /data/nsos/rootfs   (Ubuntu noble base)
# knobs: ROOT= SUITE= MIRROR= MODEL_SRC= LLAMA_BIN_DIR= NSH_SRC=
```

## Boot it

```bash
# real systemd init in a container (proven path):
sudo systemd-nspawn -D /data/nsos/rootfs --boot -M nsos --bind-ro=/etc/resolv.conf

# for bare-metal / a cloud VHD, hand the rootfs to mkosi or a disk-image step (adds kernel +
# bootloader). The rootfs is a standard Ubuntu tree, so `machinectl import-tar`, an OCI image,
# or an Azure/Graviton golden image are all straightforward downstream targets.
```

## Proven working (captured on the VM)

Booted with `systemd-nspawn --boot`; inside the running distro:

```
$ systemctl is-active nightshift-llm         -> active
$ curl -s -o /dev/null -w '%{http_code}' :8080/health   -> 200     # baked Llama serving

$ nsh "What is the hostname, OS name, and how much memory is free? One sentence."
[1] To gather the info, I will use hostname, lsb_release, and free.
    read $ hostname; lsb_release -d; free -h
    nightshift
    Mem: 660Gi total, 394Gi free ...
    /bin/bash: lsb_release: command not found          # <- the agent hit an error
[2] lsb_release is not available, so I will skip it and use free -h.
    read $ free -h
    Mem: 660Gi total, 394Gi free ...
answer: The hostname is nightshift, the OS is Linux, and ~59% of memory (394Gi/660Gi) is free.
```

Note step [2]: the agent recovered from a failed command on its own — multi-step reasoning, live,
on a 3B Llama running on the CPU inside its own OS.

## Honest scope

- **Proven boot = `systemd-nspawn`** (real systemd, real services). A full bare-metal disk image
  (UEFI + kernel + bootloader) is one downstream step (mkosi/disk-image); the rootfs is standard so
  that step is mechanical, but it is not yet included here.
- Default base is Ubuntu noble so a host-built `llama.cpp` runs as-is; a fully portable build would
  compile `llama.cpp` **inside** the image (add `build-essential cmake` to the debootstrap include
  and a chroot build step).
- The flagship model (Llama-4-Maverick Q4_0) is an opt-in swap of the baked model — a 3B ships by
  default so the image stays small and boots offline.
- This is a working invention demo, not a hardened product: the agent's safety is the `nsh`
  allow/deny classifier + audit log, not yet a sandboxed policy engine.
