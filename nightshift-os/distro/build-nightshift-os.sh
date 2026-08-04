#!/usr/bin/env bash
# build-nightshift-os.sh — build NightShift OS, an agentic aarch64 Linux distribution
# with a local Llama and the `nsh` agent baked in as first-class system services.
#
# Produces a Debian-based arm64 rootfs where, on boot:
#   * nightshift-llm.service  — a loopback llama.cpp server on the baked-in Llama model
#   * nightshift-boot-health  — the agent writes a system-health summary to the MOTD
#   * logging in greets you with the agent and how to drive the machine in English
#
# Boot it directly with:   sudo systemd-nspawn -D <rootfs> --boot
# Or hand the rootfs to mkosi / a disk-image step for bare-metal / cloud (see mkosi/).
#
# Requires: debootstrap, systemd-container, a prebuilt llama.cpp, a GGUF model.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ROOT:-/data/nsos/rootfs}"
# Ubuntu noble (24.04) base: glibc 2.39 matches modern Arm build hosts (Cobalt/Graviton),
# so a host-built llama.cpp runs as-is. Override SUITE/MIRROR for a Debian base (then build
# llama.cpp in-image for glibc portability).
SUITE="${SUITE:-noble}"
MIRROR="${MIRROR:-http://ports.ubuntu.com/ubuntu-ports}"
COMPONENTS="${COMPONENTS:-main,universe}"
MODEL_SRC="${MODEL_SRC:-/data/nsos-assets/Llama-3.2-3B-Instruct-Q4_0.gguf}"
LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-/data/llama-k3/build/bin}"
NSH_SRC="${NSH_SRC:-$HERE/../nsh.py}"
VERSION="0.1"

say(){ printf "\033[1;36m[build]\033[0m %s\n" "$*"; }
[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }

say "1/6  base arm64 rootfs (debootstrap $SUITE)"
if [ ! -e "$ROOT/etc/os-release" ]; then
  debootstrap --arch=arm64 --variant=minbase --components="$COMPONENTS" \
    --include=systemd,systemd-sysv,dbus,python3,libgomp1,ca-certificates,curl,less,procps,iproute2,nano \
    "$SUITE" "$ROOT" "$MIRROR"
fi

say "2/6  bake the engine (llama.cpp runtime + model) and the agent"
install -d -m0755 "$ROOT/opt/nightshift/bin" "$ROOT/opt/nightshift/lib" "$ROOT/opt/nightshift/models"
install -m0755 "$LLAMA_BIN_DIR/llama-server" "$ROOT/opt/nightshift/bin/llama-server"
cp -a "$LLAMA_BIN_DIR/"lib*.so* "$ROOT/opt/nightshift/lib/"
install -m0644 "$MODEL_SRC" "$ROOT/opt/nightshift/models/llama.gguf"
install -D -m0755 "$NSH_SRC" "$ROOT/usr/local/bin/nsh"
echo "/opt/nightshift/lib" > "$ROOT/etc/ld.so.conf.d/nightshift.conf"

say "3/6  OS identity + branding"
cat > "$ROOT/etc/os-release" <<EOF
NAME="NightShift OS"
PRETTY_NAME="NightShift OS $VERSION (agentic · Arm · GPU-free)"
ID=nightshift
ID_LIKE=debian
VERSION="$VERSION"
HOME_URL="https://github.com/fidel-makatia/NightShift-llama-on-Arm-CPUs"
EOF
echo nightshift > "$ROOT/etc/hostname"
cat > "$ROOT/etc/motd" <<'EOF'

  N I G H T S H I F T   O S   ·  agentic · Arm · GPU-free
  ---------------------------------------------------------
  A local Llama runs this machine. Talk to it:

      nsh "why is the root disk filling up?"
      nsh "audit listening ports and flag anything unexpected"
      nsh                       # interactive

  Everything stays on this box. Actions are gated and audited.
EOF

say "4/6  first-class system services"
cat > "$ROOT/etc/systemd/system/nightshift-llm.service" <<'EOF'
[Unit]
Description=NightShift local Llama server (GPU-free, Arm)
After=network.target
[Service]
Environment=LD_LIBRARY_PATH=/opt/nightshift/lib
ExecStart=/opt/nightshift/bin/llama-server -m /opt/nightshift/models/llama.gguf \
  -t 4 -c 8192 --host 127.0.0.1 --port 8080 --jinja --alias nightshift
Restart=on-failure
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF

# boot-time agent health check -> refreshes the MOTD with a live summary
cat > "$ROOT/etc/systemd/system/nightshift-boot-health.service" <<'EOF'
[Unit]
Description=NightShift boot-time health summary (agent)
After=nightshift-llm.service
Wants=nightshift-llm.service
[Service]
Type=oneshot
ExecStart=/usr/local/bin/nightshift-bootcheck
[Install]
WantedBy=multi-user.target
EOF

cat > "$ROOT/usr/local/bin/nightshift-bootcheck" <<'EOF'
#!/usr/bin/env bash
# wait for the model, then let the agent summarize system health into the MOTD banner
for i in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null)" = "200" ] && break
  sleep 5
done
S="$(NIGHTSHIFT_YES=0 nsh 'In one sentence, summarize this server\'s health: disk, memory, and load.' 2>/dev/null | sed -n 's/^answer: //p')"
[ -n "$S" ] && printf '\n  agent health check: %s\n' "$S" >> /etc/motd || true
EOF
chmod 0755 "$ROOT/usr/local/bin/nightshift-bootcheck"

# agent greeting on interactive login
cat > "$ROOT/etc/profile.d/nightshift.sh" <<'EOF'
if [ -n "$PS1" ]; then
  export NIGHTSHIFT_LLM_URL="http://127.0.0.1:8080"
  echo "  (this is NightShift OS — type: nsh \"<what you want done>\")"
fi
EOF

say "5/6  enable services in the image"
chroot "$ROOT" systemctl enable nightshift-llm.service nightshift-boot-health.service >/dev/null 2>&1 || {
  ln -sf /etc/systemd/system/nightshift-llm.service "$ROOT/etc/systemd/system/multi-user.target.wants/nightshift-llm.service"
  ln -sf /etc/systemd/system/nightshift-boot-health.service "$ROOT/etc/systemd/system/multi-user.target.wants/nightshift-boot-health.service"
}
# root auto-login on the container console (so `nspawn --boot` drops you in)
mkdir -p "$ROOT/etc/systemd/system/console-getty.service.d"
cat > "$ROOT/etc/systemd/system/console-getty.service.d/autologin.conf" <<'EOF'
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin root --noclear --keep-baud console 115200,38400,9600 $TERM
EOF
chroot "$ROOT" bash -c 'passwd -d root' >/dev/null 2>&1 || true

say "6/6  done -> $ROOT"
du -sh "$ROOT" 2>/dev/null || true
cat <<EOF

NightShift OS rootfs built. Boot it:
  sudo systemd-nspawn -D $ROOT --boot            # boots real systemd + the agent stack
Inside, once nightshift-llm is up:
  nsh "what's using the most memory right now?"
EOF
