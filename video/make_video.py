#!/usr/bin/env python3
"""NightShift demo-video builder.

Renders a clean terminal screencast (typed commands + revealed output) and title/section
cards to PNG frames, then encodes to MP4 with ffmpeg. Content is real — the terminal scenes
are driven by actual captured output from the Arm VM.

Usage: python3 make_video.py <scene.json> out.mp4
scene.json = list of steps: {"card": {...}} | {"term": {...}}
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
OUT_FRAMES = Path("video/_frames")
FF = os.environ.get("FFMPEG", "ffmpeg")

# palette
BG = (13, 14, 16)
TERM_BG = (18, 20, 24)
HEADER = (32, 34, 40)
FG = (222, 224, 228)
GREEN = (94, 214, 130)
CYAN = (86, 182, 224)
DIM = (120, 124, 132)
ACCENT = (232, 120, 70)
WHITE = (245, 246, 248)


def font(size, bold=False, mono=False):
    candidates = (
        ["/System/Library/Fonts/Menlo.ttc", "Menlo.ttc"] if mono else
        (["/System/Library/Fonts/HelveticaNeue.ttc", "Helvetica.ttc", "Arial Bold.ttf"] if bold
         else ["/System/Library/Fonts/HelveticaNeue.ttc", "Helvetica.ttc", "Arial.ttf"])
    )
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


MONO = font(21, mono=True)
MONO_S = font(18, mono=True)
frame_no = 0


def emit(img):
    global frame_no
    img.save(OUT_FRAMES / f"f{frame_no:06d}.png")
    frame_no += 1


def hold(img, seconds):
    for _ in range(int(seconds * FPS)):
        emit(img.copy())


# ---------------- title / section cards ----------------
def card(step):
    title = step["title"]
    subtitle = step.get("subtitle", "")
    kicker = step.get("kicker", "")
    seconds = step.get("seconds", 2.5)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    if kicker:
        d.text((90, 250), kicker.upper(), font=font(22, bold=True), fill=ACCENT)
    tf = font(step.get("title_size", 58), bold=True)
    d.text((90, 292), title, font=tf, fill=WHITE)
    y = 292 + step.get("title_size", 58) + 24
    for line in textwrap.wrap(subtitle, 62):
        d.text((92, y), line, font=font(26), fill=DIM)
        y += 38
    # accent rule
    d.rectangle([90, 278, 90 + 120, 282], fill=ACCENT)
    hold(img, seconds)


# ---------------- terminal screencast ----------------
def _term_base(title):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    m = 40
    d.rounded_rectangle([m, m, W - m, H - m], radius=12, fill=TERM_BG)
    d.rounded_rectangle([m, m, W - m, m + 44], radius=12, fill=HEADER)
    d.rectangle([m, m + 30, W - m, m + 44], fill=HEADER)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([m + 20 + i * 26, m + 15, m + 34 + i * 26, m + 29], fill=c)
    d.text((W // 2, m + 22), title, font=MONO_S, fill=DIM, anchor="mm")
    return img


def _draw_lines(base, lines, title):
    img = base.copy()
    d = ImageDraw.Draw(img)
    x, y = 66, 108
    lh = 27
    max_lines = (H - 40 - 120) // lh
    for ln in lines[-max_lines:]:
        col = FG
        txt = ln
        if ln.startswith("$ "):
            col = GREEN
        elif ln.startswith("# "):
            col = DIM
        elif ln.strip().startswith("->") or ln.strip().startswith("→"):
            col = CYAN
        elif "SUCCESS" in ln:
            col = GREEN
        d.text((x, y), txt[:96], font=MONO, fill=col)
        y += lh
    return img


def terminal(step):
    title = step.get("title", "nightshift — Azure Arm CPU (Cobalt 100)")
    base = _term_base(title)
    lines = []
    for seg in step["segments"]:
        kind = seg[0]
        text = seg[1]
        if kind == "cmd":
            # type it out
            typed = "$ "
            for ch in text:
                typed_line = typed + ch
                img = _draw_lines(base, lines + [typed_line + "█"], title)
                emit(img)
                typed = typed_line
            lines.append("$ " + text)
            hold(_draw_lines(base, lines, title), 0.35)
        elif kind == "out":
            for oline in text.split("\n"):
                lines.append(oline)
                emit(_draw_lines(base, lines, title))
            hold(_draw_lines(base, lines, title), seg[2] if len(seg) > 2 else 0.5)
        elif kind == "pause":
            hold(_draw_lines(base, lines, title), text)
    hold(_draw_lines(base, lines, title), step.get("end_hold", 1.2))


def image_card(step):
    """Full-bleed chart/screenshot still with a caption bar."""
    seconds = step.get("seconds", 3.0)
    src = Image.open(step["path"]).convert("RGB")
    canvas = Image.new("RGB", (W, H), BG)
    scale = min((W - 120) / src.width, (H - 180) / src.height)
    nw, nh = int(src.width * scale), int(src.height * scale)
    src = src.resize((nw, nh), Image.LANCZOS)
    canvas.paste(src, ((W - nw) // 2, 60))
    d = ImageDraw.Draw(canvas)
    if step.get("caption"):
        d.text((W // 2, H - 60), step["caption"], font=font(26, bold=True), fill=WHITE, anchor="mm")
    hold(canvas, seconds)


def main():
    scene = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    OUT_FRAMES.mkdir(parents=True, exist_ok=True)
    for f in OUT_FRAMES.glob("*.png"):
        f.unlink()
    for step in scene:
        (card if "card" in step else image_card if "image" in step else terminal)(
            step.get("card") or step.get("image") or step.get("term"))
    subprocess.run([FF, "-y", "-r", str(FPS), "-i", str(OUT_FRAMES / "f%06d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    out], check=True)
    print(f"wrote {out} ({frame_no} frames, {frame_no/FPS:.1f}s)")


if __name__ == "__main__":
    main()
