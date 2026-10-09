"""Demo GIF (prereg gif_rule): the picked BT safe-abort episode next to the same seed's wait-for-chunk episode,
both aligned at the interrupt, with BT node, action status and command source overlaid from the event logs.
The interrupt hold (abort done -> interrupt cleared) plays 4x faster and is marked so; everything else is real time.

    python scripts/make_gif.py <c6_score dir> results/results.json figures/demo.gif

Bad input (results marked invalid, no pick, a missing field, event log or frames, an interrupt / abort / clear step
that is not an int >= 0): exit 1 with a one-line message, nothing written.
"""
import json
import sys
import tarfile
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
SIZE, BAND, FPS = 256, 64, 10
INK, MUTED, ALERT, SURFACE = (11, 11, 11), (82, 81, 78), (190, 50, 40), (252, 252, 251)


def bad(msg):
    sys.exit(f"make_gif: {msg}")


try:
    d, res, out = Path(sys.argv[1]), json.load(open(sys.argv[2])), Path(sys.argv[3])
    gif = res["summary"]["gif"]
    pick = gif["bt"] and gif["wait_chunk"] and (gif["bt"]["key"], gif["wait_chunk"]["key"])
    reason = gif.get("reason")
except (OSError, ValueError, LookupError, TypeError, AttributeError) as x:
    bad(f"{sys.argv[2]}: missing or malformed summary.gif: {x!r}")
if "invalid" in res:
    bad(f"{sys.argv[2]}: results marked invalid: {res['invalid']}")
if not pick:
    bad(f"no GIF pick: {reason}" if isinstance(reason, str) and reason else f"no GIF pick and no reason in {sys.argv[2]}")
font, small = ImageFont.truetype(FONT, 13), ImageFont.truetype(FONT, 11)


def load(key, label, tmp, need):
    recs = [json.loads(x) for x in open(d / f"{key}.jsonl") if x.strip()]
    with tarfile.open(d / f"frames_{key}.tgz") as tf:
        tf.extractall(tmp, filter="data")
    ev = {}
    for r in recs:
        ev.setdefault(r["ev"], r["t"])
    steps = {r["t"]: r for r in recs if r["ev"] == "step"}
    if not steps or any(type(ev.get(x)) is not int or ev[x] < 0 for x in need):
        bad(f"{key}: no steps, or no int t >= 0 for each of {need}")
    status, gen = {}, 0          # action status per step from the goal events
    for r in recs:
        if r["ev"] == "goal_accepted":
            gen = r["gen"]
        s = {"goal_accepted": f"goal {gen} ACTIVE", "canceling": f"goal {gen} CANCELING", "canceled": f"goal {gen} CANCELED",
             "goal_succeeded": f"goal {gen} SUCCEEDED"}.get(r["ev"])
        if s:
            status[r["t"]] = s
    return {"label": label, "ev": ev, "steps": steps, "status": status, "dir": Path(tmp) / f"frames_{key}",
            "tI": ev["I"], "end": max(steps)}


def panel(e, t, speed):
    t = min(t, e["end"] - e["end"] % 2)
    img = Image.new("RGB", (SIZE, SIZE + BAND), SURFACE)
    img.paste(Image.open(e["dir"] / f"{t:05d}.jpg").convert("RGB").resize((SIZE, SIZE)), (0, BAND))
    st = max((k for k in e["status"] if k <= t), default=None)
    s, ev = e["steps"].get(t, {}), e["ev"]
    inter = ev["I"] <= t < ev.get("U", 10**9)
    n_old = sum(1 for k, r in e["steps"].items() if e["tI"] <= k <= t and r["src"] == "vla" and r["gen"] == 1)
    g = ImageDraw.Draw(img)
    g.text((8, 4), e["label"], fill=INK, font=font)
    g.text((8, 22), f"t - t_I = {t - e['tI']:+d}   {e['status'][st] if st is not None else ''}", fill=INK, font=small)
    g.text((8, 37), f"BT {s.get('bt', '')} · cmd {s.get('src', '')} · old-goal actions {n_old}", fill=MUTED, font=small)
    if inter:
        g.text((SIZE - 82, 4), "INTERRUPT", fill=ALERT, font=font)
    if speed > 1:
        g.text((SIZE - 30, 22), f"x{speed}", fill=ALERT, font=small)
    if t >= e["end"] - 1 and "success" in ev:
        g.text((8, 50), "task done", fill=INK, font=small)
    return img


try:
    with tempfile.TemporaryDirectory() as tmp:
        bt = load(pick[0], "BT safe-abort", tmp, ("I", "B", "U"))
        wc = load(pick[1], "wait for chunk (same seed)", tmp, ("I",))
        hold = (bt["ev"]["B"] + 6, bt["ev"]["U"] - 6)
        rel, frames = -20, []
        while bt["tI"] + rel <= bt["end"] + 6:
            t = bt["tI"] + rel
            fast = hold[0] <= t < hold[1]
            canvas = Image.new("RGB", (2 * SIZE + 8, SIZE + BAND), SURFACE)
            canvas.paste(panel(bt, t - t % 2, 4 if fast else 1), (0, 0))
            canvas.paste(panel(wc, wc["tI"] + rel - (wc["tI"] + rel) % 2, 4 if fast else 1), (SIZE + 8, 0))
            frames.append(canvas.quantize(colors=96, method=Image.Quantize.MEDIANCUT))
            rel += 8 if fast else 2
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0, optimize=True)
except (OSError, LookupError, ValueError, TypeError, tarfile.TarError) as x:
    bad(f"cannot build the GIF from {d}: {x!r}")
print(f"wrote {out}: {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB")
