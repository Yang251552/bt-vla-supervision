"""Figure 1 "stop ladder" from results/results.json: per stop mode, what happens after the interrupt, in control
steps (medians over the counted non-void interrupts of all three phases).

    python scripts/make_figures.py results/results.json figures/stop_ladder.png

Bad input (results marked invalid, a missing field, a count / latency / path that is not a finite number >= 0, a mode
without episodes): exit 1 with a one-line message, nothing written.
"""
import json
import math
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
STAGE = {"old-goal VLA actions": "#2a78d6", "settling to standstill": "#eb6834",
         "retract (BT only)": "#1baf7a", "base back-off (BT only)": "#eda100"}
MODES = [("wait_chunk", "wait for chunk"), ("language_stop", 'language "stop"'),
         ("truncate_hold", "truncate + hold"), ("bt_safe_abort", "BT safe-abort")]


def bad(msg):
    sys.exit(f"make_figures: {sys.argv[1]}: {msg}")


try:
    out = json.load(open(sys.argv[1]))
    eps = [e for e in out["episodes"] if e["counted"] and e["phase"] != "none" and not e["seed_phase_void"]
           and e["seed"] in out["summary"]["paired_seeds"] and e.get("t_I") is not None]
    W = out["meta"]["protocol"]["window_W"]
except (OSError, ValueError, LookupError, TypeError) as x:
    bad(f"missing or malformed field: {x!r}")
if "invalid" in out:
    bad(f"results marked invalid: {out['invalid']}")
num = lambda x: type(x) in (int, float) and math.isfinite(x) and x >= 0  # noqa: E731
for e in eps:
    need = ("N_post", "d_post_cm", "L_S") if e.get("still_in_W") is True else ("N_post", "d_post_cm")
    for k in ("N_post", "d_post_cm", "L_S", "L_R", "L_B"):
        if not (num(e.get(k)) or e.get(k) is None and k not in need) or type(e.get("still_in_W")) is not bool:
            bad(f"{e.get('key')}: {k} = {e.get(k)!r}, still_in_W = {e.get('still_in_W')!r}")
if not num(W):
    bad(f"window_W = {W!r}")
for mode, _ in MODES:
    if not any(e.get("mode") == mode for e in eps):
        bad(f"no counted non-void interrupted {mode} episode")
med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731

fig, ax = plt.subplots(figsize=(8.0, 3.2), dpi=150, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for y, (mode, label) in enumerate(MODES):
    g = [e for e in eps if e["mode"] == mode]
    n_post = med([e["N_post"] for e in g])
    still = [e["L_S"] for e in g if e["still_in_W"]]
    l_s = med(still) if len(still) * 2 > len(g) else None          # median censored at W
    segs = [("old-goal VLA actions", 0, n_post)]
    if mode == "bt_safe_abort":
        l_r, l_b = med([e["L_R"] for e in g if e["L_R"] is not None]), med([e["L_B"] for e in g if e["L_B"] is not None])
        segs += [("settling to standstill", n_post, l_s), ("retract (BT only)", l_s, l_r),
                 ("base back-off (BT only)", l_r, l_b)]
    elif l_s is not None:
        segs.append(("settling to standstill", n_post, l_s))
    for name, a, b in segs:
        if b is not None and b > a:
            ax.barh(len(MODES) - 1 - y, b - a, left=a, height=0.5, color=STAGE[name], edgecolor=SURFACE, linewidth=2)
    end = max(b for _, _, b in segs if b is not None)
    d = med([e["d_post_cm"] for e in g])
    txt = f"{len(g)} interrupts · eef path {d:.1f} cm" + (
        f"\nVLA keeps control; still within W in {len(still)}/{len(g)}" if mode == "language_stop" else "")
    ax.text(end + 3, len(MODES) - 1 - y, txt, va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(MODES)), [m[1] for m in reversed(MODES)], fontsize=9, color=INK)
ax.set_xlabel("control steps after the interrupt (20 Hz; median)", fontsize=9, color=INK2)
ax.set_xlim(0, 330)
ax.axvline(W, color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
ax.text(W, len(MODES) - 0.45, f"interrupt clears (W = {W})", fontsize=7.5, color=MUTED, ha="center")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(axis="x", colors=INK2, labelsize=8)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in STAGE.values()], labels=list(STAGE),
          loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=4, frameon=False, fontsize=8, labelcolor=INK2)
fig.tight_layout()
fig.savefig(sys.argv[2], facecolor=SURFACE)
print(f"wrote {sys.argv[2]}")
