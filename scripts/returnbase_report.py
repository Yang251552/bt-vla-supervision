"""ReturnBase check per BT episode log (acceptance fixed before the pre-freeze instance check):
base never past the pre-abort pose along its heading over [U, done]; max 3-step mean contact <= F over
[U, done + 50] (ReturnBase and the start of re-entry); done reached with planar error < 0.10 m (yaw error printed).
Exit 1 if any episode fails one of them.

    python3 scripts/returnbase_report.py runs/c5_rb/*_bt_safe_abort_r0.jsonl
"""
import json
import sys

import numpy as np

F_LIMIT, PAST_TOL, POS_TOL = 50.0, 0.0, 0.10
bad = 0
for path in sys.argv[1:]:
    recs = [json.loads(x) for x in open(path) if x.strip()]
    ev = {r["ev"]: r for r in recs if r["ev"] in ("I", "U", "return_base_done", "reentry_goal", "episode_end")}
    if "U" not in ev:
        print(f"{path}: no interrupt window (I {'I' in ev}), skipped")
        continue
    done = ev.get("return_base_done")
    t1 = done["t"] if done else ev.get("reentry_goal", ev["episode_end"])["t"]
    steps = [r for r in recs if r["ev"] == "step" and ev["U"]["t"] <= r["t"] <= t1 + 50]
    f = np.array([r["force"] for r in steps] or [0.0])
    fmax = float(np.convolve(f, np.ones(3) / 3, mode="valid").max()) if len(f) >= 3 else float(f.max())
    past = err = yaw = None
    if done:
        tgt, base = np.asarray(done["target"]), np.asarray(done["base"])
        h = np.array([np.cos(tgt[2]), np.sin(tgt[2])])
        past = max(float((np.asarray(r["base"][:2]) - tgt[:2]) @ h) for r in steps if r["t"] <= t1)
        err = float(np.linalg.norm(base[:2] - tgt[:2]))
        yaw = float(np.degrees((base[2] - tgt[2] + np.pi) % (2 * np.pi) - np.pi))
    ok = done is not None and past <= PAST_TOL and err < POS_TOL and fmax <= F_LIMIT
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'} {path.split('/')[-1]}: phase@I {ev['I']['phase']}, ReturnBase "
          f"{t1 - ev['U']['t']} steps, done {done is not None}, past {past if past is None else round(past, 3)} m, "
          f"error at done {err if err is None else round(err, 3)} m / {yaw if yaw is None else round(yaw, 2)} deg, "
          f"max contact {fmax:.1f} N")
sys.exit(1 if bad else 0)
