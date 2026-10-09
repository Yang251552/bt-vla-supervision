"""Latency segments and criteria 1-3 for ONE episode, computed only from its event log."""
from __future__ import annotations

import math

import numpy as np

from .contracts import Ev, ProtocolConfig

RE_MOTION_M = 0.02   # analysis threshold (eval/prereg.json), not a run parameter: kept out of the config hash


def _first(recs, ev, after=None, **match):
    for i, r in enumerate(recs):
        if r["ev"] == ev and (after is None or r["t"] >= after) and \
                all(r.get(k) == v for k, v in match.items()):
            return i, r
    return None, None


def _t(r):
    return None if r is None else r["t"]


def _sub(a, b):
    return None if a is None or b is None else a - b


def _finite(*xs):
    return all(type(x) in (int, float) and math.isfinite(x) for x in xs)


def episode_metrics(recs: list[dict], cfg: ProtocolConfig) -> dict:
    hdr = recs[0]
    if hdr["ev"] != Ev.EPISODE or sum(r["ev"] == Ev.EPISODE for r in recs) != 1:
        raise ValueError("expected exactly one episode per log file")
    step_recs = [r for r in recs if r["ev"] == Ev.STEP]
    steps = {r["t"]: r for r in step_recs}
    # malformed log (ValueError): a control step without its STEP record, non-bool grasped, non-finite numbers
    if sorted(steps) != list(range(len(steps))):
        raise ValueError("STEP records do not cover t = 0..last without gaps")
    bad = [r["t"] for r in step_recs if type(r["grasped"]) is not bool
           or not _finite(r["force"], r["lift"], *r["eef"], *r.get("base", ()))]
    if bad:
        raise ValueError(f"STEP with non-bool grasped or non-finite eef/base/force/lift at t = {bad[:5]}")
    qs = [r for r in recs if r["ev"] == Ev.Q]
    fences = {r["gen"]: r["t"] for r in reversed(qs)}   # the first Q of a generation is its fence (c1's Q below)
    # fence violations anywhere in the episode: offers refused + applied old-gen actions after Q (every STEP record)
    # + one per duplicate STEP time or duplicate Q (malformed log: two commands for one step, a generation fenced twice)
    applied_after_q = sum(1 for r in step_recs if r["src"] == "vla"
                          and r["gen"] in fences and r["t"] >= fences[r["gen"]])
    violations = (sum(1 for r in recs if r["ev"] == Ev.VIOLATION) + applied_after_q + len(step_recs) - len(steps)
                  + len(qs) - len(fences))
    _, succ = _first(recs, Ev.SUCCESS)
    m = {"cfg_hash": hdr.get("cfg_hash"), "seed": hdr.get("seed"), "mode": hdr.get("mode"),
         "phase": hdr.get("phase"), "violations": violations, "success_t": _t(succ),
         "completed": succ is not None}
    _, inj = _first(recs, Ev.I)
    asked = hdr.get("phase") if hdr.get("phase") not in (None, "none") else None
    if inj is None:
        if asked:   # phase not entered, or episode over before entry + k: no interrupt sample
            m.update(void=True, void_reason="no_injection")
        return m
    tI = inj["t"]
    gen = max((r["gen"] for r in recs if r["ev"] == Ev.GOAL_ACCEPTED and r["t"] <= tI), default=None)
    _, req = _first(recs, Ev.R, after=tI)
    _, acc = _first(recs, Ev.A, after=tI, gen=gen)
    iq, q = _first(recs, Ev.Q, gen=gen)
    ic, canc = _first(recs, Ev.CANCELED, gen=gen)
    _, halt = _first(recs, Ev.H, after=tI)
    _, retr = _first(recs, Ev.RETRACTED, after=tI)
    _, abort = _first(recs, Ev.B, after=tI)
    _, clear = _first(recs, Ev.U, after=tI)
    if clear is not None and clear["t"] != tI + cfg.window_W:
        raise ValueError(f"U = {clear['t']}, not t_I + W = {tI + cfg.window_W}")
    _, reent = _first(recs, Ev.REENTRY_GOAL, after=tI)
    vla_post = [t for t, r in steps.items() if r["src"] == "vla" and r["gen"] == gen and t >= tI]
    vla_all = [t for t, r in steps.items() if r["src"] == "vla" and r["gen"] == gen]
    chunk_starts = [r["t"] for r in recs if r["ev"] == Ev.FEEDBACK and r["gen"] == gen and r["t"] <= tI]

    # standstill counts only inside W ([t_I, U], or end of log without U; never past t_I + W);
    # d_post: eef path length from t_I to min(H, that window end)
    t_U = min(_t(clear) if clear is not None else max(steps), tI + cfg.window_W)
    still_in_W = halt is not None and halt["t"] <= t_U
    t_end = min(_t(halt), t_U) if halt else t_U
    path = [np.asarray(steps[t]["eef"]) for t in range(tI, t_end + 1) if t in steps]
    d_post = float(sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:])))

    # criterion 1: fence within slack after R, no old-generation command after Q, CANCELED after Q
    c1 = (req is not None and acc is not None and q is not None and canc is not None
          and 0 <= q["t"] - req["t"] <= cfg.fence_slack_steps and ic > iq and canc["t"] >= q["t"]
          and violations == 0)

    # criterion 2 (BT contract): retract + back-off within T, no force over F, secured object kept.
    # Window [t_I, B] for BT safe-abort; [t_I, U] (or end of log) for the other modes.
    tB = _t(abort)
    t_win = tB if tB is not None else t_U
    secured_at_I = bool(steps[tI]["grasped"] and steps[tI]["lift"] >= cfg.secured_lift_m) \
        if tI in steps else False

    def max_force(t1):   # max 3-step mean contact force over [t_I, t1]
        f = np.array([steps[t]["force"] for t in range(tI, t1 + 1) if t in steps] or [0.0])
        n = cfg.force_avg_steps
        return float((np.convolve(f, np.ones(n) / n, mode="valid") if len(f) >= n else f).max())

    def held(t1):        # object secured at t_I never released for N consecutive steps in [t_I, t1], held at t1
        if not secured_at_I:
            return True
        run, ok = 0, True
        for t in range(tI, t1 + 1):
            if t in steps:
                run = 0 if steps[t]["grasped"] else run + 1
                ok = ok and run < cfg.hysteresis_steps
        return ok and t1 in steps and steps[t1]["grasped"]

    backoff_m = abort.get("backoff_m") if abort else None
    c2 = (abort is not None and retr is not None and tI <= retr["t"] <= tB and tB - tI <= cfg.abort_budget_T
          and backoff_m >= cfg.backoff_x_m and max_force(t_win) <= cfg.force_limit_n and held(t_win))

    # shared outcomes, same window [t_I, U] for all modes. re-motion: eef leaves the standstill position
    # (H; for BT safe-abort the abort end B, since retract and back-off are intended) by > 2 cm before U;
    # undefined unless that rest step lies in [t_I, U] and was observed
    t_rest = tB if tB is not None else _t(halt)
    re_motion = None if t_rest is None or not tI <= t_rest <= t_U or t_rest not in steps else max(
        float(np.linalg.norm(np.asarray(steps[t]["eef"]) - np.asarray(steps[t_rest]["eef"])))
        for t in range(t_rest, t_U + 1) if t in steps) > RE_MOTION_M

    # criterion 3: re-entry after clearance, success within budget B of the re-entry goal
    c3 = (reent is not None and clear is not None and succ is not None
          and reent["t"] >= clear["t"] and 0 <= succ["t"] - reent["t"] <= cfg.reentry_budget_B)

    # void for all modes of this (seed, phase): interrupt landed outside the requested phase,
    # or the phase ends < hysteresis_steps after t_I
    nxt = [r["t"] for r in recs if r["ev"] == Ev.PHASE and r["t"] > tI]
    void_reason = ("phase_mismatch" if asked and inj.get("phase") != asked
                   else "phase_ends_too_soon" if nxt and nxt[0] - tI < cfg.hysteresis_steps else None)
    m.update({
        "t_I": tI, "gen": gen, "R": _t(req), "A": _t(acc), "Q": _t(q),
        "E": max(vla_all) if vla_all else None, "H": _t(halt), "RETRACTED": _t(retr),
        "B": tB, "U": _t(clear), "reentry_t": _t(reent), "canceled_t": _t(canc),
        "chunk_offset": tI - max(chunk_starts) if chunk_starts else None,
        "N_post": len(vla_post), "d_post_cm": 100.0 * d_post,
        "L_C": _sub(_t(req), tI), "L_S": _sub(_t(halt), tI) if still_in_W else None, "still_in_W": still_in_W,
        "L_R": _sub(_t(retr), tI), "L_B": _sub(tB, tI), "max_contact_avg_n": max_force(t_win),
        "backoff_m": backoff_m, "secured_at_I": secured_at_I, "obj_held": held(t_win),
        "max_contact_avg_n_W": max_force(t_U), "obj_held_W": held(t_U), "re_motion": re_motion,
        "completed_in_window": succ is not None and clear is not None and tI <= succ["t"] <= clear["t"],
        "void": void_reason is not None, "void_reason": void_reason, "phase_at_I": inj.get("phase"),
        "c1": bool(c1), "c2": bool(c2), "c3": bool(c3),
    })
    return m
