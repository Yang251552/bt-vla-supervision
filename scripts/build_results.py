"""Results: every number recomputed from the per-episode event logs of one c6_score directory with the current
analysis code (preempt/metrics.py, scripts/check_floor.py); eval/prereg.json defines every quantity. Metric fields
written into results.jsonl at run time are provisional: compared (summary.provisional_diffs), never used. Run-time
decisions (outcome, pairing) come from the results lines; pairing is re-derived from the log and compared. If they
disagree or the floor schema fails (unknown outcome, malformed line or event log), the output starts with "invalid"
(the reasons) and the exit code is 1; NaN or Infinity in the input is an error.

    python scripts/build_results.py runs/aws/<stamp>/runs/c6_score [--out results.json]

Counted episode: the last attempt of a key, if its outcome is in FINISHED and its log was analysed (as check_floor).
Summary statistics use counted episodes on paired seeds whose (seed, phase) is not void. Completion = criterion 3:
task success within B of the re-entry goal sent after U (prereg shared_outcomes.completion); success at any step is
listed as completed_any, success before U as completion_in_window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preempt.contracts import Ev, ProtocolConfig  # noqa: E402
from preempt.metrics import episode_metrics  # noqa: E402
from scripts.check_floor import FINISHED, MODES, PHASES, check  # noqa: E402

M1, M2, M3, BT = MODES


def wilson(k, n, z=1.959963984540054):
    if not n:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def rate(k, n):
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": wilson(k, n)}


def spread(xs, censored=0, cap=None):
    """median [min, max]; censored values (> cap) sort above every number."""
    v = sorted(x for x in xs if x is not None) + [math.inf] * censored
    show = lambda x: f"> {cap}" if x == math.inf else x  # noqa: E731
    return {"median": show(statistics.median(v)), "min": show(v[0]), "max": show(v[-1]), "n": len(v),
            "censored": censored, "missing": sum(x is None for x in xs)} if v else None


def max_contact(recs, t0, t1, n):
    """max n-step mean contact force over the steps in [t0, t1], as preempt.metrics does over [t_I, ...]."""
    f = np.array([r["force"] for r in recs if r["ev"] == Ev.STEP and t0 <= r["t"] <= t1])
    return float((np.convolve(f, np.ones(n) / n, mode="valid") if len(f) >= n else f).max()) if len(f) else None


def same(a, b):
    return a == b or isinstance(a, float) and isinstance(b, float) and math.isclose(a, b, rel_tol=1e-9)


def load(p: Path) -> list[dict]:   # JSON lines; NaN and +-Infinity are malformed input
    def nonfinite(c):
        raise ValueError(f"{c} in {p.name}")
    return [json.loads(x, parse_constant=nonfinite) for x in p.read_text().splitlines() if x.strip()]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def num(x):
    return "n/a" if x is None else f"{x:g}" if isinstance(x, (int, float)) else str(x)


def med(xs):
    return statistics.median(xs) if xs else None


def claim(value, ok):
    return {"value": value, "result": "not evaluable" if ok is None else "met" if ok else "not met"}


def overall(cs):
    r = [c["result"] for c in cs.values()]
    return "not met" if "not met" in r else "not evaluable" if "not evaluable" in r else "met"


def build(d: Path) -> dict:
    pre = json.loads((ROOT / "eval" / "prereg.json").read_text())
    proto, man = ROOT / "config" / "protocol.json", ROOT / "eval" / "frozen_manifest.json"
    cfg = ProtocolConfig.load(proto)
    F, W, B, nF = cfg.force_limit_n, cfg.window_W, cfg.reentry_budget_B, cfg.force_avg_steps
    lines = load(d / "results.jsonl")

    # attempts: a "started" line opens one, the key's next other line finishes it (run_episode.py)
    atts, live = [], {}
    for r in lines:
        if r.get("outcome") == "started":
            live[r["key"]] = a = {"start": r, "final": None}
            atts.append(a)
        else:
            if r["key"] not in live:          # finished line without a started line
                atts.append(live.setdefault(r["key"], {"start": r, "final": None}))
            live.pop(r["key"])["final"] = r
    last = {a["start"]["key"]: a for a in atts}

    eps = []
    for a in atts:
        s, f = a["start"], a["final"]
        key, i = s["key"], s.get("attempt", 0)
        sfx = "" if last[key] is a else f".attempt{i}"   # run_episode.py renames an earlier attempt's files
        log = d / f"{key}{sfx}.jsonl"
        recs, m, err = [], {}, None
        try:
            recs = load(log) if log.exists() else []
            m = episode_metrics(recs, cfg) if recs else {}
        except Exception:
            err = traceback.format_exc().strip().splitlines()[-1]
        if f and not recs and not err:
            err = "no event log"
        hdr = recs[0] if recs else {}
        entered = {r["phase"] for r in recs if r["ev"] == Ev.PHASE}
        U = m.get("U")
        rb = next((r for r in recs if r["ev"] == Ev.RETURN_BASE_DONE), None)
        past = None
        if rb is not None and U is not None:   # farthest base position along the pre-abort heading, minus the pose
            tgt = np.asarray(rb["target"])
            h = np.array([np.cos(tgt[2]), np.sin(tgt[2])])
            past = max(float((np.asarray(r["base"][:2]) - tgt[:2]) @ h)
                       for r in recs if r["ev"] == Ev.STEP and U <= r["t"] <= rb["t"])
        rb = rb and rb["t"]
        base = {"key": key, "seed": s["seed"], "phase": s["phase"], "mode": s["mode"], "rep": s["rep"], "attempt": i,
                "outcome": f["outcome"] if f else "killed", "counted": False,
                "error": f["error"].strip().splitlines()[-1] if f and f.get("error") else None,
                "metrics_error": err, "log": log.name if log.exists() else None,
                "frames": next((p.name for p in (d / f"frames_{key}{sfx}.tgz", d / f"frames_{key}{sfx}")
                                if p.exists()), None)}
        e = {**base, **{k: v for k, v in m.items() if k not in base}}
        e.update({k: hdr.get(k) for k in ("obj_lang", "layout_id", "style_id", "task")},
                 o15=m.get("chunk_offset") == 15,
                 state_hash_t_I=next((r.get("state_hash") for r in recs if r["ev"] == Ev.I), None),
                 phase_entry={"approach": 0, **{r["phase"]: r["t"] for r in recs if r["ev"] == Ev.PHASE}} if recs else None,
                 last_step=max((r["t"] for r in recs if r["ev"] == Ev.STEP), default=None),
                 return_base_done_t=rb, return_base_past_m=past,
                 contact_after_U_n=max_contact(recs, U, math.inf, nF) if U is not None else None,
                 contact_return_base_n=max_contact(recs, U, rb, nF) if U is not None and rb is not None else None,
                 paired=f.get("paired") if f else None,
                 paired_rederived=bool(s["phase"] == "none" and f and not f.get("error") and m.get("success_t") is not None
                                       and m["success_t"] <= B and {"grasp", "transport"} <= entered))
        e["counted"] = last[key] is a and f is not None and f.get("outcome") in FINISHED and not err
        a.update(m=m, err=err, e=e, rt={"phases_entered": sorted(entered), "completed": m.get("completed"),
                                         "n_goal_accepted": sum(r["ev"] == Ev.GOAL_ACCEPTED for r in recs),
                                         "n_feedback": sum(r["ev"] == Ev.FEEDBACK for r in recs),
                                         "state_hash_t_I": e["state_hash_t_I"],
                                         **{k: e[k] for k in ("obj_lang", "layout_id")}})   # scene: the log header
        eps.append(e)

    # floor input: the results lines with every provisional metric field replaced by its recomputed value; a killed
    # attempt's "started" line gets its recomputed fields too (correctness scans every attempt)
    mkeys = set().union(*(a["m"] for a in atts))
    own = {id(a["final"] or a["start"]): a for a in atts}

    def replayed(a):
        r = {k: v for k, v in (a["final"] or a["start"]).items() if k not in mkeys and k != "metrics_error"}
        r.update(a["m"], **a["rt"])
        r.update({k: a["start"][k] for k in ("key", "seed", "phase", "mode", "rep")})   # identity: the results line
        return {**r, "metrics_error": a["err"]} if a["err"] else r

    floor_lines = [replayed(own[id(r)]) if id(r) in own else r for r in lines]
    floor = check(floor_lines, pre["protocol"]["cfg_hash"], sha(man))
    prov = {}
    for a in atts:
        if a["e"]["counted"]:
            f, m = a["final"], a["m"]
            diff = {k: [f.get(k), m.get(k)] for k in sorted((mkeys & f.keys()) | m.keys()) if not same(f.get(k), m.get(k))}
            if diff:
                prov[a["e"]["key"]] = diff

    cnt = [e for e in eps if e["counted"]]
    scr = {e["seed"]: e for e in cnt if e["phase"] == "none" and e["rep"] == 0}
    paired = sorted(s for s, e in scr.items() if e["paired"])
    inter = [e for e in cnt if e["phase"] != "none"]
    void = {}
    for e in inter:
        if e.get("void"):
            void.setdefault((e["seed"], e["phase"]), {})[e["mode"]] = e["void_reason"]
    for e in eps:
        e["seed_phase_void"] = (e["seed"], e["phase"]) in void
    valid = [e for e in inter if e["seed"] in paired and not e["seed_phase_void"] and e.get("t_I") is not None]
    by = {(e["seed"], e["phase"], e["mode"]): e for e in valid}

    def cell(mode, phase):
        g = [e for e in valid if e["mode"] == mode and e["phase"] == phase]
        n, c = len(g), lambda k: sum(bool(e.get(k)) for e in g)
        dfd, sec = [e for e in g if e["re_motion"] is not None], [e for e in g if e["secured_at_I"]]
        lat = {k: spread([e[k] for e in g]) for k in ("N_post", "L_C", "L_S", "L_R", "L_B", "d_post_cm")}
        lat["L_S"] = spread([e["L_S"] for e in g if e["still_in_W"]], sum(not e["still_in_W"] for e in g), W)
        aft = [e["contact_after_U_n"] for e in g if e["contact_after_U_n"] is not None]
        out = {"n": n, "seeds": sorted(e["seed"] for e in g), "completion": rate(c("c3"), n),
               "completion_in_window": rate(c("completed_in_window"), n), "completed_any": rate(c("completed"), n),
               "re_motion": {**rate(sum(e["re_motion"] for e in dfd), len(dfd)), "undefined": n - len(dfd)},
               "over_limit_contact_W": rate(sum(e["max_contact_avg_n_W"] > F for e in g), n),
               "object_kept_W": rate(sum(bool(e["obj_held_W"]) for e in sec), len(sec)), "latency": lat,
               "contact_after_U": {"max_3step_mean_n": spread(aft), "over_limit": rate(sum(x > F for x in aft), len(aft))}}
        if mode == BT:
            rbc = [e["contact_return_base_n"] for e in g if e["contact_return_base_n"] is not None]
            out.update(c1=rate(c("c1"), n), c2=rate(c("c2"), n), c3=rate(c("c3"), n),
                       all3=rate(sum(e["c1"] and e["c2"] and e["c3"] for e in g), n),
                       contact_return_base={"max_3step_mean_n": spread(rbc), "over_limit": rate(sum(x > F for x in rbc), len(rbc))})
        return out

    # R0: replicates (rep 1, 2) on paired seeds, success within B of the goal sent at t = 0
    reps = {e["key"]: e for e in cnt if e["phase"] == "none" and e["rep"] > 0 and e["seed"] in paired}
    done = lambda e: e["success_t"] is not None and e["success_t"] <= B  # noqa: E731

    def stays(e, p):   # entered p, next phase entry >= entry + k + hysteresis (void rule), or never left p
        pe = e["phase_entry"]
        nxt = min((t for t in pe.values() if t > pe[p]), default=None) if p in pe else None
        return p in pe and (nxt is None or nxt - pe[p] >= cfg.k(p) + cfg.hysteresis_steps)

    r0c = {}
    for p in PHASES:
        sel = [e for e in reps.values() if stays(e, p)]
        lim = lambda e: e["phase_entry"][p] + cfg.k(p) + cfg.hysteresis_steps  # noqa: E731
        r0c[p] = {**rate(sum(map(done, sel)), len(sel)), "keys": sorted(e["key"] for e in sel),
                  "never_left_but_log_ends_before_entry_k_5": sorted(
                      e["key"] for e in sel if max(e["phase_entry"].values()) == e["phase_entry"][p]
                      and (e["last_step"] or 0) < lim(e))}

    disc = {}
    for p in PHASES:
        prs = [(s, by[s, p, BT]["c3"], by[s, p, M2]["c3"]) for s in paired if (s, p, BT) in by and (s, p, M2) in by]
        disc[p] = {"pairs": len(prs), "b_M4_only": [s for s, x, y in prs if x and not y],
                   "c_M2_only": [s for s, x, y in prs if y and not x]}
    disc["all"] = {"pairs": sum(v["pairs"] for v in disc.values()),
                   "b": sum(len(v["b_M4_only"]) for v in disc.values()), "c": sum(len(v["c_M2_only"]) for v in disc.values())}

    scr_att = {}
    for e in eps:
        if e["phase"] == "none" and e["rep"] == 0:
            scr_att.setdefault(e["seed"], []).append(e)
    unscreenable = sorted(s for s, al in scr_att.items()
                          if s not in scr and sum(e["outcome"] in ("error", "killed") for e in al) >= 3)
    order = {}   # first start of each interrupted key, in results-file order
    for a in atts:
        s = a["start"]
        if s["phase"] != "none" and s.get("attempt", 0) == 0:
            order.setdefault(s["seed"], {}).setdefault(s["phase"], []).append(s["mode"])
    pos = {m: [0] * len(MODES) for m in MODES}
    for ph in order.values():
        for ms in ph.values():
            for j, m in enumerate(ms):
                pos[m][j] += 1
    rot = lambda r: list(MODES[r % 4:] + MODES[:r % 4])  # noqa: E731   run_matrix.sh, rank among paired seeds

    # GIF (prereg gif_rule): lowest seed, transport BT meeting 1-3, secured at t_I, o <= 14; no other phase
    q = [e for e in valid if e["mode"] == BT and e["phase"] == "transport" and e["c1"] and e["c2"] and e["c3"]
         and e["secured_at_I"] and e["chunk_offset"] is not None and e["chunk_offset"] <= 14]
    pick = min(q, key=lambda e: e["seed"], default=None)
    partner = by.get((pick["seed"], "transport", M1)) if pick else None
    gif = {"bt": {"key": pick["key"], "frames": pick["frames"], "chunk_offset": pick["chunk_offset"]} if pick else None,
           "wait_chunk": {"key": partner["key"], "frames": partner["frames"]} if partner else None,
           "reason": None if partner else "no transport bt_safe_abort episode qualifies" if not pick
           else f"seed {pick['seed']}: no counted non-void transport wait_chunk episode"}

    # predictions, each sub-claim literally
    vals = lambda mode, k, ph=PHASES: [e[k] for e in valid if e["mode"] == mode and e["phase"] in ph]  # noqa: E731
    n1, n2, n4 = vals(M1, "N_post"), vals(M2, "N_post"), vals(BT, "N_post")
    eq = [by[s, p, M2]["N_post"] == by[s, p, BT]["N_post"] for s in paired for p in PHASES
          if (s, p, M2) in by and (s, p, BT) in by]
    o_ok = [e["N_post"] == 16 - e["chunk_offset"] for e in valid if e["mode"] == M1]
    p1 = {"M2_and_M4_N_post_le_1": claim({"max_M2": max(n2, default=None), "max_M4": max(n4, default=None)},
                                         all(x <= 1 for x in n2 + n4) if n2 and n4 else None),
          "M2_eq_M4_N_post_same_seed_phase": claim(f"{sum(eq)}/{len(eq)} equal", all(eq) if eq else None),
          "M1_eq_16_minus_o": claim(f"{sum(o_ok)}/{len(o_ok)} equal", all(o_ok) if o_ok else None),
          "M1_median_about_8": claim(med(n1), None),
          "M1_max_16": claim(max(n1, default=None), max(n1) == 16 if n1 else None)}
    for p in ("approach", "transport"):
        for mode, op, lim in ((M1, ">=", 3), (M2, "<=", 1), (BT, "<=", 1)):
            x = med(vals(mode, "d_post_cm", (p,)))
            p1[f"d_post_median_{p}_{mode}_{op}_{lim}cm"] = claim(x, None if x is None else x >= lim if op == ">=" else x <= lim)
    m3 = [e for e in valid if e["mode"] == M3]
    ns, cw = sum(not e["still_in_W"] for e in m3), sum(bool(e["completed_in_window"]) for e in m3)
    p2 = {"M3_no_standstill_in_W_ge_50pct": claim(f"{ns}/{len(m3)}", ns / len(m3) >= 0.5 if m3 else None),
          "M3_ge_1_completes_in_W": claim(cw, cw >= 1 if m3 else None)}
    k2 = lambda ph: (sum(bool(x) for x in vals(BT, "c2", ph)), len(vals(BT, "c2", ph)))  # noqa: E731
    c3r = lambda mode, p: (lambda xs: sum(map(bool, xs)) / len(xs) if xs else None)(vals(mode, "c3", (p,)))  # noqa: E731
    k, n = k2(("approach", "transport"))
    c2r = {p: (lambda kn: kn[0] / kn[1] if kn[1] else None)(k2((p,))) for p in PHASES}
    gaps = {p: None if c3r(BT, p) is None or c3r(M2, p) is None else c3r(M2, p) - c3r(BT, p) for p in PHASES}

    def lowest(x, others):   # strictly below every other; a tie is not decided by the prediction
        return None if x is None or None in others else True if x < min(others) else None if x == min(others) else False

    p3 = {"c2_ge_80pct_approach_and_transport_pooled": claim(f"{k}/{n}", k / n >= 0.8 if n else None),
          "c2_grasp_lowest_phase": claim(c2r, lowest(c2r["grasp"], [c2r["approach"], c2r["transport"]])),
          "completion_M4_le_M2_each_phase": claim({p: [c3r(BT, p), c3r(M2, p)] for p in PHASES},
                                                  None if None in gaps.values() else all(g >= 0 for g in gaps.values())),
          "completion_gap_M2_minus_M4_largest_in_transport": claim(gaps, None if None in gaps.values() else lowest(
              -gaps["transport"], [-gaps["approach"], -gaps["grasp"]]))}
    preds = {name: {"result": overall(cs), "claims": cs} for name, cs in (("P1", p1), ("P2", p2), ("P3", p3))}

    # result line (prereg result_line_template); N_post pooled over phases
    bt = [e for e in valid if e["mode"] == BT]
    all3 = lambda es: sum(e["c1"] and e["c2"] and e["c3"] for e in es)  # noqa: E731
    ph = {p: [e for e in bt if e["phase"] == p] for p in PHASES}
    mm = lambda xs: f"{num(med(xs))} [{num(min(xs))}, {num(max(xs))}]" if xs else "n/a"  # noqa: E731
    c4, c2 = vals(BT, "c3"), vals(M2, "c3")
    S, P = len(scr) + len(unscreenable), sum(bool(e["completed"]) for e in scr.values())
    fill = {"k/n interrupts": f"{all3(bt)}/{len(bt)} interrupts",
            **{f"{p} {x}": f"{p} {all3(ph[p])}/{len(ph[p])}" for p, x in zip(PHASES, ("a/n_a", "b/n_g", "c/n_t"))},
            "on N paired": f"on {len(paired)} paired", "(S screened, P succeeded": f"({S} screened, {P} succeeded",
            "<m1>": num(med(n1)), "<M1>": num(max(n1, default=None)), "<m3>": mm(vals(M3, "N_post")),
            "<m2>": mm(n2), "<m4>": mm(n4), "<c4>/n": f"{sum(map(bool, c4))}/{len(c4)}",
            "<c2>/n": f"{sum(map(bool, c2))}/{len(c2)}", "<r>/2N": f"{sum(map(done, reps.values()))}/{len(reps)}"}
    line = pre["result_line_template"]
    for x, y in fill.items():
        if x not in line:
            raise ValueError(f"result_line_template changed: {x!r} not found")
        line = line.replace(x, y)
    no_int = [s for s in paired if not any(e["seed"] == s for e in valid)]
    notes = ([f"replicates counted {len(reps)}, 2N = {2 * len(paired)}"] if len(reps) != 2 * len(paired) else []) + \
            ([f"BT n = {len(c4)}, truncate-and-hold n = {len(c2)}"] if len(c4) != len(c2) else []) + \
            ([f"paired seeds without a counted non-void interrupted episode: {no_int}"] if no_int else [])

    by_key = {e["key"]: e for e in cnt}
    mism = sorted(e["key"] for e in cnt if e["phase"] == "none" and bool(e["paired"]) != e["paired_rederived"])
    schema = floor["items"]["schema"]
    why = ([f"floor schema {json.dumps({k: v for k, v in schema.items() if k != 'pass'})}"] if not schema["pass"] else []) \
        + ([f"run-time pairing differs from the event logs: {mism}"] if mism else [])
    return {
        **({"invalid": why} if why else {}),
        "meta": {
            "freeze_commit": git("log", "-1", "--format=%h %s", "--", "eval/frozen_manifest.json"),
            "analysis_code": {"head": git("rev-parse", "--short", "HEAD"), "uncommitted": git(
                "status", "--porcelain", "--", "preempt/metrics.py", "scripts/check_floor.py",
                "scripts/build_results.py").splitlines()},
            "c6_score_dir": str(d), "results_sha256": sha(d / "results.jsonl"), "prereg_frozen_utc": pre["frozen_utc"],
            "cfg_hash_frozen": pre["protocol"]["cfg_hash"], "cfg_hash_analysis": cfg.hash(),
            "cfg_hashes_in_logs": sorted({str(e.get("cfg_hash")) for e in eps if e["log"]}),
            "manifest_sha256_frozen": sha(man),
            "manifest_sha256_in_results": sorted({str(r.get("manifest_sha256")) for r in lines}),
            "task": pre["task"], "task_in_logs": sorted({str(e["task"]) for e in eps if e["log"]}),
            "checkpoint": pre["policy"]["checkpoint"], "protocol": json.loads(proto.read_text()),
            "protocol_sha256": sha(proto),
            "boots": load(d / "boots.jsonl") if (d / "boots.jsonl").exists() else []},
        "screening": [{"seed": s, "attempts": len(al), "status": "counted" if s in scr else
                       "unscreenable" if s in unscreenable else "unfinished",
                       **{k: (scr.get(s) or al[-1]).get(k) for k in ("key", "layout_id", "style_id", "obj_lang", "completed",
                                                                     "success_t", "paired", "phase_entry")}}
                      for s, al in sorted(scr_att.items())],
        "replicates": [{"seed": s, "rep": r, "key": f"s{s}_none_none_r{r}", "counted": e is not None,
                        **({"completed": e["completed"], "success_t": e["success_t"], "completed_within_B": done(e),
                            "phase_entry": e["phase_entry"], "stays_through_k_5": {p: stays(e, p) for p in PHASES}}
                           if e else {})}
                       for s in paired for r in (1, 2) for e in [by_key.get(f"s{s}_none_none_r{r}")]],
        "episodes": eps,
        "summary": {
            "attempts": len(eps), "finished": sum(e["outcome"] != "killed" for e in eps), "counted": len(cnt),
            "screened_seeds": sorted(scr_att), "paired_seeds": paired, "unscreenable_seeds": unscreenable,
            "screening_unfinished": sorted(s for s in scr_att if s not in scr and s not in unscreenable),
            "aborted_attempts": [[e["key"], e["attempt"]] for e in eps if e["outcome"] == "killed"],
            "error_attempts": [[e["key"], e["attempt"]] for e in eps if e["outcome"] == "error"],
            "pairing_rederived_mismatch": mism,
            "mode_order": {"position_counts": pos, "observed": {str(s): v for s, v in sorted(order.items())},
                           "differs_from_rotation": sorted([s, p] for s, v in order.items() for p, ms in v.items()
                                                           if s in paired and ms != rot(paired.index(s))[:len(ms)])},
            "by_mode_phase": {m: {p: cell(m, p) for p in PHASES} for m in MODES},
            "R0": rate(sum(map(done, reps.values())), len(reps)), "R0_phase_conditional": r0c,
            "M4_vs_M2_discordant": disc,
            "void": {"seed_phase": sorted([s, p, ms] for (s, p), ms in void.items()),
                     "counts": {p: {"episodes": dict(Counter(e["void_reason"] for e in inter if e["phase"] == p and e.get("void"))),
                                    "seed_phase": dict(Counter(r for (s, q), ms in void.items() if q == p for r in set(ms.values())))}
                                for p in PHASES}},
            "pooled_by_mode": {m: {
                "contact_after_U_over_limit": rate(*(lambda xs: (sum(x > F for x in xs), len(xs)))(
                    [e["contact_after_U_n"] for e in valid if e["mode"] == m and e["contact_after_U_n"] is not None])),
                "completion": rate(sum(bool(e["c3"]) for e in valid if e["mode"] == m), sum(e["mode"] == m for e in valid))}
                for m in MODES},
            "return_base": {"episodes": len(rbp := [e["return_base_past_m"] for e in valid if e["mode"] == BT
                                                    and e["return_base_past_m"] is not None]),
                            "past_pre_abort_pose_m": {"max": max(rbp, default=None), "min": min(rbp, default=None)},
                            "contact_over_limit": sum(e["contact_return_base_n"] > F for e in valid if e["mode"] == BT
                                                      and e["contact_return_base_n"] is not None)},
            "pairing_exact": {"seed_phase_groups": len(grp := {(e["seed"], e["phase"]) for e in valid}),
                              "identical_state_hash_at_t_I": sum(
                                  len({e["state_hash_t_I"] for e in valid if (e["seed"], e["phase"]) == g}) == 1
                                  and all(e["state_hash_t_I"] for e in valid if (e["seed"], e["phase"]) == g)
                                  for g in grp)},
            "floor": floor, "gif": gif, "predictions": preds, "provisional_diffs": prov},
        "result_line": line, "result_line_notes": notes}


def report(out: dict) -> list[str]:
    s, fl = out["summary"], out["summary"]["floor"]
    pct = lambda r: f"{r['k']}/{r['n']}" if r else "-"  # noqa: E731
    L = [f"{out['meta']['c6_score_dir']}  (freeze {out['meta']['freeze_commit'].split()[0]}, cfg "
         f"{','.join(h[:8] for h in out['meta']['cfg_hashes_in_logs'])}, manifest "
         f"{','.join(h[:8] for h in out['meta']['manifest_sha256_in_results'])})",
         f"attempts {s['attempts']} (finished {s['finished']}, killed {s['aborted_attempts']}, error {s['error_attempts']}), "
         f"counted {s['counted']}",
         f"screened {len(s['screened_seeds'])} {s['screened_seeds']}, paired {s['paired_seeds']}, "
         f"unscreenable {s['unscreenable_seeds']}, unfinished {s['screening_unfinished']}",
         "floor " + ("PASS" if fl["pass"] else "FAIL") + ": " + " ".join(
             f"{k}={'ok' if v['pass'] else 'FAIL'}" for k, v in fl["items"].items()),
         f"{'mode':14s} {'phase':9s}  n  compl  in_W  N_post med[min,max]   L_S med     d_post_cm med  c1 c2 c3 all3"]
    for m, ph in s["by_mode_phase"].items():
        for p, c in ph.items():
            lat = c["latency"]
            f = lambda k: "-" if not lat[k] else f"{num(lat[k]['median'])} [{num(lat[k]['min'])}, {num(lat[k]['max'])}]"  # noqa: E731
            d = lat["d_post_cm"]
            L.append(f"{m:14s} {p:9s} {c['n']:2d}  {pct(c['completion']):5s}  {pct(c['completion_in_window']):4s}  "
                     f"{f('N_post'):20s} {num(lat['L_S']['median']) if lat['L_S'] else '-':10s}  "
                     f"{num(round(d['median'], 2)) if d else '-':13s}  " +
                     (" ".join(pct(c[x]) for x in ("c1", "c2", "c3", "all3")) if m == BT else ""))
    r0 = s["R0_phase_conditional"]
    L += [f"R0 {pct(s['R0'])}; phase-conditional R0 " + ", ".join(f"{p} {pct(r)}" for p, r in r0.items()),
          "M4 vs M2 discordant (completion): " + ", ".join(
              f"{p} b={v['b_M4_only']} c={v['c_M2_only']} of {v['pairs']}" for p, v in s["M4_vs_M2_discordant"].items()
              if p != "all") + f"; all b={s['M4_vs_M2_discordant']['all']['b']} c={s['M4_vs_M2_discordant']['all']['c']}",
          f"void: {s['void']['seed_phase'] or 'none'}",
          f"GIF: {s['gif']['bt']['key'] if s['gif']['bt'] else None} + "
          f"{s['gif']['wait_chunk']['key'] if s['gif']['wait_chunk'] else None}" +
          (f" ({s['gif']['reason']})" if s["gif"]["reason"] else ""),
          *(f"{k}: {v['result']}  " + "; ".join(f"{c} {num(x['value']) if not isinstance(x['value'], dict) else x['value']}"
                                                 f" -> {x['result']}" for c, x in v["claims"].items())
            for k, v in s["predictions"].items()),
          f"provisional fields differing from recomputed: {len(s['provisional_diffs'])} episodes"
          + (f" {sorted(s['provisional_diffs'])}" if s["provisional_diffs"] else ""),
          "RESULT LINE: " + out["result_line"], *(f"  note: {n}" for n in out["result_line_notes"])]
    return L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("c6_score", type=Path)
    p.add_argument("--out", type=Path, default=Path("results.json"))
    a = p.parse_args()
    out = build(a.c6_score)
    a.out.write_text(json.dumps(out, indent=1, default=str, allow_nan=False) + "\n")
    print("\n".join(report(out)))
    print(f"wrote {a.out}")
    if "invalid" in out:
        sys.exit(f"INVALID results in {a.out}: " + "; ".join(out["invalid"]))


if __name__ == "__main__":
    main()
