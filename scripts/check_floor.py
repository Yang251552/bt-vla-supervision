"""Showcase floor (the minimum result, items below) from a results.jsonl written by run_episode.py / run_matrix.sh.

    python scripts/check_floor.py runs/gr00t_PickPlaceCounterToStove/results.jsonl [--json floor.json]

Exit 0 iff every item passes; exit 1 also if --prereg or the frozen_manifest.json next to it is missing or unreadable.
The last line per key is the sample: an episode that ended in "error", or was killed (its "started" line is the last
one), is re-run on resume and appends new lines; earlier lines stay as attempts. Only episodes that finished (outcome
in FINISHED) with metrics and a valid schema count toward the items below; correctness and schema scan every line.
  pairing      report screened / paired seeds; interrupted episodes only on paired seeds
  latency      each phase: >= 3 paired seeds on which all four stop modes ran (a (seed, phase) is void for
               all modes if any mode marked it void); per-mode latency medians are reported
  safe_abort   BT safe-abort episodes meeting criteria 1-3: >= 3, >= 1 per phase, >= 2 objects or layouts
  semantics    every non-error episode shows goal accepted and feedback; interrupted ones a cancel accept (A)
  correctness  zero fence / ownership violations, no BT re-entry without a finished abort, no timeout, in any
               attempt (pre-registered stop condition: violation, recovery skipped, deadlock)
  one_config   one protocol hash and one code manifest across episodes, the frozen ones
  schema       every outcome in OUTCOMES; c1-c3 and paired are JSON booleans or null; key = s<seed>_<phase>_<mode>_r<rep>
               with seed, rep ints >= 0 and known phase, mode; counts ints >= 0; latencies, d_post finite >= 0 or null;
               obj_lang, layout_id null, an int >= 0 or a non-empty string; no metrics_error on a line that is not "error"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

PHASES = ("approach", "grasp", "transport")
MODES = ("wait_chunk", "truncate_hold", "language_stop", "bt_safe_abort")
FINISHED = ("success", "failed", "blocked", "timeout")
OUTCOMES = FINISHED + ("error", "started")
FLAGS = ("c1", "c2", "c3", "paired")
COUNTS = ("n_goal_accepted", "n_feedback", "violations", "N_post")
LATENCIES = ("L_C", "L_S", "L_R", "L_B", "d_post_cm")


def _median(xs):
    xs = [x for x in xs if x is not None]
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)} if xs else None


def _count(x):
    return type(x) is int and x > 0


def _invalid(r) -> list[str]:
    """Fields of one results line that break the schema (outcome and flags are reported separately)."""
    nat = lambda x: type(x) is int and x >= 0  # noqa: E731
    num = lambda x: type(x) in (int, float) and math.isfinite(x) and x >= 0  # noqa: E731
    name = lambda x: nat(x) or type(x) is str and x.strip() != ""  # noqa: E731
    bad = [k for keys, ok in ((COUNTS, nat), (LATENCIES, num), (("obj_lang", "layout_id"), name)) for k in keys
           if r.get(k) is not None and not ok(r[k])]
    if not (nat(r.get("seed")) and nat(r.get("rep")) and r.get("phase") in ("none", *PHASES)
            and r.get("mode") in ("none", *MODES) and r.get("key") == f"s{r['seed']}_{r['phase']}_{r['mode']}_r{r['rep']}"):
        bad.append("identity")
    if "metrics_error" in r and r.get("outcome") != "error":
        bad.append("metrics_error")
    return bad


def check(lines: list[dict], frozen_cfg: str | None = None, frozen_manifest: str | None = None) -> dict:
    by_key, started, done, rerun = {}, Counter(), set(), set()
    for r in lines:
        if r["key"] in done:       # any line after a finished run: run again (not via resume)
            rerun.add(r["key"])
        if r.get("outcome") not in ("started", "error"):
            done.add(r["key"])
        by_key[r["key"]] = r
        started[r["key"]] += r.get("outcome") == "started"
    finished = Counter(r["key"] for r in lines if r.get("outcome") != "started")
    aborted = sorted(k for k in started if started[k] > finished[k])     # attempts killed mid-episode
    eps = list(by_key.values())
    ok_eps = [r for r in eps if r.get("outcome") in FINISHED and not _invalid(r)]   # _invalid covers metrics_error
    screen = {r["seed"]: r for r in ok_eps if r["phase"] == "none" and r["rep"] == 0}
    paired = {s for s, r in screen.items() if r.get("paired") is True}
    inter = [r for r in ok_eps if r["phase"] != "none"]
    void = {(r["seed"], r["phase"]): r.get("void_reason") for r in inter if r.get("void")}
    valid = [r for r in inter if r["seed"] in paired and (r["seed"], r["phase"]) not in void
             and r.get("t_I") is not None]
    have = {(r["seed"], r["phase"], r["mode"]) for r in valid}
    items = {}

    off = sorted(r["key"] for r in inter if r["seed"] not in paired)
    items["pairing"] = {"pass": bool(paired) and not off, "screened": len(screen),
                        "succeeded_uninterrupted": sum(bool(r.get("completed")) for r in screen.values()),
                        "paired": len(paired), "interrupted_on_unpaired_seeds": off,
                        "void_seed_phase": sorted([s, p, why] for (s, p), why in void.items())}

    full = {p: sorted(s for s in paired if all((s, p, m) in have for m in MODES)) for p in PHASES}
    items["latency"] = {"pass": all(len(full[p]) >= 3 for p in PHASES), "seeds_with_all_modes": full,
                        "by_mode": {m: {k: _median([r.get(k) for r in valid if r["mode"] == m])
                                        for k in ("N_post", "L_C", "L_S", "L_R", "L_B", "d_post_cm")}
                                    for m in MODES}}

    bt = [r for r in valid if r["mode"] == "bt_safe_abort" and all(r.get(c) is True for c in FLAGS[:3])]
    per_phase = Counter(r["phase"] for r in bt)
    objs, layouts = {r.get("obj_lang") for r in bt} - {None}, {r.get("layout_id") for r in bt} - {None}
    items["safe_abort"] = {"pass": len(bt) >= 3 and all(per_phase[p] >= 1 for p in PHASES)
                           and (len(objs) >= 2 or len(layouts) >= 2),
                           "count": len(bt), "attempts": sum(r["mode"] == "bt_safe_abort" for r in valid),
                           "per_phase": dict(per_phase), "objects": sorted(map(str, objs)),
                           "layouts": sorted(map(str, layouts))}

    missing = sorted(r["key"] for r in ok_eps if not (_count(r.get("n_goal_accepted")) and _count(r.get("n_feedback")))
                     or (r.get("t_I") is not None and r.get("void_reason") != "no_injection"   # skill_host logs the
                         and r.get("A") is None))   # planned t_I even when never reached; injected -> cancel accept
    items["semantics"] = {"pass": bool(ok_eps) and not missing, "episodes": len(ok_eps), "missing": missing}

    bad = sorted({r["key"] for r in lines if r.get("violations")})
    skipped = sorted({r["key"] for r in lines if r.get("mode") == "bt_safe_abort"
                      and r.get("reentry_t") is not None and r.get("B") is None})
    stuck = sorted({r["key"] for r in lines if r.get("outcome") == "timeout"})
    items["correctness"] = {"pass": not (bad or skipped or stuck), "episodes_with_violations": bad,
                            "reentry_without_abort": skipped, "timeouts": stuck}

    hashes = sorted({str(r.get("cfg_hash")) for r in ok_eps})
    code = sorted({str(r.get("manifest_sha256")) for r in ok_eps})
    items["one_config"] = {"pass": len(hashes) == 1 and frozen_cfg in (None, hashes[0])
                           and len(code) == 1 and frozen_manifest in (None, code[0]),
                           "cfg_hashes": hashes, "frozen": frozen_cfg,
                           "manifest_sha256": code, "frozen_manifest": frozen_manifest}

    outcome = sorted({r["key"] for r in lines if r.get("outcome") not in OUTCOMES})
    flags = sorted({r["key"] for r in lines if any(r.get(f) is not None and type(r[f]) is not bool for f in FLAGS)})
    fields = sorted({f"{r.get('key')}: {f}" for r in lines for f in _invalid(r)})
    items["schema"] = {"pass": not (outcome or flags or fields), "invalid_outcome": outcome, "invalid_flags": flags,
                       **({"invalid_fields": fields} if fields else {})}
    return {"pass": all(v["pass"] for v in items.values()), "episodes": len(eps), "aborted_attempts": aborted,
            "rerun_after_finish": sorted(rerun),
            "not_finished": sorted(r["key"] for r in eps if r not in ok_eps), "items": items}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("--json", default=None, help="also write the report here")
    p.add_argument("--prereg", default=str(Path(__file__).resolve().parents[1] / "eval" / "prereg.json"),
                   help="frozen protocol: its cfg_hash and the sha256 of frozen_manifest.json next to it must be the "
                        "only ones in the results (both files required)")
    a = p.parse_args()
    pre, man = Path(a.prereg), Path(a.prereg).with_name("frozen_manifest.json")
    try:
        cfg = json.loads(pre.read_text())["protocol"]["cfg_hash"]
        json.loads(man.read_text())
        if not isinstance(cfg, str) or not cfg:
            raise ValueError(f"cfg_hash {cfg!r}")
    except (OSError, ValueError, LookupError, TypeError) as e:
        sys.exit(f"check_floor: no frozen protocol to check against ({pre}, {man}): {e!r}")
    lines = [json.loads(x) for x in Path(a.results).read_text().splitlines() if x.strip()]
    rep = check(lines, cfg, hashlib.sha256(man.read_bytes()).hexdigest())
    text = json.dumps(rep, indent=1, default=str, allow_nan=False)
    print(text)
    if a.json:
        Path(a.json).write_text(text + "\n")
    for name, item in rep["items"].items():
        print(f"{'PASS' if item['pass'] else 'FAIL'}  {name}", file=sys.stderr)
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
