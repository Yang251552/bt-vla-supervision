"""Per-episode report of a scripted-expert dev run, from results.jsonl and each episode's event log (stdlib only,
so it also runs locally on a copy of runs/c2_dev/... fetched from the instance).

    python scripts/c2_report.py <out>/results.jsonl

Every episode: outcome, success step, phase entry steps (approach = 0) and whether each phase lasts long enough for
its injection offset (entry + k + hysteresis before the next phase), max object lift, final object-target xy
distance (success needs < 0.07 and the eef > 0.25 m away), max raw contact force. Interrupted episodes add the
latency segments (steps after t_I), criteria 1-3, RETRACTED / B steps and the criterion-2 force (3-step mean).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def report(results: Path) -> list[str]:
    cfg = json.loads((ROOT / "config" / "protocol.json").read_text())
    k = {"approach": cfg["k_approach"], "grasp": cfg["k_grasp"], "transport": cfg["k_transport"]}
    out, n_ok, n = [], 0, 0
    for r in _read(results):
        n += 1
        n_ok += bool(r.get("completed"))
        log = Path(r["log"])
        if not log.exists():                       # fetched run: the log sits next to results.jsonl
            log = results.parent / log.name
        recs = _read(log)
        entry = {"approach": 0, **{e["phase"]: e["t"] for e in recs if e["ev"] == "phase"}}
        steps = [e for e in recs if e["ev"] == "step"]
        nxt = {"approach": "grasp", "grasp": "transport", "transport": "done"}
        last = r.get("success_t") if r.get("success_t") is not None else r.get("steps")
        fits = {p: p in entry and (entry.get(q, last) or 0) - entry[p] >= k[p] + cfg["hysteresis_steps"]
                for p, q in nxt.items()}     # a phase still running at the end counts until success / last step
        out.append(f"{r['key']}: outcome={r.get('outcome')} completed={r.get('completed')} success_t={r.get('success_t')} "
                   f"steps={r.get('steps')} err={'yes' if r.get('error') else 'no'} violations={r.get('violations')}")
        out.append("  phase entry " + " ".join(f"{p}={entry.get(p)}" for p in ("approach", "grasp", "transport", "done"))
                   + "  injection fits (entry+k+5 <= next entry): " + " ".join(f"{p}={'y' if v else 'n'}" for p, v in fits.items()))
        out.append(f"  max_lift_m={max((e['lift'] for e in steps), default=None)} obj_tgt_xy_m={r.get('obj_tgt_xy_m')} "
                   f"max_force_raw_n={max((e['force'] for e in steps), default=None)} "
                   f"policy_ms_mean={r.get('policy_ms_mean')} wall_s={r.get('wall_s')}")
        if r.get("t_I") is not None:
            out.append(f"  t_I={r['t_I']} (phase at I: {next((e.get('phase') for e in recs if e['ev'] == 'I'), None)}) "
                       f"R={r.get('R')} A={r.get('A')} Q={r.get('Q')} E={r.get('E')} H={r.get('H')} "
                       f"RETRACTED={r.get('RETRACTED')} B={r.get('B')} U={r.get('U')} reentry={r.get('reentry_t')}")
            out.append(f"  latency L_C={r.get('L_C')} L_S={r.get('L_S')} L_R={r.get('L_R')} L_B={r.get('L_B')} "
                       f"(T={cfg['abort_budget_T']}) N_post={r.get('N_post')} d_post_cm={r.get('d_post_cm')} "
                       f"chunk_offset={r.get('chunk_offset')}")
            out.append(f"  c1={r.get('c1')} c2={r.get('c2')} c3={r.get('c3')} void={r.get('void')} "
                       f"backoff_m={r.get('backoff_m')} max_contact_avg_n={r.get('max_contact_avg_n')} "
                       f"(F={cfg['force_limit_n']}) secured_at_I={r.get('secured_at_I')} obj_held={r.get('obj_held')}")
        if r.get("error") or r.get("metrics_error"):
            out.append("  " + (r.get("error") or r.get("metrics_error")).strip().splitlines()[-1])
    out.append(f"SUMMARY {n_ok}/{n} completed ({results})")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print("\n".join(report(Path(sys.argv[1]))))
