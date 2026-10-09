"""One episode = one (seed, phase, mode, rep): skill_host runs in this process, bt_runner as a child process.
Writes <out>/<key>.jsonl (event log), <out>/<key>.bt.log, optional frames, and appends exactly one JSON line
to the results file when the episode ends (also on error, with outcome "error").

    python scripts/run_episode.py --seed 1000 --phase transport --mode bt_safe_abort --policy gr00t [--video]
    python scripts/run_episode.py --seed 1000 --phase none --mode none --policy fake   # screening run
    python scripts/run_episode.py --ping

--skip-done       exit 0 without running if the results file's last line for this key finished (outcome in
                  FINISHED; run_matrix resume; the earlier lines stay as attempts); re-run after "error" or "started";
                  exit 1 without running on any other outcome (malformed results file)
Every attempt first appends a "started" line, so an attempt killed mid-episode (boot stop) stays visible as a
"started" line with no outcome after it; its partial log and frames are kept as <key>.attemptN.*.
BTVLA_MANIFEST_SHA (set by scripts/remote/c6_score.sh) is copied into every results line.
--require-paired  for interrupted runs and replicates (rep > 0): exit 3 without running unless this seed's
                  screening line (phase none, rep 0) is well-formed, finished without error and paired (paired
                  is true: completed within budget B, grasp and transport phases both entered)
--ping            exit 0 if both servers answer; else 1 = sim down, 2 = policy down, 3 = both down
rep > 0 changes only the policy noise stream (policy seed = seed + 1e6 * rep): replicate runs for R0.
Needs: ros venv, /opt/ros/humble + ~/ros_ws/install sourced, sim_server and a policy server listening.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preempt.contracts import Ev, Phase, ProtocolConfig, StopMode  # noqa: E402
from preempt.eventlog import _jsonable, read  # noqa: E402

PHASES = ["none"] + [p.value for p in (Phase.APPROACH, Phase.GRASP, Phase.TRANSPORT)]
MODES = ["none"] + [m.value for m in StopMode]
FINISHED = ("success", "failed", "blocked", "timeout")   # the episode ran to its end without an exception


def key_of(seed, phase, mode, rep) -> str:
    return f"s{seed}_{phase}_{mode}_r{rep}"


def paired_screening(r: dict, seed: int) -> bool:   # same condition in scripts/remote/c6_score.sh state()
    return ((r.get("key"), r.get("seed"), r.get("phase"), r.get("mode"), r.get("rep"))
            == (key_of(seed, "none", "none", 0), seed, "none", "none", 0)
            and r.get("paired") is True and r.get("outcome") in FINISHED and not r.get("error"))


def load_lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def head(a, key: str) -> dict:
    return {"key": key, "seed": a.seed, "phase": a.phase, "mode": a.mode, "rep": a.rep, "policy": a.policy,
            "attempt": a.attempt, "manifest_sha256": os.environ.get("BTVLA_MANIFEST_SHA")}


def summarize(a, key: str, res: dict, err: str | None, wall_s: float, bt_rc) -> dict:
    cfg = ProtocolConfig.load(a.config)
    line = {**head(a, key), "outcome": res.get("outcome"), "error": err, "wall_s": round(wall_s, 1), "bt_returncode": bt_rc,
            "log": str(a.log), "frames": str(a.frame_dir) if a.frame_dir else None,
            **{k: v for k, v in res.items() if k not in ("outcome", "meta")}, **res.get("meta", {})}
    try:
        recs = read(a.log)
    except Exception:
        recs, line["log_error"] = [], traceback.format_exc()
    try:   # analysis (not frozen): provisional, recomputed from the logs later
        from preempt.metrics import episode_metrics
        line.update(episode_metrics(recs, cfg))
    except Exception:
        line["metrics_error"] = traceback.format_exc()
    # execution decisions (pairing gates which seeds get interrupted runs): raw log only, final at run time
    entered = {r["phase"] for r in recs if r["ev"] == Ev.PHASE}
    succ_t = next((r["t"] for r in recs if r["ev"] == Ev.SUCCESS), None)
    line.update(
        phases_entered=sorted(entered), completed=succ_t is not None,
        n_goal_accepted=sum(r["ev"] == Ev.GOAL_ACCEPTED for r in recs),
        n_feedback=sum(r["ev"] == Ev.FEEDBACK for r in recs),
        state_hash_t_I=next((r.get("state_hash") for r in recs if r["ev"] == Ev.I), None),
        paired=bool(a.phase == "none" and err is None and succ_t is not None and succ_t <= cfg.reentry_budget_B
                    and {Phase.GRASP.value, Phase.TRANSPORT.value} <= entered))
    return line


def append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=_jsonable) + "\n")
        f.flush()
        os.fsync(f.fileno())


def ping(a) -> int:
    from policy.client import call
    from sim.protocol import SimClient
    out = {}
    for name, fn in (("sim", lambda: SimClient(a.sim, timeout_s=5).call("ping").head),
                     ("policy", lambda: call(a.policy_host, a.policy_port, "ping", timeout_s=5))):
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = {"error": repr(e)}
    print(json.dumps(out, default=str))
    return int("error" in out["sim"]) + 2 * int("error" in out["policy"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int)
    p.add_argument("--phase", choices=PHASES, default="none")
    p.add_argument("--mode", choices=MODES, default="none")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--policy", choices=["fake", "gr00t"], default="gr00t",
                   help="label for the results; fake also turns on --priv")
    p.add_argument("--priv", action="store_true", help="attach privileged state to policy requests")
    p.add_argument("--sim", default="tcp://127.0.0.1:5556")
    p.add_argument("--policy-host", default="127.0.0.1")
    p.add_argument("--policy-port", type=int, default=5560)
    p.add_argument("--config", default=str(ROOT / "config" / "protocol.json"))
    p.add_argument("--out", default=str(ROOT / "runs" / "matrix"))
    p.add_argument("--results", default=None, help="default: <out>/results.jsonl")
    p.add_argument("--video", action="store_true", help="save the agentview_left frame every 2nd step")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--require-paired", action="store_true")
    p.add_argument("--ping", action="store_true")
    a = p.parse_args()
    if a.ping:
        return ping(a)
    if a.seed is None:
        p.error("--seed is required")
    if min(a.seed, a.rep, a.max_steps) < 0:
        p.error("--seed, --rep and --max-steps must be >= 0")
    if (a.phase == "none") != (a.mode == "none"):
        p.error("--phase none goes with --mode none (screening / replicate run), and only with it")
    a.priv = a.priv or a.policy == "fake"
    out = Path(a.out)
    results = Path(a.results) if a.results else out / "results.jsonl"
    key = key_of(a.seed, a.phase, a.mode, a.rep)
    lines = load_lines(results)
    done = {r["key"]: r for r in lines}                # last line per key
    a.attempt = sum(r["key"] == key and r.get("outcome") == "started" for r in lines)
    if a.skip_done and key in done and done[key].get("outcome") in FINISHED:   # error or killed: re-run
        print(f"skip {key}: already in {results}")
        return 0
    if a.skip_done and key in done and done[key].get("outcome") not in ("error", "started"):
        sys.exit(f"{key}: malformed results file {results}: last outcome {done[key].get('outcome')!r}")
    if a.require_paired and (a.phase != "none" or a.rep > 0) and not paired_screening(
            done.get(key_of(a.seed, "none", "none", 0), {}), a.seed):
        print(f"skip {key}: seed {a.seed} is not a paired seed")
        return 3

    out.mkdir(parents=True, exist_ok=True)
    a.log = out / f"{key}.jsonl"
    a.frame_dir = out / f"frames_{key}" if a.video else None
    n = max(a.attempt - 1, 0)                         # files of attempt n-1 keep its index
    for old, kept in ((a.log, out / f"{key}.attempt{n}.jsonl"), (out / f"frames_{key}", out / f"frames_{key}.attempt{n}"),
                      (out / f"frames_{key}.tgz", out / f"frames_{key}.attempt{n}.tgz"),
                      (out / f"{key}.bt.log", out / f"{key}.attempt{n}.bt.log")):
        if old.exists():                              # an earlier attempt of this key (error or killed): keep it
            old.rename(kept)
    if a.frame_dir:
        a.frame_dir.mkdir()
    append(results, {**head(a, key), "outcome": "started", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    a.policy_seed = a.seed + 1_000_000 * a.rep
    with open(out / f"{key}.bt.log", "a") as bt_log:
        t0, err, res, bt, bt_rc = time.time(), None, {"outcome": "error"}, None, None
        try:
            bt = subprocess.Popen([sys.executable, str(ROOT / "ros" / "bt_runner.py"), "--mode", a.mode,
                                   "--config", a.config], stdout=bt_log, stderr=subprocess.STDOUT)
            from ros.skill_host import run_episode
            res = run_episode(a)
        except Exception:
            err = traceback.format_exc()
            print(err, file=sys.stderr)
        finally:
            if bt is not None:
                bt.terminate()
                try:
                    bt_rc = bt.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    bt.kill()
                    bt_rc = bt.wait()
    line = summarize(a, key, res, err, time.time() - t0, bt_rc)
    append(results, line)
    print(json.dumps({k: line.get(k) for k in ("key", "outcome", "completed", "c1", "c2", "c3", "N_post",
                                                "violations", "paired", "error")}, default=str))
    return 0 if err is None else 1


if __name__ == "__main__":
    sys.exit(main())
