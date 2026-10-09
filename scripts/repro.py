"""Reproduction gate: does the checkpoint reach >= 8/20 on the task in the upstream rollout loop,
and does our sim_server path agree within 20 percentage points?

    # gr00t venv (gr00t + robocasa importable), policy/seeded_server.py on :5555, sim_server on :5556
    python scripts/repro.py --task PickPlaceCounterToStove --n 20 [--seeds 1000-1009] [--legs native,ours]

native  upstream gr00t.eval.simulation.SimulationInferenceClient.run_simulation with SimulationConfig(
        env_name="robocasa/<task>", split="target", n_episodes=n, n_envs, multistep n_action_steps=16,
        max_episode_steps=get_task_horizon(task)) - the loop scripts/run_eval.py runs per task
        [V gr00t_run_eval.py, gr00t_simulation.py]; unseeded requests, so the seeded server acts as upstream.
ours    sim_server + ZmqPolicyClient + SkillController (the scoring path minus ROS/BT), no interrupt,
        budget = task horizon, seeded per (seed, query).
Prints one JSON line per leg; the native line carries gate_pass (successes >= 0.4 n).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def native(task: str, n: int, host: str, port: int, n_envs: int, video_dir: str | None) -> dict:
    from gr00t.eval.simulation import MultiStepConfig, SimulationConfig, SimulationInferenceClient, VideoConfig
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    horizon = get_task_horizon(task)
    cfg = SimulationConfig(env_name=f"robocasa/{task}", split="target", n_episodes=n, n_envs=n_envs,
                           video=VideoConfig(video_dir=video_dir),
                           multistep=MultiStepConfig(n_action_steps=16, max_episode_steps=horizon))
    t0 = time.time()
    _, succ = SimulationInferenceClient(host=host, port=port).run_simulation(cfg)
    k = int(sum(bool(x) for x in succ))
    return {"leg": "native", "task": task, "split": "target", "horizon": horizon, "successes": k,
            "attempts": len(succ), "gate_pass": k >= math.ceil(0.4 * len(succ)), "wall_s": round(time.time() - t0)}


def ours(task: str, seeds: list[int], sim_addr: str, host: str, port: int, config: str) -> dict:
    from policy.client import ZmqPolicyClient
    from preempt.arbiter import Arbiter
    from preempt.contracts import GoalStatus, ProtocolConfig, StopMode
    from preempt.skill import SkillController
    from sim.protocol import SimClient
    sim = SimClient(sim_addr)
    info = sim.info()
    if info["task"] != task:
        raise SystemExit(f"sim_server runs {info['task']}, not {task}")
    cfg = dataclasses.replace(ProtocolConfig.load(config), reentry_budget_B=int(info["horizon"]))
    per_seed, t0 = [], time.time()
    for seed in seeds:
        step = sim.reset(seed)
        meta = step.head["meta"]
        pol = ZmqPolicyClient(host, port, episode_seed=seed, chunk_len=cfg.chunk_len)
        arb = Arbiter(lambda *a, **k: None)
        sk = SkillController(cfg, pol, arb, lambda *a, **k: None, max_wait_polls=pol.max_wait_polls)
        sk.submit_goal(0, f"repro-{seed}", meta["lang"], StopMode.TRUNCATE_HOLD)
        t = 0
        for t in range(cfg.reentry_budget_B + 1):
            sk.pre_step(t, step.state, obs={"policy": step.obs, "priv": None})
            if not sk.busy:
                break
            sk.act(t)
            a, _ = arb.resolve(t)
            step = sim.step(a)
        pol.close()
        ok = sk.status_of(f"repro-{seed}") == GoalStatus.SUCCEEDED
        per_seed.append({"seed": seed, "success": ok, "steps": t, "obj_lang": meta["obj_lang"],
                         "layout_id": meta["layout_id"]})
    k = sum(r["success"] for r in per_seed)
    return {"leg": "ours", "task": task, "horizon": cfg.reentry_budget_B, "successes": k, "attempts": len(seeds),
            "per_seed": per_seed, "wall_s": round(time.time() - t0)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="PickPlaceCounterToStove",
                   choices=["PickPlaceCounterToStove", "PickPlaceCounterToCabinet"])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--seeds", default="1000-1009")   # dev seeds; eval seeds 0..39 only via c6_score.sh
    p.add_argument("--legs", default="native,ours")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--sim", default="tcp://127.0.0.1:5556")
    p.add_argument("--config", default=str(ROOT / "config" / "protocol.json"))
    p.add_argument("--video-dir", default=None)
    p.add_argument("--out", default=None, help="append the JSON lines here too")
    a = p.parse_args()
    for leg in a.legs.split(","):
        res = (native(a.task, a.n, a.host, a.port, a.n_envs, a.video_dir) if leg == "native" else
               ours(a.task, parse_seeds(a.seeds), a.sim, a.host, a.port, a.config) if leg == "ours" else
               sys.exit(f"unknown leg {leg}"))
        line = json.dumps(res)
        print(line, flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
