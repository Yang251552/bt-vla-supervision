"""Upstream GR00T inference server with per-query seeding (gr00t venv).

    python policy/seeded_server.py --model_path $CKPT [--port 5560] [--deterministic]

Same construction as upstream scripts/run_eval.py run_server [V gr00t_run_eval.py]: Gr00tPolicy(model_path,
modality_config, modality_transform, embodiment_tag, denoising_steps=4) behind
gr00t.eval.robot.RobotInferenceServer [V gr00t/eval/robot.py], same wire protocol and endpoints.
Differences, all additive:
  - data["_seed"] = [episode_seed, query_idx] -> torch (CPU+CUDA), numpy and random are seeded from that pair
    right before get_action. The flow-matching head samples its initial noise with an unseeded torch.randn
    [V fm_head.py get_action], so without this, inference is random. Requests without "_seed" (e.g. the
    upstream eval client used by scripts/repro.py) run unseeded, exactly as upstream.
  - data["_priv"] is dropped (only the fake server uses it).
  - binds 127.0.0.1 instead of * ; extra endpoint "stats" (latency, peak GPU memory).
Whether seeding makes chunks bitwise reproducible on the GPU (flash-attn, cuBLAS) was not known in advance;
eval/errata.md has what the scored data shows.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
from gr00t.eval.robot import RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


def query_seed(episode_seed: int, query_idx: int) -> int:
    return (int(episode_seed) * 100_003 + int(query_idx)) % 2**31


class SeededPolicy:
    def __init__(self, policy, ident: dict):
        self.policy, self.ident = policy, ident   # ident: server args, reported by stats (c6 identity check)
        self.ms: list[float] = []

    def get_action(self, data: dict) -> dict:
        seed = data.pop("_seed", None)
        data.pop("_priv", None)
        if seed is not None:
            s = query_seed(*seed)
            torch.manual_seed(s)  # seeds every device's generator
            np.random.seed(s)
            random.seed(s)
        t0 = time.perf_counter()
        out = self.policy.get_action(data)
        self.ms.append(1000.0 * (time.perf_counter() - t0))
        return out

    def get_modality_config(self):
        return self.policy.get_modality_config()

    def stats(self) -> dict:
        ms = np.array(self.ms or [np.nan])
        return {**self.ident, "n": len(self.ms), "mean_ms": float(ms.mean()), "p95_ms": float(np.percentile(ms, 95)),
                "max_ms": float(ms.max()),
                "max_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_config", default="panda_omron")
    p.add_argument("--embodiment_tag", default="new_embodiment")
    p.add_argument("--denoising_steps", type=int, default=4)   # upstream run_server value
    p.add_argument("--port", type=int, default=5560)
    p.add_argument("--deterministic", action="store_true",
                   help="cudnn deterministic + torch.use_deterministic_algorithms(warn_only=True)")
    a = p.parse_args()
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # read at first cuBLAS use, not at import
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    dc = DATA_CONFIG_MAP[a.data_config]
    policy = Gr00tPolicy(model_path=a.model_path, modality_config=dc.modality_config(),
                         modality_transform=dc.transform(), embodiment_tag=a.embodiment_tag,
                         denoising_steps=a.denoising_steps)
    seeded = SeededPolicy(policy, {k: getattr(a, k) for k in ("model_path", "data_config", "embodiment_tag",
                                                              "denoising_steps", "deterministic")})
    server = RobotInferenceServer(seeded, host="127.0.0.1", port=a.port)
    server.register_endpoint("stats", seeded.stats, requires_input=False)
    server.run()


if __name__ == "__main__":
    main()
