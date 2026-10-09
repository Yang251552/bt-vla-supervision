"""zmq wire format v1 between sim_server (robocasa venv, numpy 2.2.5) and its clients (ROS venv, repro).

A message is a multipart frame list [JSON header, *raw array buffers]; header["arrays"] lists
[key, dtype, shape] for each buffer, in order. JSON + raw bytes and no pickle, so the two sides can
run different numpy majors. Every header carries "v"; a mismatch is an error, never a guess.

Requests:  {"cmd": "reset", "seed": int} | {"cmd": "step"} + array "action" (12,) | {"cmd": "info"} |
           {"cmd": "ping"} | {"cmd": "close"}
Replies:   {"ok": true, "t", "state": SimState fields, "hash", "target_pos", "obs_text", "priv", "meta"?}
           ("priv" = privileged extras for the fake policy only; see sim/sim_server.py docstring)
           + arrays = policy observation (video.* uint8 HxWx3, state.* float32), or
           {"ok": false, "error": str}.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np

from preempt.contracts import SimState

VERSION = 1
_STATE_FIELDS = {f.name for f in dataclasses.fields(SimState)}


def send(sock, header: dict, arrays: dict | None = None) -> None:
    arrays = {k: np.ascontiguousarray(v) for k, v in (arrays or {}).items()}
    head = dict(header, v=VERSION, arrays=[[k, a.dtype.str, list(a.shape)] for k, a in arrays.items()])
    sock.send_multipart([json.dumps(head).encode()] + [a.tobytes() for a in arrays.values()])


def recv(sock) -> tuple[dict, dict]:
    frames = sock.recv_multipart()
    head = json.loads(frames[0])
    if head.get("v") != VERSION:
        raise RuntimeError(f"wire version {head.get('v')!r} != {VERSION}")
    specs = head.pop("arrays", [])
    if len(specs) != len(frames) - 1:
        raise RuntimeError(f"{len(specs)} array specs for {len(frames) - 1} buffers")
    arrays = {k: np.frombuffer(f, dtype=np.dtype(dt)).reshape(shape)
              for (k, dt, shape), f in zip(specs, frames[1:])}
    return head, arrays


def state_from_dict(d: dict) -> SimState:
    missing = _STATE_FIELDS - set(d)
    if missing:
        raise ValueError(f"state lacks {sorted(missing)}")
    return SimState(**{k: np.asarray(v, dtype=float) if isinstance(v, list) else v
                       for k, v in d.items() if k in _STATE_FIELDS})


@dataclasses.dataclass
class Step:
    head: dict
    arrays: dict

    @property
    def state(self) -> SimState:
        return state_from_dict(self.head["state"])

    @property
    def obs(self) -> dict:
        """Policy observation exactly as RoboCasaGymEnv.get_observation returns it (single env)."""
        return {**self.arrays, **self.head.get("obs_text", {})}


class SimClient:
    def __init__(self, addr: str = "tcp://127.0.0.1:5556", timeout_s: float = 300.0):
        import zmq  # only clients that talk to the server need pyzmq (bt_runner imports this module too)
        self.sock = zmq.Context.instance().socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(addr)

    def call(self, cmd: str, arrays: dict | None = None, **kw) -> Step:
        send(self.sock, dict(kw, cmd=cmd), arrays)
        head, arrays = recv(self.sock)
        if not head.get("ok"):
            raise RuntimeError(f"sim_server {cmd}: {head.get('error')}")
        return Step(head, arrays)

    def reset(self, seed: int) -> Step:
        return self.call("reset", seed=int(seed))

    def step(self, action) -> Step:
        return self.call("step", {"action": np.asarray(action, dtype=np.float64).reshape(12)})

    def info(self) -> dict:
        return self.call("info").head
