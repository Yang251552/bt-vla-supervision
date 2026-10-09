"""preempt.skill.PolicyClient over the GR00T zmq inference protocol, plus the obs/action mapping for the
panda_omron data config. Talks to policy/seeded_server.py (real GR00T) or policy/fake_server.py.

Protocol (unchanged from upstream gr00t/eval/service.py [V gr00t_service.py]): zmq REQ/REP, each message is
a torch.save-serialised dict {"endpoint": name, "data": dict}; replies are dicts, {"error": str} on failure.
Extra request keys we add, which our servers pop before calling the policy:
  data["_seed"] = [episode_seed, query_idx]   (seeded_server seeds torch/numpy/random from it)
  data["_priv"] = privileged state dict        (fake_server only; sent with priv=True): the SimState fields
                   + target_pos + the sim_server "priv" block (object / pan bboxes, finger qpos and axis,
                   arm root). seeded_server pops it unread, so GR00T never sees it.

panda_omron keys [V gr00t/experiment/data_config.py PandaOmronDataConfig]: 3 video keys, 5 state keys
(3+4+2+3+4 = 16-D), language "annotation.human.task_description", 16 action steps.
Batch layout = what upstream eval sends with n_envs=1: SyncVectorEnv (B=1) over MultiStepWrapper (T=1)
(seen in upstream gr00t_simulation.py, mstep.py); Gr00tPolicy treats state ndim 3 as batched (gr00t/model/policy.py).
The language entry is a length-B tuple of strings (gymnasium vector concatenation of Text spaces; assumed, not seen
in the upstream source).
"""
from __future__ import annotations

import io
import queue
import threading
import time

import numpy as np

from preempt import action_layout as al
from preempt.skill import ChunkReply, ChunkRequest

VIDEO_KEYS = ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")
STATE_KEYS = ("state.end_effector_position_relative", "state.end_effector_rotation_relative",
              "state.gripper_qpos", "state.base_position", "state.base_rotation")
LANG_KEY = "annotation.human.task_description"


class TorchSerializer:
    """Byte-identical to gr00t.eval.service.TorchSerializer (copied so the ROS venv needs only torch-cpu)."""

    @staticmethod
    def to_bytes(data: dict) -> bytes:
        import torch
        buf = io.BytesIO()
        torch.save(data, buf)
        return buf.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        import torch
        return torch.load(io.BytesIO(data), weights_only=False)  # trusted local server, as upstream


def to_request(obs: dict, instruction: str) -> dict:
    d = {k: np.asarray(obs[k])[None, None] for k in VIDEO_KEYS + STATE_KEYS}
    d[LANG_KEY] = (instruction,)
    return d


def to_chunk(reply: dict, chunk_len: int) -> np.ndarray:
    acts = reply.get("actions", reply)   # upstream eval accepts both shapes
    d = {}
    for k, v in acts.items():
        if k.startswith("action."):
            v = np.asarray(v)
            d[k] = v[0] if v.ndim == 3 else v   # drop the batch dim
    chunk = al.from_gr00t(d)
    if chunk.shape != (chunk_len, al.DIM):
        raise ValueError(f"chunk shape {chunk.shape}, expected {(chunk_len, al.DIM)}")
    return chunk


def _socket(host: str, port: int, timeout_s: float):
    import zmq
    s = zmq.Context.instance().socket(zmq.REQ)
    s.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    s.setsockopt(zmq.LINGER, 0)
    s.connect(f"tcp://{host}:{port}")
    return s


def call(host: str, port: int, endpoint: str, data: dict | None = None, timeout_s: float = 10.0) -> dict:
    """One-off request on a fresh socket (ping / stats / get_modality_config)."""
    s = _socket(host, port, timeout_s)
    try:
        req = {"endpoint": endpoint} if data is None else {"endpoint": endpoint, "data": data}
        s.send(TorchSerializer.to_bytes(req))
        return TorchSerializer.from_bytes(s.recv())
    finally:
        s.close()


class ZmqPolicyClient:
    """request() returns at once; a worker thread does the blocking REQ/REP call; poll() hands back finished
    replies tagged (gen, query_idx). The SkillController fences stale generations, not this class.
    poll() blocks up to poll_wait_s only while a request is outstanding, so the controller's lockstep wait
    (max_wait_polls iterations) spans timeout_s of wall time: pass max_wait_polls = timeout_s / poll_wait_s.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5560, episode_seed: int = 0, chunk_len: int = 16,
                 timeout_s: float = 120.0, priv: bool = False, poll_wait_s: float = 0.005):
        self.host, self.port, self.episode_seed, self.chunk_len = host, port, int(episode_seed), chunk_len
        self.timeout_s, self.priv, self.poll_wait_s = timeout_s, priv, poll_wait_s
        self.latency_ms: list[float] = []
        self._todo: queue.Queue = queue.Queue()
        self._done: queue.Queue = queue.Queue()
        self._outstanding = 0
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @property
    def max_wait_polls(self) -> int:
        return int(self.timeout_s / self.poll_wait_s) + 1

    def request(self, req: ChunkRequest) -> None:
        self._outstanding += 1
        self._todo.put(req)

    def poll(self) -> list[ChunkReply]:
        items = []
        try:
            items.append(self._done.get(timeout=self.poll_wait_s) if self._outstanding else self._done.get_nowait())
            while True:
                items.append(self._done.get_nowait())
        except queue.Empty:
            pass
        out = []
        for item in items:
            self._outstanding -= 1
            if isinstance(item, BaseException):
                raise item
            out.append(item)
        return out

    def close(self) -> None:
        self._todo.put(None)

    def _worker(self) -> None:
        sock = _socket(self.host, self.port, self.timeout_s)
        while True:
            req = self._todo.get()
            if req is None:
                sock.close()
                return
            try:
                data = to_request(req.obs["policy"], req.instruction)
                data["_seed"] = [self.episode_seed, req.query_idx]
                if self.priv:
                    data["_priv"] = req.obs["priv"]
                t0 = time.perf_counter()
                sock.send(TorchSerializer.to_bytes({"endpoint": "get_action", "data": data}))
                resp = TorchSerializer.from_bytes(sock.recv())  # zmq.Again after timeout_s: REQ socket is dead
                self.latency_ms.append(1000.0 * (time.perf_counter() - t0))
                if "error" in resp:
                    raise RuntimeError(f"policy server: {resp['error']}")
                self._done.put(ChunkReply(req.gen, req.query_idx, to_chunk(resp, self.chunk_len)))
            except BaseException as e:  # surfaces in poll(); the episode then fails loudly
                self._done.put(e)
