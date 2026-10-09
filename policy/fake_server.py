"""Scripted expert standing in for the GR00T server, same zmq protocol (g4dn plumbing runs, T calibration).

    python policy/fake_server.py [--port 5560] [--latency-ms 0] [--vmax 0.006] [--eef-rad-per-unit 0.33]

Needs only torch + pyzmq + numpy (run it from the gr00t venv; gr00t itself is not imported).

Input: the privileged block the client attaches as data["_priv"] (run_episode --policy fake turns on --priv;
policy/seeded_server.py pops it unread, so GR00T never sees it). It holds the SimState fields, target_pos
(the pan = "container" body) and the sim_server "priv" extras: obj_bbox / target_bbox (8 world corners),
finger_qpos, finger_axis, arm_root_pos (sim/sim_server.py docstring). Missing extras fall back to
defaults (no yaw alignment, no base motion, guessed sizes). Without data["_priv"] the server returns an error.

Behaviour (PickPlaceCounterToStove: success = object touching the pan, object body within 0.07 m of the pan
body in xy, eef site > 0.25 m from the object [V rc_kitchen_pick_place.py, object_utils.py]):
approach to above the object (fingers turned across the object's short side if it is elongated) ->
descend -> close -> lift >= LIFT -> carry above the pan rim -> lower -> release -> retreat up and back.
Every query re-reads the real state and plans one 16-step chunk open-loop on a kinematic model of the
measured response (recovery.Gains: 0.0118 m/step per unit eef, 0.0041 m/step base); the stage is a pure
function of the state, so a re-entry after a safe-abort simply continues. Commands are saturated to
[-1, 1]; the base stays still (arm mode, last dim < 0) unless the next aim point is out of reach from the
arm root. Deterministic: same request -> same chunk. Instruction "stop" yields a hold chunk (the stub's
FakePolicy(obey_stop=True) behaviour).
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.client import TorchSerializer  # noqa: E402
from preempt.recovery import Gains  # noqa: E402

CHUNK = 16
EEF_M, BASE_M = Gains().eef_m_per_unit, Gains().base_m_per_unit   # measured 2026-10-08 g4dn selftest
KP, KP_BASE = 0.5, 0.5          # fraction of the remaining model error commanded per step
HOVER, LIFT, CLEAR, GAP = 0.08, 0.08, 0.05, 0.02   # above obj top; lift; over pan rim; obj bottom over pan floor
XY_TOL, Z_TOL, PLACE_XY, IN_PAN = 0.012, 0.012, 0.015, 0.07
YAW_TOL, ROT_RATE = 0.15, 0.06  # rad; planned eef yaw rate cap, rad per step
RETREAT_UP, RETREAT_BACK = 0.15, 0.25   # final eef-object distance ~0.29 m > 0.25
Q_OPEN, Q_SHUT, Q_FULL, Q_RATE = 0.035, 0.005, 0.04, 0.005   # finger joint qpos (m); check_obj_grasped: < 0.035
CLOSE_STEPS, OPEN_STEPS = 12, 8
REACH, REACH_GOAL = 0.75, 0.65  # assumed: horizontal arm-root -> aim distance that triggers / ends base motion
CLOSE, OPEN = 1.0, 0.0          # action.gripper_close (>= 0.5 closes); action.control_mode 0 arm, 1 base


@dataclass(frozen=True)
class Opts:
    vmax: float = 0.006          # eef path speed cap, m per step (x 20 Hz = 12 cm/s); sets the phase lengths
    eef_rad_per_unit: float = 0.33   # assumed: signed planning gain for eef yaw: chunks converge for any true gain in
    #                                  (0, 0.66); set it to the selftest probe eef_rotvec_* z value (sign included)


def _rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _unit_xy(v) -> np.ndarray | None:
    v = np.array([v[0], v[1]], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def _short_axis(pts: np.ndarray):
    """bbox corners (order p0, px, py, pz, ...) -> (unit xy of the shorter horizontal edge or None, half
    width of the shorter horizontal edge). None when the footprint is roughly round."""
    edges = [pts[i] - pts[0] for i in (1, 2, 3)]
    lens = [float(np.linalg.norm(e)) for e in edges]
    vert = int(np.argmax([abs(e[2]) / max(n, 1e-9) for e, n in zip(edges, lens)]))
    (a, la), (b, lb) = [(e, n) for i, (e, n) in enumerate(zip(edges, lens)) if i != vert]
    short, ls, ll = (a, la, lb) if la <= lb else (b, lb, la)
    return (_unit_xy(short) if ll >= 1.4 * ls and ll > 0.05 else None), ls / 2


class _Model:
    """World state predicted inside one chunk; rebuilt from the real state at every query."""

    def __init__(self, p: dict, gq, o: Opts):
        f = lambda k: np.array(p[k], dtype=float)  # noqa: E731  (copies: the model moves them in place)
        self.o, self.eef, self.base, self.obj = o, f("eef_pos_world"), f("base_pose"), f("obj_pos")
        self.tgt = None if p.get("target_pos") is None else f("target_pos")
        self.success = bool(p["success"])
        self.obj_z0 = self.obj[2] - float(p["obj_lift"])
        if p.get("obj_bbox") is not None:
            pts = f("obj_bbox")
            ctr = pts.mean(0)
            self.c_off, self.top, self.bot = ctr - self.obj, pts[:, 2].max() - ctr[2], ctr[2] - pts[:, 2].min()
            self.short, self.half_w = _short_axis(pts)
        else:
            self.c_off, self.top, self.bot, self.short, self.half_w = np.zeros(3), 0.03, 0.03, None, 0.02
        if p.get("target_bbox") is not None:
            z = f("target_bbox")[:, 2]
            self.pan_bot, self.pan_top = float(z.min()), float(z.max())
        elif self.tgt is not None:
            self.pan_bot, self.pan_top = self.tgt[2], self.tgt[2] + 0.06   # assumed: pan body origin height
        q = gq if p.get("finger_qpos") is None else p["finger_qpos"]
        self.q = None if q is None else float(np.mean(np.abs(np.asarray(q, dtype=float).reshape(-1)[:2])))
        self.fax = None if p.get("finger_axis") is None else _unit_xy(p["finger_axis"])
        self.root = None if p.get("arm_root_pos") is None else f("arm_root_pos")
        # holding = check_obj_grasped, or fingers stopped part-way on a lifted object right at the eef (a one-step
        # flicker of the contact check must not make the expert open and drop it)
        self.grasped = bool(p["grasped"]) or (self.q is not None and Q_SHUT < self.q < Q_OPEN and
                                              float(p["obj_lift"]) > 0.01 and np.linalg.norm(self.eef - self.ctr) < 0.05)
        self.n_open = OPEN_STEPS if (self.q is None and not self.grasped) else 0
        at = self._at_grasp()
        self.n_close = 1 if (at and self.q is not None and Q_SHUT <= self.q <= Q_OPEN and not self.grasped) else 0
        self.aligned0 = abs(self._yaw_err()) < YAW_TOL          # real alignment gates the descent
        self.offset = self.obj - self.eef                       # held object rides with the eef

    # ---- geometry ----
    @property
    def ctr(self) -> np.ndarray:
        return self.obj + self.c_off

    def _yaw_err(self) -> float:
        if self.short is None or self.fax is None:
            return 0.0
        d = np.arctan2(self.short[1], self.short[0]) - np.arctan2(self.fax[1], self.fax[0])
        return float((d + np.pi / 2) % np.pi - np.pi / 2)    # fingers are symmetric: modulo pi

    def _at_grasp(self) -> bool:
        c = self.ctr
        return np.linalg.norm(self.eef[:2] - c[:2]) < XY_TOL and self.eef[2] - c[2] < Z_TOL

    def _open(self) -> bool:
        return self.n_open >= OPEN_STEPS or (self.q is not None and self.q > Q_OPEN)

    def _z_for_bottom(self, z_bottom: float) -> float:
        """Object body height at which the object's bottom is at z_bottom."""
        return z_bottom + self.bot - self.c_off[2]

    # ---- one step: decide, command, advance the model ----
    def decide(self):
        """-> (stage, eef aim in world or None, gripper, wants yaw alignment)."""
        if self.success or self.tgt is None:
            return "hold", None, OPEN if not self.grasped else CLOSE, False
        tgt, obj, c = self.tgt, self.obj, self.ctr
        carry = max(self.obj_z0 + LIFT, self._z_for_bottom(self.pan_top + CLEAR))
        release = self._z_for_bottom(self.pan_bot + GAP)
        if self.grasped:
            if np.linalg.norm(obj[:2] - tgt[:2]) > PLACE_XY:
                xy = obj[:2] if obj[2] < carry - 0.03 else tgt[:2]   # lift straight up first, then carry
                want = np.array([xy[0], xy[1], carry])
            elif obj[2] > release + Z_TOL:
                want = np.array([tgt[0], tgt[1], release])
            else:
                return "release", None, OPEN, False
            return "transport", self.eef + (want - obj), CLOSE, False
        if np.linalg.norm(obj[:2] - tgt[:2]) < IN_PAN and obj[2] < release + 0.03:   # released in the pan
            if not self._open():
                return "release", None, OPEN, False
            up = obj[2] + RETREAT_UP
            if self.eef[2] < up - Z_TOL:
                return "retreat", np.array([self.eef[0], self.eef[1], up]), OPEN, False
            back = -_rz(self.base[2])[:, 0] * RETREAT_BACK
            return "retreat", np.array([obj[0] + back[0], obj[1] + back[1], up]), OPEN, False
        hover = np.array([c[0], c[1], c[2] + self.top + HOVER])
        if self.n_close == 0 and self.q is not None and self.q < Q_SHUT:           # closed on nothing
            return "reopen", np.array([self.eef[0], self.eef[1], max(self.eef[2], hover[2])]), OPEN, False
        if self.n_close > 0 or (self._at_grasp() and self._open()):
            return "close", None, CLOSE, False
        aligned = self.aligned0 and abs(self._yaw_err()) < YAW_TOL
        if np.linalg.norm(self.eef[:2] - c[:2]) > XY_TOL or not aligned:
            return "approach", hover, OPEN, True
        if not self._open():
            return "approach", None, OPEN, False
        return "descend", np.array([c[0], c[1], c[2]]), OPEN, False

    def step(self):
        stage, aim, grip, align = self.decide()
        o, yaw = self.o, self.base[2]
        u, v, wz, mode = np.zeros(3), np.zeros(2), 0.0, 0.0
        if aim is not None:
            d = (aim - self.root)[:2] if self.root is not None else np.zeros(2)
            if np.linalg.norm(d) > REACH:                       # out of reach: move the base, not the arm
                disp = _rz(-yaw)[:2, :2] @ (d / np.linalg.norm(d) * (np.linalg.norm(d) - REACH_GOAL))
                v, mode = np.clip(KP_BASE * disp / BASE_M, -1.0, 1.0), 1.0
                dw = _rz(yaw)[:2, :2] @ (v * BASE_M)
                for p in (self.eef, self.root, self.base):
                    p[:2] += dw
            else:
                u = KP * (_rz(-yaw) @ (aim - self.eef)) / EEF_M
                u *= min(1.0, min(1.0, o.vmax / EEF_M) / max(np.linalg.norm(u), 1e-9))   # straight path, <= vmax
                self.eef = self.eef + _rz(yaw) @ (u * EEF_M)
        if align and self.fax is not None and mode == 0.0:
            dyaw = float(np.clip(self._yaw_err(), -ROT_RATE, ROT_RATE))
            wz = float(np.clip(dyaw / o.eef_rad_per_unit, -1.0, 1.0))
            self.fax = _rz(wz * o.eef_rad_per_unit)[:2, :2] @ self.fax
        # gripper and object
        if grip == CLOSE:
            self.n_open = 0
            if stage == "close":
                self.n_close += 1
                if self.q is not None:
                    self.q = max(self.q - Q_RATE, self.half_w)
                if self.n_close >= CLOSE_STEPS:
                    self.grasped, self.offset = True, self.obj - self.eef
        else:
            self.n_open, self.n_close = self.n_open + 1, 0
            if self.q is not None:
                self.q = min(self.q + Q_RATE, Q_FULL)
            if self.grasped and self.n_open >= 3:
                self.grasped = False                            # object stays where it was released
        if self.grasped:
            self.obj = self.eef + self.offset
        return stage, np.concatenate([u, [0.0, 0.0, wz], [grip], v, [0.0, 0.0], [mode]])  # base yaw, torso = 0


def action_dict(rows) -> dict:
    """rows (n, 12) = eef(3) rot(3) gripper_close(1) base vx vy yaw torso(4) control_mode(1) -> wire dict
    with batch dim 1, as GR00T replies. Motion commands are saturated to [-1, 1]."""
    r = np.asarray(rows, dtype=float)
    f = np.float32
    return {"action.end_effector_position": np.clip(r[:, 0:3], -1, 1)[None].astype(f),
            "action.end_effector_rotation": np.clip(r[:, 3:6], -1, 1)[None].astype(f),
            "action.gripper_close": r[:, 6:7][None].astype(f),
            "action.base_motion": np.clip(r[:, 7:11], -1, 1)[None].astype(f),
            "action.control_mode": r[:, 11:12][None].astype(f)}


def plan(priv: dict, instruction: str, gripper_qpos=None, opts: Opts = Opts()) -> tuple[dict, list[str]]:
    """One 16-step chunk (wire dict, batch dim 1) and the stage of each step."""
    if instruction.strip().lower() == "stop":
        rows = np.zeros((CHUNK, 12))
        rows[:, 6] = CLOSE if priv["grasped"] else OPEN
        return action_dict(rows), ["stop"] * CHUNK
    m = _Model(priv, gripper_qpos, opts)
    stages, rows = [], []
    for _ in range(CHUNK):
        st, row = m.step()
        stages.append(st)
        rows.append(row)
    return action_dict(rows), stages


def _trace(n, seed, priv, gq, opts, stages) -> str:
    """One policy_server.log line per query: where the expert thinks it is (diagnosis on the instance)."""
    m = _Model(priv, gq, opts)
    runs = [s for i, s in enumerate(stages) if i == 0 or s != stages[i - 1]]
    d = None if m.tgt is None else round(float(np.linalg.norm(m.obj[:2] - m.tgt[:2])), 3)
    miss = [k for k in ("obj_bbox", "target_bbox", "finger_qpos", "finger_axis", "arm_root_pos") if priv.get(k) is None]
    return (f"q{n} seed={seed} {'>'.join(runs)} eef={np.round(m.eef, 3).tolist()} obj={np.round(m.obj, 3).tolist()} "
            f"obj_tgt_xy={d} grasped={priv.get('grasped')} lift={round(float(priv['obj_lift']), 3)} "
            f"q={None if m.q is None else round(m.q, 4)} yaw_err={round(m._yaw_err(), 2)} "
            f"quat_base={np.round(np.asarray(priv.get('eef_quat_base', [])), 3).tolist()} missing={miss}")


def main():
    import zmq
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5560)
    p.add_argument("--latency-ms", type=float, default=0.0, help="emulated inference time per query")
    p.add_argument("--vmax", type=float, default=Opts.vmax, help="eef path speed cap, m per control step")
    p.add_argument("--eef-rad-per-unit", type=float, default=Opts.eef_rad_per_unit,
                   help="eef yaw planning gain (rad per step per unit)")
    a = p.parse_args()
    opts = Opts(vmax=a.vmax, eef_rad_per_unit=a.eef_rad_per_unit)
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{a.port}")
    print(f"fake policy server (scripted expert, {opts}) on 127.0.0.1:{a.port}", flush=True)
    n = 0
    while True:
        req = TorchSerializer.from_bytes(sock.recv())
        ep = req.get("endpoint", "get_action")
        try:
            if ep == "ping":
                resp = {"status": "ok", "message": "Server is running"}
            elif ep == "kill":
                sock.send(TorchSerializer.to_bytes({"status": "ok"}))
                return
            elif ep == "get_modality_config":
                resp = {}
            elif ep == "stats":
                resp = {"n": n, "fake": True}
            elif ep == "get_action":
                data = req["data"]
                if "_priv" not in data:
                    raise ValueError("fake server needs data['_priv'] (run_episode --priv)")
                time.sleep(a.latency_ms / 1000.0)
                g = data.get("state.gripper_qpos")
                g = None if g is None else np.asarray(g).reshape(-1)
                priv = data["_priv"]
                resp, stages = plan(priv, str(data["annotation.human.task_description"][0]), g, opts)
                print(_trace(n, data.get("_seed"), priv, g, opts, stages), flush=True)
                n += 1
            else:
                resp = {"error": f"Unknown endpoint: {ep}"}
        except Exception as e:
            resp = {"error": repr(e)}
        sock.send(TorchSerializer.to_bytes(resp))


if __name__ == "__main__":
    main()
