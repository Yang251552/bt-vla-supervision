"""Deterministic kinematic stand-in for RoboCasa + a scripted chunked policy.

Not physics: first-order velocity lag on eef and base (so halting takes a few steps),
grasp when the gripper is closed within GRASP_RADIUS, object follows the eef while grasped
and drops to the counter on release. The eef is rigidly attached to the base frame in both
base modes (simplification of HybridMobileBase). Success = released inside the target region.
"""
from __future__ import annotations

import copy

import numpy as np

from . import action_layout as al
from .contracts import SimState
from .recovery import Gains, quat_mul, rotvec_to_quat
from .skill import ChunkReply, ChunkRequest

DT = 0.05
_G = Gains()   # the stub responds exactly as recovery.Gains says the real env does (steady state)
EEF_M_PER_UNIT, EEF_RAD_PER_UNIT = _G.eef_m_per_unit, _G.eef_rad_per_unit
BASE_M_PER_UNIT, BASE_RAD_PER_UNIT = _G.base_m_per_unit, _G.base_rad_per_unit
LAG = 0.5
GRASP_RADIUS = 0.012
OBJ_H = 0.03                 # object centre above the counter (counter top = z 0)
FINGER = 0.03                # finger tips below the eef point
K_CONTACT = 5000.0           # N per m of penetration into the counter
HOME_POS = np.array([0.40, 0.0, 0.30])         # reset eef pose, base frame
HOME_QUAT = np.array([1.0, 0.0, 0.0, 0.0])     # xyzw, gripper pointing down
TARGET_HALF = 0.05


class StubEnv:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.base = np.zeros(3)
        self.eef_b = HOME_POS.copy()
        self.quat_b = HOME_QUAT.copy()
        self.v_eef = np.zeros(3)
        self.v_base = np.zeros(3)
        self.obj = np.array([0.70 + rng.uniform(-0.05, 0.05), -0.20 + rng.uniform(-0.05, 0.05), OBJ_H])
        self.obj_z0 = OBJ_H
        self.target = np.array([0.60 + rng.uniform(-0.05, 0.05), 0.30 + rng.uniform(-0.05, 0.05), OBJ_H])
        self.grasped = False
        self.offset = np.zeros(3)
        self.gripper = al.GRIP_OPEN
        self._eef_speed = 0.0
        self._base_speed = 0.0

    def eef_world(self) -> np.ndarray:
        c, s = np.cos(self.base[2]), np.sin(self.base[2])
        x, y, z = self.eef_b
        return np.array([self.base[0] + c * x - s * y, self.base[1] + s * x + c * y, z])

    def step(self, action) -> None:
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        eef0, base0 = self.eef_world(), self.base[:2].copy()
        self.v_eef += LAG * (al.get(a, "eef_pos") * EEF_M_PER_UNIT - self.v_eef)
        self.eef_b = self.eef_b + self.v_eef
        self.quat_b = quat_mul(rotvec_to_quat(al.get(a, "eef_rot") * EEF_RAD_PER_UNIT), self.quat_b)
        bv = al.get(a, "base_vel") * np.array([BASE_M_PER_UNIT, BASE_M_PER_UNIT, BASE_RAD_PER_UNIT])
        self.v_base += LAG * (bv - self.v_base)
        c, s = np.cos(self.base[2]), np.sin(self.base[2])
        vx, vy, wz = self.v_base
        self.base = self.base + np.array([c * vx - s * vy, s * vx + c * vy, wz])
        self.gripper = float(al.get(a, "gripper")[0])
        eef = self.eef_world()
        if self.gripper > 0 and not self.grasped and np.linalg.norm(eef - self.obj) < GRASP_RADIUS:
            self.grasped, self.offset = True, self.obj - eef
        elif self.gripper <= 0 and self.grasped:
            self.grasped = False
            self.obj = np.array([self.obj[0], self.obj[1], OBJ_H])  # drops onto the counter
        if self.grasped:
            self.obj = eef + self.offset
        self._eef_speed = float(np.linalg.norm(eef - eef0) / DT)
        self._base_speed = float(np.linalg.norm(self.base[:2] - base0) / DT)

    def success(self) -> bool:
        return (not self.grasped and abs(self.obj[2] - OBJ_H) < 1e-6
                and bool(np.all(np.abs(self.obj[:2] - self.target[:2]) <= TARGET_HALF)))

    def state(self) -> SimState:
        eef = self.eef_world()
        pen = max(0.0, FINGER - eef[2])
        if self.grasped:
            pen = max(pen, OBJ_H - self.obj[2])
        return SimState(eef_pos_base=self.eef_b.copy(), eef_quat_base=self.quat_b.copy(),
                        eef_pos_world=eef, base_pose=self.base.copy(), obj_pos=self.obj.copy(),
                        obj_lift=float(self.obj[2] - self.obj_z0), grasped=self.grasped,
                        eef_speed=self._eef_speed, base_speed=self._base_speed,
                        contact_force=K_CONTACT * pen, success=self.success())


class FakePolicy:
    """Scripted chunked policy: plans 16 steps open-loop by rolling a copy of the env.

    `obs` in a request must be a StubEnv. Replies arrive after `latency` polls.
    With obey_stop, instruction "stop" yields a hold chunk; otherwise it is ignored.
    """
    V_MAX_M, KP, HOVER, CARRY = 0.004, 0.3, 0.08, 0.08   # V_MAX_M: eef speed cap, m per step

    def __init__(self, chunk_len: int = 16, latency: int = 0, obey_stop: bool = True,
                 stop_instruction: str = "stop"):
        self.chunk_len, self.latency, self.obey_stop = chunk_len, latency, obey_stop
        self.stop_instruction = stop_instruction
        self.requests: list[ChunkRequest] = []
        self._pending: list[list] = []

    def request(self, req: ChunkRequest) -> None:
        self.requests.append(req)
        env = copy.deepcopy(req.obs)
        chunk = []
        for _ in range(self.chunk_len):
            a = self.reactive(env, req.instruction)
            chunk.append(a)
            env.step(a)
        self._pending.append([self.latency, ChunkReply(req.gen, req.query_idx, np.array(chunk))])

    def poll(self) -> list[ChunkReply]:
        ready = [r for n, r in self._pending if n <= 0]
        self._pending = [[n - 1, r] for n, r in self._pending if n > 0]
        return ready

    def _to(self, env: StubEnv, aim_world, grip):
        c, s = np.cos(env.base[2]), np.sin(env.base[2])
        e = aim_world - env.eef_world()
        e_b = np.array([c * e[0] + s * e[1], -s * e[0] + c * e[1], e[2]])
        cap = self.V_MAX_M / EEF_M_PER_UNIT
        return al.make(eef_pos=np.clip(self.KP * e_b / EEF_M_PER_UNIT, -cap, cap),
                       gripper=grip)

    def reactive(self, env: StubEnv, instruction: str) -> np.ndarray:
        if (instruction == self.stop_instruction and self.obey_stop) or env.success():
            return al.hold(env.gripper)
        eef, obj, tgt = env.eef_world(), env.obj, env.target
        if not env.grasped:
            if env.gripper > 0:
                return al.hold(al.GRIP_OPEN)  # re-open after a released / missed grasp
            if np.linalg.norm(obj[:2] - eef[:2]) > 0.01:
                return self._to(env, obj + np.array([0, 0, self.HOVER]), al.GRIP_OPEN)
            if np.linalg.norm(obj - eef) > 0.008:
                return self._to(env, obj, al.GRIP_OPEN)
            return self._to(env, eef, al.GRIP_CLOSE)
        carry_z = env.obj_z0 + self.CARRY
        far = np.linalg.norm(tgt[:2] - obj[:2]) > 0.01
        if far and obj[2] < carry_z - 0.005 and np.linalg.norm(tgt[:2] - obj[:2]) > 0.1:
            want = np.array([obj[0], obj[1], carry_z])
        elif far:
            want = np.array([tgt[0], tgt[1], carry_z])
        else:
            want = np.array([tgt[0], tgt[1], OBJ_H])
            if obj[2] - OBJ_H < 0.004:
                return al.hold(al.GRIP_OPEN)
        return self._to(env, eef + (want - obj), al.GRIP_CLOSE)
