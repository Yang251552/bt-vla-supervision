"""Recovery motions inside the 12-D action space: pure functions of state -> action.

No joint commands exist in this space, so retract is an end-effector-space P-loop to the
reset eef pose. No teleport, no qpos writes: every motion is an ordinary env action.
`AbortSequencer` chains the stages (halt -> gripper -> lift -> retract -> back-off); the BT
node only drives it, so a ROS scripted-motion server can wrap the same object later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import action_layout as al
from .contracts import Ev, Phase, ProtocolConfig, SimState


@dataclass(frozen=True)
class Gains:
    """Calibration knobs (implementation, not protocol): env response per control step per unit command.

    Measured 2026-10-08 g4dn selftest (runs/c1/selftest.json "probe"; 10-step mean at |a| = 0.5 from rest,
    so it includes the ramp-up): eef 0.0118 m (base-frame x, arm mode), base forward 0.0041 m and yaw
    0.046 rad (base mode). The robosuite defaults assumed before were 0.05 m / 0.02 m / 0.5 rad.
    eef rotation was NOT probed: 0.5 rad is the OSC output_max, an upper bound on the true response, so the
    P-loops below are under-gained there (slower, never unstable). The arm cap keeps scripted arm motion
    <= 1 cm/step. The base probe is NOT the steady state: the dev-episode logs show 0.74 m/s (3.7 cm/step) at
    |a| = 1, so the base P-loops are about 9x over-gained and drive at full speed. Back-off tolerates that
    (it overshoots away from the counter); ReturnBase does not, so it runs a speed loop on the measured
    base speed instead (return_* below) and stops short of the pre-abort pose.
    """
    eef_m_per_unit: float = 0.0118   # measured 2026-10-08 g4dn selftest
    eef_rad_per_unit: float = 0.5    # NOT measured (upper bound); selftest probe eef_rotvec_* measures it next boot
    base_m_per_unit: float = 0.0041  # measured 2026-10-08 g4dn selftest (forward; lateral assumed equal)
    base_rad_per_unit: float = 0.046  # measured 2026-10-08 g4dn selftest (base yaw)
    kp: float = 0.3                  # fraction of the remaining error commanded per step
    max_arm_cmd: float = 0.85        # cap |a| for scripted arm motion (0.85 x 0.0118 = 1 cm/step)
    max_base_cmd: float = 1.0        # full scale = 4.1 mm/step
    gripper_settle_steps: int = 10
    backoff_margin_m: float = 0.02   # aim slightly past x so the base settles beyond it
    return_short_m: float = 0.05     # ReturnBase stops this far before the pre-abort pose (tolerance 0.10 m)
    return_v_max: float = 0.20       # m/s
    return_kv: float = 1.5           # desired speed per m of remaining distance, 1/s
    return_ki: float = 0.4           # command change per step per m/s of speed error (plant 0.08-0.74 m/s/unit)


# ---- quaternions, xyzw (robosuite convention) ----
def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def quat_inv(q):
    q = np.asarray(q, dtype=float)
    return np.array([-q[0], -q[1], -q[2], q[3]]) / float(q @ q)


def quat_to_rotvec(q):
    q = np.asarray(q, dtype=float) / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    s = np.linalg.norm(q[:3])
    if s < 1e-12:
        return 2.0 * q[:3]
    return q[:3] / s * 2.0 * np.arctan2(s, q[3])


def rotvec_to_quat(r):
    r = np.asarray(r, dtype=float)
    ang = np.linalg.norm(r)
    if ang < 1e-12:
        q = np.array([r[0] / 2, r[1] / 2, r[2] / 2, 1.0])
        return q / np.linalg.norm(q)
    return np.concatenate([r / ang * np.sin(ang / 2), [np.cos(ang / 2)]])


def rot_error(q_target, q_current) -> np.ndarray:
    """Axis-angle that takes q_current to q_target, applied on the left (OSC delta convention)."""
    return quat_to_rotvec(quat_mul(q_target, quat_inv(q_current)))


# ---- stage primitives ----
def still(s: SimState, cfg: ProtocolConfig) -> bool:
    return s.eef_speed < cfg.standstill_speed and s.base_speed < cfg.standstill_speed


def _p(err, per_unit, g: Gains, cap):
    return np.clip(g.kp * np.asarray(err, dtype=float) / per_unit, -cap, cap)


def move_eef_action(s: SimState, pos_target, quat_target, grip: float, g: Gains = Gains()):
    return al.make(eef_pos=_p(pos_target - s.eef_pos_base, g.eef_m_per_unit, g, g.max_arm_cmd),
                   eef_rot=_p(rot_error(quat_target, s.eef_quat_base), g.eef_rad_per_unit, g,
                              g.max_arm_cmd),
                   gripper=grip)


def at_pose(s: SimState, pos_target, quat_target, cfg: ProtocolConfig) -> bool:
    return (np.linalg.norm(pos_target - s.eef_pos_base) < cfg.retract_pos_tol_m and
            np.degrees(np.linalg.norm(rot_error(quat_target, s.eef_quat_base)))
            < cfg.retract_rot_tol_deg)


def backoff_distance(s: SimState, start_pose) -> float:
    """Planar displacement of the base backwards along the heading it had at `start_pose`."""
    heading = np.array([np.cos(start_pose[2]), np.sin(start_pose[2])])
    return float(-(s.base_pose[:2] - start_pose[:2]) @ heading)


def backoff_action(s: SimState, start_pose, x: float, grip: float, g: Gains = Gains()):
    """Base mode (arm moves with the base), drive straight back until >= x."""
    remaining = x + g.backoff_margin_m - backoff_distance(s, start_pose)
    vx = -float(np.clip(g.kp * remaining / g.base_m_per_unit, 0.0, g.max_base_cmd))
    return al.make(base_vel=(vx, 0.0, 0.0), base_mode=al.BASE_MODE, gripper=grip)


def return_base_action(s: SimState, target_pose, grip: float, u: float, g: Gains = Gains()):
    """Undo the back-off: drive straight forward along the pre-abort heading to return_short_m before
    `target_pose`, then command 0. Speed loop on the measured base speed (independent of the base gain),
    desired speed falls with the remaining distance. Returns (action, new command u)."""
    heading = np.array([np.cos(target_pose[2]), np.sin(target_pose[2])])
    left = float((np.asarray(target_pose[:2]) - s.base_pose[:2]) @ heading) - g.return_short_m
    v_des = min(g.return_kv * left, g.return_v_max)
    u = float(np.clip(u + g.return_ki * (v_des - s.base_speed), 0.0, g.max_base_cmd)) if v_des > 0 else 0.0
    return al.make(base_vel=(u, 0.0, 0.0), base_mode=al.BASE_MODE, gripper=grip), u


def return_base_past(s: SimState, target_pose) -> bool:
    """Base beyond `target_pose` along its heading. Tolerance 0, the same test as the published
    past_pre_abort_pose_m <= 0: the loop stops return_short_m before the pose, so a valid return never nears 0."""
    heading = np.array([np.cos(target_pose[2]), np.sin(target_pose[2])])
    return float((s.base_pose[:2] - np.asarray(target_pose[:2])) @ heading) > 0.0


def return_base_done(s: SimState, target_pose, cfg: ProtocolConfig) -> bool:
    return (np.linalg.norm(np.asarray(target_pose[:2]) - s.base_pose[:2]) < cfg.return_base_tol_m
            and still(s, cfg) and not return_base_past(s, target_pose))


def gripper_rule(phase: Phase, s: SimState, cfg: ProtocolConfig, last_grip: float):
    """-> (rule, gripper command, lift in m). Chosen once at abort start, then latched.

    TRANSPORT = secured (grasped AND lifted >= 2 cm): keep closed, lift 3 cm.
    GRASP     = past approach but not secured: open, lift 5 cm.
    APPROACH  : gripper unchanged, no lift.
    The live secured state wins over the latched phase: the phase latch lags by the
    hysteresis steps, and opening a secured object would drop it.
    """
    if cfg.secured(s):
        return Phase.TRANSPORT, al.GRIP_CLOSE, cfg.lift_transport_m
    if Phase(phase) in (Phase.APPROACH, Phase.DONE):
        return Phase.APPROACH, last_grip, 0.0
    return Phase.GRASP, al.GRIP_OPEN, cfg.lift_grasp_m


class AbortSequencer:
    """halt -> gripper -> lift -> retract -> back-off; fails if not done within T steps."""

    STAGES = ("halt", "gripper", "lift", "retract", "backoff", "done")

    def __init__(self, cfg: ProtocolConfig, t0: int, phase: Phase, s: SimState, last_grip: float,
                 home_pos, home_quat, log, gains: Gains = Gains()):
        self.cfg, self.g, self.log, self.t0 = cfg, gains, log, t0
        self.rule, self.grip, self.lift = gripper_rule(phase, s, cfg, last_grip)
        self.hold_grip = last_grip               # during halt the gripper keeps its command
        self.home_pos, self.home_quat = np.asarray(home_pos), np.asarray(home_quat)
        self.start_base = s.base_pose.copy()
        self.stage, self._n, self._lift_from = "halt", 0, None
        log(Ev.ABORT_START, t0, rule=self.rule, gripper=self.grip, lift_m=self.lift,
            base=self.start_base)

    def step(self, t: int, s: SimState):
        """-> (action, 'running' | 'done' | 'failed')."""
        if self.stage == "done":
            return al.hold(self.grip), "done"
        if t - self.t0 > self.cfg.abort_budget_T:
            self.log(Ev.ABORT_FAILED, t, stage=self.stage)
            return al.hold(self.hold_grip if self.stage == "halt" else self.grip), "failed"
        cfg, g = self.cfg, self.g
        if self.stage == "halt":
            self._n = self._n + 1 if still(s, cfg) else 0
            if self._n < cfg.standstill_steps:
                return al.hold(self.hold_grip), "running"
            self.stage, self._n = "gripper", 0
        if self.stage == "gripper":
            if self.grip != self.hold_grip and self._n < g.gripper_settle_steps:
                self._n += 1
                return al.hold(self.grip), "running"
            self.stage, self._lift_from = "lift", s.eef_pos_base.copy()
        if self.stage == "lift":
            target = self._lift_from + np.array([0.0, 0.0, self.lift])
            if self.lift > 0 and target[2] - s.eef_pos_base[2] > cfg.retract_pos_tol_m:
                return move_eef_action(s, target, s.eef_quat_base, self.grip, g), "running"
            self.stage = "retract"
        if self.stage == "retract":
            if not at_pose(s, self.home_pos, self.home_quat, cfg):
                return move_eef_action(s, self.home_pos, self.home_quat, self.grip, g), "running"
            self.log(Ev.RETRACTED, t)
            self.stage, self._n = "backoff", 0
        # backoff: drive back >= x, then wait for standstill
        if backoff_distance(s, self.start_base) < cfg.backoff_x_m or not still(s, cfg):
            self._n = 0
            return backoff_action(s, self.start_base, cfg.backoff_x_m, self.grip, g), "running"
        self._n += 1
        if self._n < cfg.standstill_steps:
            return al.hold(self.grip), "running"
        self.stage = "done"
        self.log(Ev.B, t, backoff_m=backoff_distance(s, self.start_base))
        return al.hold(self.grip), "done"
