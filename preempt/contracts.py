"""Shared names and the frozen protocol config (config/protocol.json)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class StopMode(str, Enum):
    WAIT_CHUNK = "wait_chunk"          # M1: finish the committed chunk, then hold
    TRUNCATE_HOLD = "truncate_hold"    # M2: hold from the next step
    LANGUAGE_STOP = "language_stop"    # M3: keep chunk, re-query with "stop" at later boundaries within W
    BT_SAFE_ABORT = "bt_safe_abort"    # M4: truncate like M2, hand ownership to recovery


class Phase(str, Enum):
    APPROACH = "approach"
    GRASP = "grasp"
    TRANSPORT = "transport"
    DONE = "done"


class GoalStatus(str, Enum):
    """Mirrors the ROS 2 action goal lifecycle (PENDING = sent, not yet accepted)."""
    PENDING = "pending"
    ACTIVE = "active"
    CANCELING = "canceling"
    SUCCEEDED = "succeeded"
    CANCELED = "canceled"
    ABORTED = "aborted"
    REJECTED = "rejected"


BUSY = (GoalStatus.PENDING, GoalStatus.ACTIVE, GoalStatus.CANCELING)


class Ev:
    """Event names in the JSONL log."""
    I = "I"    # interrupt injected (start of control step t_I)
    R = "R"    # cancel requested by the supervisor
    A = "A"    # cancel accepted by the skill server
    Q = "Q"    # command ownership relinquished for a generation (fence)
    E = "E"    # last old-generation VLA action applied (logged at fence time)
    H = "H"    # robot at standstill (eef and base speed < threshold, N steps)
    B = "B"    # safe-abort complete (retracted, backed off >= x, standstill)
    U = "U"    # interrupt cleared
    GOAL_ACCEPTED = "goal_accepted"
    GOAL_REJECTED = "goal_rejected"
    FEEDBACK = "feedback"              # one per chunk that starts executing
    CANCELING = "canceling"
    CANCELED = "canceled"
    GOAL_SUCCEEDED = "goal_succeeded"
    GOAL_ABORTED = "goal_aborted"
    CANCEL_IGNORED = "cancel_ignored"  # repeated / late cancel, no state change
    REENTRY_GOAL = "reentry_goal"
    FIRST_FRESH_ACTION = "first_fresh_action"  # first applied action of a generation
    SUCCESS = "success"                # task success first observed
    PHASE = "phase"
    LATE_REPLY = "late_reply"          # inference reply rejected (fenced / stale)
    VIOLATION = "violation"            # non-owner offered a command (old gen after fence, or overlap)
    ABORT_START = "abort_start"        # latched gripper rule, lift, pre-abort base pose
    RETRACTED = "retracted"
    ABORT_FAILED = "abort_failed"
    RETURN_BASE_DONE = "return_base_done"
    RETURN_BASE_PAST = "return_base_past"  # base past the pre-abort pose along its heading: ReturnBase fails
    STEP = "step"                      # one per control step: applied command source + state summary
    EPISODE = "episode"                # header line: config hash + meta
    EPISODE_END = "episode_end"


@dataclass(frozen=True)
class SimState:
    """What the sim server reports each control step (privileged state). Quaternions are xyzw."""
    eef_pos_base: np.ndarray    # (3,) eef position in robot base frame
    eef_quat_base: np.ndarray   # (4,) xyzw
    eef_pos_world: np.ndarray   # (3,)
    base_pose: np.ndarray       # (3,) x, y, yaw in world
    obj_pos: np.ndarray         # (3,) target object, world
    obj_lift: float             # object height minus its height at reset (m)
    grasped: bool               # check_obj_grasped
    eef_speed: float            # m/s, world
    base_speed: float           # m/s, planar
    contact_force: float        # N, robot/held-object vs environment, target-object/gripper excluded
    success: bool               # task's own success check

    @property
    def eef_obj_dist(self) -> float:
        return float(np.linalg.norm(self.eef_pos_world - self.obj_pos))


@dataclass(frozen=True)
class ProtocolConfig:
    control_hz: int = 20
    chunk_len: int = 16
    action_dim: int = 12
    # phase boundaries (5-step hysteresis, monotonic latch)
    grasp_dist_m: float = 0.10
    secured_lift_m: float = 0.02
    hysteresis_steps: int = 5
    # injection offsets after phase entry
    k_approach: int = 40
    k_grasp: int = 10
    k_transport: int = 20
    # safe-abort / criteria
    backoff_x_m: float = 0.20
    force_limit_n: float = 50.0
    force_avg_steps: int = 3
    abort_budget_T: int = 184       # 2026-10-08: max(B - t_I) = 92 steps over 15 g4dn scripted-expert dev episodes, x 2
    T_provisional: bool = False     # calibrated before the first GR00T episode (see abort_budget_T)
    window_W: int = 200
    reentry_budget_B: int = 750
    fence_slack_steps: int = 1      # criterion 1: Q - R <= 1
    standstill_speed: float = 0.02  # m/s, eef and base
    standstill_steps: int = 5
    retract_pos_tol_m: float = 0.01
    retract_rot_tol_deg: float = 5.0
    lift_grasp_m: float = 0.05
    lift_transport_m: float = 0.03
    return_base_tol_m: float = 0.10
    stop_instruction: str = "stop"

    def k(self, phase: Phase) -> int:
        return {Phase.APPROACH: self.k_approach, Phase.GRASP: self.k_grasp,
                Phase.TRANSPORT: self.k_transport}[Phase(phase)]

    def secured(self, s: SimState) -> bool:
        """"Held/secured" = grasped AND lifted >= 2 cm (user decision 10-08)."""
        return bool(s.grasped and s.obj_lift >= self.secured_lift_m)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))

    def hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(dataclasses.asdict(self), sort_keys=True, indent=1) + "\n")

    @classmethod
    def load(cls, path) -> "ProtocolConfig":
        data = json.loads(Path(path).read_text())
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown protocol keys: {sorted(unknown)}")
        return cls(**data)
