"""The single table mapping the 12-D PandaOmron env action vector to named fields.

UNVERIFIED: the index ORDER below is an assumption. robosuite builds it at runtime from
`robot.composite_controller._action_split_indexes` (part order comes from the robot's
controller config), so the AWS selftest must call `check_against_env` on the real env.

Verified from source (robosuite composite_controller.py / robocasa gym wrapper):
- base_mode is the LAST dim (HybridMobileBase.action_limits appends one entry);
  > 0 = base mode (arm goal tracks the moving base), <= 0 = arm mode (OSC goal "achieved").
- gripper: -1 = open, +1 = close (unmap_action: gripper_close < 0.5 -> -1).
- OSC default: input_type="delta", input_ref_frame="base", output_max 0.05 m / 0.5 rad per unit.
UNVERIFIED: base velocity frame and signs (assumed robot frame, +x forward, +yaw CCW);
robocasa's PandaOmron OSC config may differ from the robosuite default above.
"""
from __future__ import annotations

import numpy as np

DIM = 12
ARM_MODE, BASE_MODE = -1.0, 1.0
GRIP_OPEN, GRIP_CLOSE = -1.0, 1.0

# field -> slice. UNVERIFIED order (see module docstring).
LAYOUT = {
    "eef_pos": slice(0, 3),
    "eef_rot": slice(3, 6),
    "gripper": slice(6, 7),
    "base_vel": slice(7, 10),
    "torso": slice(10, 11),
    "base_mode": slice(11, 12),
}
# robosuite part name (robot0 prefix dropped) -> fields it covers, in order.
PARTS = {"right": ("eef_pos", "eef_rot"), "right_gripper": ("gripper",),
         "base": ("base_vel",), "torso": ("torso",)}


def make(eef_pos=(0, 0, 0), eef_rot=(0, 0, 0), gripper=GRIP_OPEN, base_vel=(0, 0, 0),
         torso=0.0, base_mode=ARM_MODE) -> np.ndarray:
    a = np.zeros(DIM)
    a[LAYOUT["eef_pos"]] = eef_pos
    a[LAYOUT["eef_rot"]] = eef_rot
    a[LAYOUT["gripper"]] = gripper
    a[LAYOUT["base_vel"]] = base_vel
    a[LAYOUT["torso"]] = torso
    a[LAYOUT["base_mode"]] = base_mode
    return np.clip(a, -1.0, 1.0)


def get(a: np.ndarray, field: str) -> np.ndarray:
    return np.asarray(a)[..., LAYOUT[field]]


def hold(gripper: float) -> np.ndarray:
    """Hold = zero eef delta in arm mode, zero base velocity, gripper keeps its last command."""
    return make(gripper=gripper)


def from_gr00t(d: dict) -> np.ndarray:
    """GR00T action dict (chunk, k) -> (chunk, 12), following robocasa's unmap_action."""
    pos = np.asarray(d["action.end_effector_position"], dtype=float)
    n = pos.shape[0]
    out = np.zeros((n, DIM))
    out[:, LAYOUT["eef_pos"]] = pos
    out[:, LAYOUT["eef_rot"]] = np.asarray(d["action.end_effector_rotation"], dtype=float)
    out[:, LAYOUT["gripper"]] = np.where(
        np.asarray(d["action.gripper_close"], dtype=float).reshape(n, 1) < 0.5, GRIP_OPEN, GRIP_CLOSE)
    bm = np.asarray(d["action.base_motion"], dtype=float)
    out[:, LAYOUT["base_vel"]] = bm[:, 0:3]
    out[:, LAYOUT["torso"]] = bm[:, 3:4]
    out[:, LAYOUT["base_mode"]] = np.where(
        np.asarray(d["action.control_mode"], dtype=float).reshape(n, 1) < 0.5, ARM_MODE, BASE_MODE)
    return out


def check_against_env(split_indexes: dict, action_dim: int) -> list[str]:
    """Compare LAYOUT with the real env. Pass `cc._action_split_indexes` and `env.action_dim`.

    Keys may carry the robot prefix ("robot0_right") or not. Returns a list of mismatches
    (empty = layout confirmed). The AWS selftest must fail if this is non-empty.
    """
    problems = []
    if action_dim != DIM:
        problems.append(f"action_dim {action_dim} != {DIM}")
    got = {k.split("robot0_")[-1]: tuple(v) for k, v in split_indexes.items()}
    for part, fields in PARTS.items():
        want = (LAYOUT[fields[0]].start, LAYOUT[fields[-1]].stop)
        if got.get(part) != want:
            problems.append(f"{part}: env {got.get(part)} != layout {want}")
    extra = set(got) - set(PARTS)
    extra = {p for p in extra if got[p][0] != got[p][1]}  # zero-width parts are harmless
    if extra:
        problems.append(f"unexpected parts with width: {sorted(extra)}")
    if LAYOUT["base_mode"].start != action_dim - 1:
        problems.append("base_mode is not the last dim")
    return problems
