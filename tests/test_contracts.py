import dataclasses
from pathlib import Path

import numpy as np
import pytest

from preempt import action_layout as al
from preempt.contracts import ProtocolConfig

ROOT = Path(__file__).resolve().parents[1]


def test_config_roundtrip_and_hash(tmp_path):
    cfg = ProtocolConfig()
    cfg.save(tmp_path / "p.json")
    assert ProtocolConfig.load(tmp_path / "p.json").hash() == cfg.hash()
    assert dataclasses.replace(cfg, abort_budget_T=161).hash() != cfg.hash()
    (tmp_path / "bad.json").write_text('{"abort_budget_T": 160, "typo_key": 1}')
    with pytest.raises(ValueError):
        ProtocolConfig.load(tmp_path / "bad.json")


def test_committed_protocol_file_matches_code_defaults():
    # changing a protocol value must be done in both places, deliberately
    assert ProtocolConfig.load(ROOT / "config" / "protocol.json") == ProtocolConfig()


def test_protocol_values_from_spec():
    c = ProtocolConfig()
    assert (c.grasp_dist_m, c.secured_lift_m, c.hysteresis_steps) == (0.10, 0.02, 5)
    assert (c.k_approach, c.k_grasp, c.k_transport) == (40, 10, 20)
    assert (c.backoff_x_m, c.force_limit_n, c.abort_budget_T, c.window_W, c.reentry_budget_B) == \
        (0.20, 50.0, 184, 200, 750)  # T calibrated on the scripted-expert dev runs (max 92 x 2); x, F, W, B unchanged
    assert (c.chunk_len, c.retract_pos_tol_m, c.retract_rot_tol_deg) == (16, 0.01, 5.0)
    assert (c.lift_grasp_m, c.lift_transport_m, c.return_base_tol_m) == (0.05, 0.03, 0.10)


def test_action_layout_fields_and_hold():
    a = al.make(eef_pos=(0.1, 0.2, 0.3), eef_rot=(0.4, 0.5, 0.6), gripper=al.GRIP_CLOSE,
                base_vel=(-0.1, 0.0, 0.2), torso=0.0, base_mode=al.BASE_MODE)
    assert a.shape == (al.DIM,)
    assert np.allclose(al.get(a, "eef_rot"), (0.4, 0.5, 0.6)) and al.get(a, "base_mode")[0] == 1
    h = al.hold(al.GRIP_CLOSE)
    assert not h[al.LAYOUT["eef_pos"]].any() and not h[al.LAYOUT["base_vel"]].any()
    assert al.get(h, "base_mode")[0] == al.ARM_MODE and al.get(h, "gripper")[0] == al.GRIP_CLOSE


def test_from_gr00t_follows_robocasa_unmap_action():
    n = 16
    d = {"action.end_effector_position": np.full((n, 3), 0.1),
         "action.end_effector_rotation": np.full((n, 3), -0.2),
         "action.gripper_close": np.r_[np.zeros(8), np.ones(8)].reshape(n, 1),
         "action.base_motion": np.tile([0.3, 0.0, -0.1, 0.05], (n, 1)),
         "action.control_mode": np.ones((n, 1))}
    out = al.from_gr00t(d)
    assert out.shape == (n, al.DIM)
    assert (al.get(out, "gripper")[:8] == al.GRIP_OPEN).all() and (al.get(out, "gripper")[8:] == al.GRIP_CLOSE).all()
    assert np.allclose(al.get(out, "base_vel")[0], (0.3, 0.0, -0.1)) and np.allclose(al.get(out, "torso"), 0.05)
    assert (al.get(out, "base_mode") == al.BASE_MODE).all()


def test_layout_check_against_env_spec():
    ok = {"robot0_right": (0, 6), "robot0_right_gripper": (6, 7), "robot0_base": (7, 10),
          "robot0_torso": (10, 11), "robot0_left": (11, 11)}
    assert al.check_against_env(ok, 12) == []
    swapped = {"robot0_base": (0, 3), "robot0_torso": (3, 4), "robot0_right": (4, 10),
               "robot0_right_gripper": (10, 11)}
    assert len(al.check_against_env(swapped, 12)) == 4
    assert al.check_against_env(ok, 13)
