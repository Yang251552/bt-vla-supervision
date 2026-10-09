import numpy as np
import pytest

from preempt import action_layout as al
from preempt.contracts import Phase, ProtocolConfig, SimState
from preempt.recovery import (AbortSequencer, at_pose, backoff_action, backoff_distance,
                              gripper_rule, move_eef_action, return_base_action, return_base_done,
                              rotvec_to_quat, quat_mul)
from preempt.stub import HOME_POS, HOME_QUAT, StubEnv
from rig import ListLog

CFG = ProtocolConfig()


def st(grasped=False, lift=0.0, speed=0.0):
    z = np.zeros(3)
    return SimState(z, np.array([0, 0, 0, 1.0]), z, z, z, lift, grasped, speed, speed, 0.0, False)


@pytest.mark.parametrize("phase,state,want", [
    (Phase.APPROACH, st(), (Phase.APPROACH, al.GRIP_OPEN, 0.0)),            # gripper unchanged
    (Phase.GRASP, st(grasped=True, lift=0.01), (Phase.GRASP, al.GRIP_OPEN, 0.05)),
    (Phase.TRANSPORT, st(grasped=True, lift=0.05), (Phase.TRANSPORT, al.GRIP_CLOSE, 0.03)),
    (Phase.GRASP, st(grasped=True, lift=0.02), (Phase.TRANSPORT, al.GRIP_CLOSE, 0.03)),  # live secured wins
    (Phase.TRANSPORT, st(grasped=False, lift=0.05), (Phase.GRASP, al.GRIP_OPEN, 0.05)),  # slipped
])
def test_phase_aware_gripper_rule(phase, state, want):
    assert gripper_rule(phase, state, CFG, last_grip=al.GRIP_OPEN) == want


def test_halt_holds_with_the_last_gripper_until_standstill():
    seq = AbortSequencer(CFG, 0, Phase.TRANSPORT, st(grasped=True, lift=0.05), al.GRIP_CLOSE,
                         HOME_POS, HOME_QUAT, ListLog())
    for t in range(10):
        a, status = seq.step(t, st(grasped=True, lift=0.05, speed=0.1))
        assert status == "running" and np.allclose(a, al.hold(al.GRIP_CLOSE))
    for t in range(10, 10 + CFG.standstill_steps):
        seq.step(t, st(grasped=True, lift=0.05))
    assert seq.stage != "halt"


def test_retract_reaches_reset_pose_in_eef_space_including_rotation():
    env = StubEnv(0)
    env.eef_b = HOME_POS + [0.25, -0.15, -0.2]
    env.quat_b = quat_mul(rotvec_to_quat([0.0, 0.0, 0.6]), HOME_QUAT)
    for _ in range(120):
        a = move_eef_action(env.state(), HOME_POS, HOME_QUAT, al.GRIP_OPEN)
        assert a.shape == (12,) and np.all(np.abs(a) <= 1) and al.get(a, "base_mode")[0] == al.ARM_MODE
        env.step(a)
    assert at_pose(env.state(), HOME_POS, HOME_QUAT, CFG)


def test_backoff_moves_base_with_arm_and_return_base_comes_back():
    env = StubEnv(0)
    start = env.state().base_pose.copy()
    for _ in range(80):
        a = backoff_action(env.state(), start, CFG.backoff_x_m, al.GRIP_CLOSE)
        assert al.get(a, "base_mode")[0] == al.BASE_MODE and al.get(a, "gripper")[0] == al.GRIP_CLOSE
        env.step(a)
    assert backoff_distance(env.state(), start) >= CFG.backoff_x_m
    eef_b = env.state().eef_pos_base
    assert np.allclose(eef_b, HOME_POS)                    # arm rode along with the base
    u = 0.0
    for _ in range(120):
        if return_base_done(env.state(), start, CFG):
            break
        a, u = return_base_action(env.state(), start, al.GRIP_CLOSE, u)
        env.step(a)
    assert return_base_done(env.state(), start, CFG)
    assert backoff_distance(env.state(), start) >= 0.0           # stopped short of the pre-abort pose
