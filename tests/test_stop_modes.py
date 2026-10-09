"""Stop-mode semantics: VLA actions applied after the cancel request R (control steps)."""
import numpy as np
import pytest

from preempt import action_layout as al
from preempt.arbiter import RECOVERY
from preempt.contracts import Ev, StopMode
from rig import Rig

CHUNK_START = 32  # chunks start at 0, 16, 32, ... when nothing interrupts


def cancel(rig, goal="g"):
    return lambda t: rig.skill.cancel(t, goal)


@pytest.mark.parametrize("offset", [0, 1, 8, 15])
def test_wait_chunk_finishes_the_committed_chunk(offset):
    r = Rig()
    r.skill.submit_goal(0, "g", "pick", StopMode.WAIT_CHUNK)
    tc = CHUNK_START + offset
    r.run(0, 80, at={tc: cancel(r)})
    assert len(r.vla_steps(1, after=tc)) == 16 - offset          # depends on the chunk offset
    assert r.log.of(Ev.Q, gen=1)[0]["t"] == tc + 16 - offset
    assert r.log.index(Ev.CANCELED) > r.log.index(Ev.Q)
    assert r.arb.violations == 0


@pytest.mark.parametrize("mode", [StopMode.TRUNCATE_HOLD, StopMode.BT_SAFE_ABORT])
def test_truncating_modes_execute_no_vla_action_after_the_request(mode):
    r = Rig()
    r.skill.submit_goal(0, "g", "pick", mode)
    tc = 130  # seed 0: object secured, gripper closed
    r.run(0, 160, at={tc: cancel(r)})
    assert r.vla_steps(1, after=tc) == []
    assert r.log.of(Ev.Q, gen=1)[0]["t"] == tc == r.log.of(Ev.A, gen=1)[0]["t"]
    assert r.log.index(Ev.CANCELED) > r.log.index(Ev.Q)
    assert r.arb.owner == (RECOVERY if mode == StopMode.BT_SAFE_ABORT else None)
    # nobody offers recovery commands in this rig, so both modes hold
    for _, owner, a in [x for x in r.applied if x[0] >= tc]:
        assert owner is None
        assert np.allclose(a, al.hold(al.GRIP_CLOSE))   # zero delta, zero base, arm mode, gripper kept
    assert r.env.grasped


def test_language_stop_keeps_the_chunk_and_requeries_with_stop_within_W():
    ref = Rig()
    ref.skill.submit_goal(0, "g", "pick", StopMode.LANGUAGE_STOP)
    ref.run(0, 48)
    r = Rig()
    r.skill.submit_goal(0, "g", "pick", StopMode.LANGUAGE_STOP)
    tc = CHUNK_START + 5
    r.run(0, tc + r.cfg.window_W + 20, at={tc: cancel(r)})
    # no truncation: the rest of the current chunk is executed exactly as without the cancel
    for (t, o, a), (t2, o2, a2) in zip(r.applied[tc:48], ref.applied[tc:48]):
        assert o == o2 == ("vla", 1) and np.allclose(a, a2)
    chunks = [f for f in r.log.of(Ev.FEEDBACK) if f["t"] > tc]
    assert chunks[0]["t"] == 48                                   # first re-query at the next boundary
    assert all(f["instruction"] == "stop" for f in chunks)
    assert [f["t"] for f in chunks] == list(range(48, tc + r.cfg.window_W, 16))
    assert r.log.of(Ev.Q, gen=1)[0]["t"] == tc + r.cfg.window_W
    assert len(r.vla_steps(1, after=tc)) == r.cfg.window_W
    assert r.arb.violations == 0
