"""Cancel edge cases at the skill-server level. The other two cases (early interrupt
clearance during recovery, recovery failure blocks the task branch) are BT behaviour and
live in test_bt.py."""
import pytest

from bt.client import FakeSkillClient
from preempt import action_layout as al
from preempt.arbiter import RECOVERY, Arbiter
from preempt.contracts import Ev, GoalStatus, StopMode
from rig import ListLog, Rig


def test_cancel_before_goal_acceptance_rejects_the_goal():
    # server level: the cancel overtakes its goal (separate ROS 2 services)
    r = Rig()
    assert r.skill.cancel(0, "g")                      # unknown id -> tombstone
    assert not r.skill.submit_goal(1, "g", "pick", StopMode.TRUNCATE_HOLD)
    assert r.skill.status_of("g") == GoalStatus.REJECTED
    r.run(1, 20)
    assert all(o is None for _, o, _ in r.applied) and r.arb.owner is None
    # client level: goal still in flight when the BT cancels
    r = Rig()
    c = FakeSkillClient(r.skill, r.log, accept_latency=2)
    c.send_goal(0, "pick", StopMode.BT_SAFE_ABORT)
    assert c.status() == GoalStatus.PENDING
    c.cancel(1)
    assert c.status() == GoalStatus.CANCELING          # WaitSkillIdle keeps waiting
    c.pump(2)
    assert c.status() == GoalStatus.REJECTED
    assert r.log.of(Ev.GOAL_REJECTED)[0]["reason"] == "cancelled_before_accept"
    assert r.arb.violations == 0


@pytest.mark.parametrize("mode", [StopMode.TRUNCATE_HOLD, StopMode.BT_SAFE_ABORT, StopMode.WAIT_CHUNK])
def test_late_inference_reply_after_fence_is_rejected(mode):
    r = Rig(latency=3, lockstep=False)
    r.skill.submit_goal(0, "g", "pick", mode)
    r.run(0, 12, at={1: lambda t: r.skill.cancel(t, "g")})
    assert r.log.of(Ev.Q, gen=1)[0]["t"] == 1           # an in-flight chunk is not committed
    late = r.log.of(Ev.LATE_REPLY, gen=1)
    assert len(late) == 1 and late[0]["fenced"] and late[0]["t"] == 4
    assert r.vla_steps(1) == [] and r.arb.violations == 0


def test_late_reply_with_language_stop_is_superseded_by_a_stop_query():
    r = Rig(latency=3, lockstep=False)
    r.skill.submit_goal(0, "g", "pick", StopMode.LANGUAGE_STOP)
    r.run(0, 12, at={1: lambda t: r.skill.cancel(t, "g")})
    assert len(r.log.of(Ev.LATE_REPLY, gen=1)) == 1
    assert [f["instruction"] for f in r.log.of(Ev.FEEDBACK)] == ["stop"]
    assert [q.instruction for q in r.policy.requests] == ["pick", "stop"]


@pytest.mark.parametrize("mode", list(StopMode))
def test_repeated_cancel_is_idempotent(mode):
    r = Rig()
    r.skill.submit_goal(0, "g", "pick", mode)
    again = lambda t: r.skill.cancel(t, "g")
    r.run(0, 300, at={40: again, 41: again, 45: again, 280: again})
    assert len(r.log.of(Ev.A, gen=1)) == 1
    assert len(r.log.of(Ev.Q, gen=1)) == 1 and len(r.log.of(Ev.CANCELED)) == 1
    assert len(r.log.of(Ev.CANCEL_IGNORED)) == 3
    assert r.skill.status_of("g") == GoalStatus.CANCELED and r.arb.violations == 0


def test_overlapping_goal_is_rejected_and_a_late_old_reply_cannot_leak():
    r = Rig()
    r.skill.submit_goal(0, "g1", "pick", StopMode.TRUNCATE_HOLD)
    r.run(0, 10)
    assert not r.skill.submit_goal(10, "g2", "pick", StopMode.TRUNCATE_HOLD)   # busy
    r.run(10, 20)
    assert len(r.vla_steps(1, after=10)) == 10                                # g1 unaffected

    r = Rig(latency=2, lockstep=False)
    r.skill.submit_goal(0, "g1", "pick", StopMode.TRUNCATE_HOLD)

    def cancel_and_resend(t):
        r.skill.cancel(t, "g1")
        assert r.skill.submit_goal(t, "g3", "pick", StopMode.TRUNCATE_HOLD)  # gen 2 after Q
    r.run(0, 20, at={1: cancel_and_resend})
    assert r.log.of(Ev.LATE_REPLY, gen=1)[0]["fenced"]
    assert r.vla_steps(1) == [] and r.vla_steps(2) != []
    assert r.arb.violations == 0


def test_goal_is_rejected_while_recovery_owns_the_command():
    r = Rig()
    r.skill.submit_goal(0, "g1", "pick", StopMode.BT_SAFE_ABORT)
    r.run(0, 5, at={3: lambda t: r.skill.cancel(t, "g1")})
    assert r.arb.owner == RECOVERY
    assert not r.skill.submit_goal(5, "g2", "pick", StopMode.BT_SAFE_ABORT)
    assert r.log.of(Ev.GOAL_REJECTED)[0]["reason"] == "owner_busy"


def test_arbiter_blocks_and_counts_an_old_generation_command_after_the_fence():
    log = ListLog()
    arb = Arbiter(log)
    assert arb.acquire(0, ("vla", 1))
    arb.fence(5, 1, "test")
    arb.offer(5, ("vla", 1), al.make(eef_pos=(1, 0, 0)))   # a buggy server keeps sending
    action, owner = arb.resolve(5)
    assert owner is None and (action == al.hold(al.GRIP_OPEN)).all()
    assert arb.violations == 1 and log.of(Ev.VIOLATION)[0]["kind"] == "after_fence"


def test_offer_made_before_a_release_in_the_same_tick_is_not_applied():
    arb = Arbiter(ListLog())
    assert arb.acquire(0, RECOVERY)
    arb.offer(0, RECOVERY, al.make(base_vel=(-0.5, 0, 0), base_mode=al.BASE_MODE))
    arb.release(0, RECOVERY)
    action, owner = arb.resolve(0)
    assert owner is None and (action == al.hold(al.GRIP_OPEN)).all()
