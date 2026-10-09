"""rclpy-free core of the skill_host <-> bt_runner lockstep protocol (wire version WIRE).

Shared by both nodes and exercised on the stub by tests/test_lockstep.py, which replays the protocol in one
process (same message order, no transport) and checks it reproduces the stub results.

/bt/state(t)  skill_host -> bt_runner  {"v","t","state","phase","interrupt","goals":{goal_id: status},
                                        "owner","last_gripper","lang"} or {"v","t","end": true}
/bt/ack(t)    bt_runner -> skill_host  {"v","kind":"tick","t","ops","events","requests","root","task",
                                        "abort_failed","goals_sent","tip","fb_rx","cfg_hash"}
                                       then, if requests > 0, {"v","kind":"done","t"} once every goal /
                                       cancel response of the tick is in.
"""
from __future__ import annotations

import uuid
from typing import Optional

import numpy as np

from preempt import action_layout as al
from preempt.contracts import Ev, GoalStatus, StopMode

WIRE = 1


class Forward:
    """Ctx.log for the BT: events travel in the ack and skill_host writes them into the episode log."""

    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, ev, t, **fields):
        self.events.append(dict(fields, ev=ev, t=int(t)))


class RemoteArbiter:
    """What bt/tree.py uses of preempt.arbiter.Arbiter (owner, last_gripper, acquire, release, offer).
    Reads mirror skill_host's arbiter at the start of the tick; writes are queued as ops and applied by
    skill_host at the same control step, so skill_host stays the single command owner."""

    def __init__(self):
        self.owner, self.last_gripper, self.ops = None, al.GRIP_OPEN, []

    def sync(self, owner, last_gripper: float) -> None:
        self.owner = None if owner is None else tuple(owner)
        self.last_gripper, self.ops = float(last_gripper), []

    def acquire(self, t, owner) -> bool:
        if self.owner is not None:
            return False
        self.owner = tuple(owner)
        self.ops.append(["acquire", list(owner)])
        return True

    def release(self, t, owner) -> None:
        if self.owner == tuple(owner):
            self.owner = None
            self.ops.append(["release", list(owner)])

    def offer(self, t, owner, action) -> None:
        self.ops.append(["offer", list(owner), [float(x) for x in action]])


class DeferredSkillClient:
    """bt.client.SkillClient across a process boundary. Goal and cancel requests made during a tick are queued
    and sent by flush() after the tick's ack (a subclass implements flush). Status, as the ROS action client
    sees it: result if any; REJECTED; CANCELING once a cancel was sent; PENDING until the goal response;
    ACTIVE. Like FakeSkillClient, R (cancel requested) is logged here, every time, with a repeat flag."""

    def __init__(self, log):
        self.log = log
        self.goal_id: Optional[str] = None
        self.goals: dict[str, dict] = {}
        self.goal_out: Optional[tuple[str, str, str]] = None    # (goal_id, instruction, stop_mode)
        self.cancel_out: list[str] = []
        self.fb = None

    def send_goal(self, t, instruction, stop_mode) -> str:
        gid = uuid.uuid4().hex
        self.goal_id = gid
        self.goals[gid] = dict(handle=None, accepted=None, result=None, cancel_sent=False, cancel_on_accept=False)
        self.goal_out = (gid, instruction, StopMode(stop_mode).value)
        return gid

    def cancel(self, t) -> None:
        if self.goal_id is None:
            return
        g = self.goals[self.goal_id]
        self.log(Ev.R, t, goal_id=self.goal_id, repeat=g["cancel_sent"])
        g["cancel_sent"] = True
        if self.goal_id not in self.cancel_out:
            self.cancel_out.append(self.goal_id)

    def status(self) -> Optional[GoalStatus]:
        return None if self.goal_id is None else self.status_of(self.goal_id)

    def status_of(self, gid) -> GoalStatus:
        g = self.goals[gid]
        if g["result"] is not None:
            return g["result"]
        if g["accepted"] is False:
            return GoalStatus.REJECTED
        if g["cancel_sent"]:
            return GoalStatus.CANCELING
        return GoalStatus.PENDING if g["accepted"] is None else GoalStatus.ACTIVE

    def feedback(self):
        return self.fb

    def pending(self) -> int:
        return int(self.goal_out is not None) + len(self.cancel_out)

    def caught_up(self, reported: dict) -> bool:
        """Barrier: True once this client's view of every goal it sent equals skill_host's report."""
        return all(self.status_of(g).value == st for g, st in reported.items() if g in self.goals)


# ---- skill_host side ----
def goal_reject_reason(skill, arb, stop_mode: str) -> Optional[str]:
    """SkillController.submit_goal's own checks, without its side effects (rclpy's goal_callback does not get
    the goal id, so the goal is judged here and submitted once its handle exists)."""
    if stop_mode not in {m.value for m in StopMode}:
        return f"bad_stop_mode:{stop_mode!r}"
    return "busy" if skill.busy else "owner_busy" if arb.owner is not None else None


def apply_ops(arb, t: int, ops: list, log, action_dim: int = al.DIM) -> None:
    for op in ops:
        kind, owner = op[0], tuple(op[1])
        if kind == "acquire":
            if not arb.acquire(t, owner):   # the bt_runner mirror predicted success: a protocol fault
                log(Ev.VIOLATION, t, kind="acquire_refused", by=owner[0], owner=arb.owner and arb.owner[0])
        elif kind == "release":
            arb.release(t, owner)
        elif kind == "offer":
            arb.offer(t, owner, np.asarray(op[2], dtype=float).reshape(action_dim))
        else:
            raise ValueError(f"unknown arbiter op {kind!r}")


def report_statuses(skill) -> dict:
    """Goal statuses as the ROS client will see them (a goal refused after ROS acceptance is aborted)."""
    return {g: ("aborted" if s == GoalStatus.REJECTED else s.value) for g, s in skill.statuses.items()}


def cancelable(skill, gid: str) -> bool:
    """rcl only hands ACCEPTED / EXECUTING goals to the cancel callback (assumed); skill_host keeps the ROS goal state
    equal to the controller's at every step boundary, so that is controller status ACTIVE."""
    return skill.statuses.get(gid) == GoalStatus.ACTIVE

