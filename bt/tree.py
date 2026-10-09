"""py_trees supervisor (lockstep: ticked exactly once per control step by bt.episode).

root = Selector[
  interrupt: Sequence(memory=True)[InterruptGuard, CancelSkill, WaitSkillIdle, SafeAbort, WaitClearance],
  task:      Sequence(memory=True)[ReturnBase, Preconditions, VlaSkill, Postcondition]]
memory=True latches the interrupt branch once the guard fires, so an early clearance cannot
skip SafeAbort. A failed SafeAbort sets ctx.abort_failed: SafeAbort and Preconditions then
fail on every later tick, so the task branch stays blocked and the robot holds.
SafeAbort only moves the robot in BT_SAFE_ABORT mode; for the other modes it is a no-op
and the arbiter holds (no owner).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import py_trees
from py_trees.common import Status

from preempt.arbiter import RECOVERY, Arbiter
from preempt.contracts import BUSY, Ev, GoalStatus, Phase, ProtocolConfig, SimState, StopMode
from preempt.recovery import AbortSequencer, return_base_action, return_base_done, return_base_past

from .client import SkillClient


@dataclass
class Ctx:
    cfg: ProtocolConfig
    client: SkillClient
    arbiter: Arbiter
    log: object
    stop_mode: StopMode
    instruction: str
    home_pos: np.ndarray      # reset eef pose, base frame (retract target)
    home_quat: np.ndarray
    t: int = 0
    state: Optional[SimState] = None
    phase: Phase = Phase.APPROACH
    interrupt: bool = False
    goals_sent: int = 0
    max_goals: int = 2        # initial goal + exactly one re-entry
    abort_failed: bool = False
    return_target: Optional[np.ndarray] = field(default=None)


class _Node(py_trees.behaviour.Behaviour):
    def __init__(self, ctx: Ctx, name: Optional[str] = None):
        super().__init__(name or type(self).__name__)
        self.ctx = ctx


class InterruptGuard(_Node):
    def update(self):
        return Status.SUCCESS if self.ctx.interrupt else Status.FAILURE


class CancelSkill(_Node):
    def update(self):
        if self.ctx.client.status() in BUSY:
            self.ctx.client.cancel(self.ctx.t)
        return Status.SUCCESS


class WaitSkillIdle(_Node):
    """py_trees_ros cancels asynchronously: do not move on until the goal is terminal."""
    def update(self):
        return Status.RUNNING if self.ctx.client.status() in BUSY else Status.SUCCESS


class SafeAbort(_Node):
    def initialise(self):
        c = self.ctx
        self.seq = None
        if c.stop_mode != StopMode.BT_SAFE_ABORT or c.abort_failed:
            return
        # the M4 cancel already handed ownership to recovery; acquire if no goal was ever accepted
        if c.arbiter.owner != RECOVERY and not c.arbiter.acquire(c.t, RECOVERY):
            return
        self.seq = AbortSequencer(c.cfg, c.t, c.phase, c.state, c.arbiter.last_gripper,
                                  c.home_pos, c.home_quat, c.log)

    def update(self):
        c = self.ctx
        if c.stop_mode != StopMode.BT_SAFE_ABORT:
            return Status.SUCCESS
        if c.abort_failed or self.seq is None:
            c.abort_failed = True
            return Status.FAILURE
        action, st = self.seq.step(c.t, c.state)
        if st == "running":
            c.arbiter.offer(c.t, RECOVERY, action)
            return Status.RUNNING
        c.arbiter.release(c.t, RECOVERY)
        if st == "done":
            c.return_target = self.seq.start_base
            return Status.SUCCESS
        c.abort_failed = True
        return Status.FAILURE


class WaitClearance(_Node):
    def update(self):
        return Status.RUNNING if self.ctx.interrupt else Status.SUCCESS


class ReturnBase(_Node):
    """Scripted base motion back to the pre-abort base pose (+-tolerance). Not a teleport.
    Fails (robot holds) as soon as the base is past that pose along the pre-abort heading."""
    def initialise(self):
        self.t0 = self.ctx.t
        self.owns = False
        self.u = 0.0

    def update(self):
        c = self.ctx
        if c.return_target is None:
            return Status.SUCCESS
        if not self.owns:
            self.owns = c.arbiter.acquire(c.t, RECOVERY)
            if not self.owns:
                return Status.FAILURE
        if return_base_past(c.state, c.return_target):
            c.arbiter.release(c.t, RECOVERY)
            c.log(Ev.RETURN_BASE_PAST, c.t, base=c.state.base_pose, target=c.return_target)
            return Status.FAILURE
        if return_base_done(c.state, c.return_target, c.cfg):
            c.arbiter.release(c.t, RECOVERY)
            c.log(Ev.RETURN_BASE_DONE, c.t, base=c.state.base_pose, target=c.return_target)
            c.return_target = None
            return Status.SUCCESS
        if c.t - self.t0 > c.cfg.abort_budget_T:  # reuses T as the bound; own param if needed
            c.arbiter.release(c.t, RECOVERY)
            return Status.FAILURE
        a, self.u = return_base_action(c.state, c.return_target, c.arbiter.last_gripper, self.u)
        c.arbiter.offer(c.t, RECOVERY, a)
        return Status.RUNNING

    def terminate(self, new_status):
        if getattr(self, "owns", False) and new_status == Status.INVALID:
            self.ctx.arbiter.release(self.ctx.t, RECOVERY)


class Preconditions(_Node):
    """Not blocked by a failed abort; gripper empty OR holding the target object (admitted).
    The stub can only hold the target object, so any held object is admitted here."""
    def update(self):
        return Status.FAILURE if self.ctx.abort_failed else Status.SUCCESS


class VlaSkill(_Node):
    def initialise(self):
        c = self.ctx
        self.goal_id = None
        if c.goals_sent >= c.max_goals:
            return
        c.goals_sent += 1
        if c.goals_sent > 1:
            c.log(Ev.REENTRY_GOAL, c.t, n=c.goals_sent)
        self.goal_id = c.client.send_goal(c.t, c.instruction, c.stop_mode)

    def update(self):
        if self.goal_id is None:
            return Status.FAILURE
        st = self.ctx.client.status()
        if st in BUSY:
            return Status.RUNNING
        return Status.SUCCESS if st == GoalStatus.SUCCEEDED else Status.FAILURE

    def terminate(self, new_status):
        # like py_trees_ros: a preempted action client cancels its goal (server treats repeats as no-ops)
        if new_status == Status.INVALID and getattr(self, "goal_id", None) \
                and self.ctx.client.status() in BUSY:
            self.ctx.client.cancel(self.ctx.t)


class Postcondition(_Node):
    def update(self):
        return Status.SUCCESS if self.ctx.state.success else Status.FAILURE


def build_tree(ctx: Ctx):
    """-> (root, task_branch)."""
    interrupt = py_trees.composites.Sequence("interrupt", memory=True, children=[
        InterruptGuard(ctx), CancelSkill(ctx), WaitSkillIdle(ctx), SafeAbort(ctx), WaitClearance(ctx)])
    task = py_trees.composites.Sequence("task", memory=True, children=[
        ReturnBase(ctx), Preconditions(ctx), VlaSkill(ctx), Postcondition(ctx)])
    root = py_trees.composites.Selector("root", memory=False, children=[interrupt, task])
    return root, task
