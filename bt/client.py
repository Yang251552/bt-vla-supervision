"""SkillClient: what the BT sees of the VLA action server. The ROS-backed client comes later."""
from __future__ import annotations

from typing import Optional, Protocol

from preempt.contracts import Ev, GoalStatus, StopMode
from preempt.skill import SkillController


class SkillClient(Protocol):
    def send_goal(self, t: int, instruction: str, stop_mode: StopMode) -> str: ...
    def cancel(self, t: int) -> None: ...
    def status(self) -> Optional[GoalStatus]: ...
    def feedback(self) -> Optional[dict]: ...


class FakeSkillClient:
    """In-process client. Goal requests reach the server after `accept_latency` steps
    (delivered by `pump`); cancels go out immediately, so a cancel can overtake its goal
    exactly as with ROS 2's separate goal and cancel services."""

    def __init__(self, server: SkillController, log, accept_latency: int = 0):
        self.server, self.log, self.accept_latency = server, log, accept_latency
        self.goal_id: Optional[str] = None
        self._n = 0
        self._inflight: list[tuple[int, str, str, StopMode]] = []
        self._cancel_sent: set[str] = set()

    def send_goal(self, t: int, instruction: str, stop_mode: StopMode) -> str:
        self._n += 1
        self.goal_id = f"goal-{self._n}"
        self._inflight.append((t + self.accept_latency, self.goal_id, instruction, stop_mode))
        self.pump(t)
        return self.goal_id

    def pump(self, t: int) -> None:
        due = [g for g in self._inflight if g[0] <= t]
        self._inflight = [g for g in self._inflight if g[0] > t]
        for _, gid, instr, mode in due:
            self.server.submit_goal(t, gid, instr, mode)

    def cancel(self, t: int) -> None:
        if self.goal_id is None:
            return
        self.log(Ev.R, t, goal_id=self.goal_id, repeat=self.goal_id in self._cancel_sent)
        self._cancel_sent.add(self.goal_id)
        self.server.cancel(t, self.goal_id)

    def status(self) -> Optional[GoalStatus]:
        if self.goal_id is None:
            return None
        st = self.server.status_of(self.goal_id)
        if st is None:  # still in flight
            return GoalStatus.CANCELING if self.goal_id in self._cancel_sent else GoalStatus.PENDING
        return st

    def feedback(self) -> Optional[dict]:
        g = self.server.goal
        return self.server.feedback if g is not None and g.goal_id == self.goal_id else None
