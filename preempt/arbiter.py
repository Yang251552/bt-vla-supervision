"""Single owner of the per-step command; generation fence; criterion-1 bookkeeping.

Owners are ("vla", gen) or ("recovery", None). Each step, sources `offer` a command and
`resolve` applies the current owner's offer, or HOLD when there is none. An offer from a
fenced generation (or from any non-owner) is never applied: it is replaced by HOLD and
counted as a violation, so criterion 1 is both enforced and measured here.
"""
from __future__ import annotations

from . import action_layout as al
from .contracts import Ev

RECOVERY = ("recovery", None)


class Arbiter:
    def __init__(self, log):
        self.log = log
        self.owner = None
        self.fenced: set[int] = set()
        self.violations = 0
        self.last_gripper = al.GRIP_OPEN
        self._offer = None
        self._last_applied: dict[int, int] = {}  # gen -> last step its action was applied

    def acquire(self, t: int, owner) -> bool:
        if self.owner is not None:
            return False
        if owner[0] == "vla" and owner[1] in self.fenced:
            return False
        self.owner = owner
        return True

    def release(self, t: int, owner) -> None:
        if self.owner == owner:
            self.owner = None

    def is_fenced(self, gen: int) -> bool:
        return gen in self.fenced

    def fence(self, t: int, gen: int, reason: str, successor=None) -> None:
        """Cut off generation `gen` from step t on. Idempotent."""
        if gen in self.fenced:
            return
        self.fenced.add(gen)
        self.log(Ev.E, self._last_applied.get(gen, -1), gen=gen)
        self.log(Ev.Q, t, gen=gen, reason=reason, successor=successor and successor[0])
        if self.owner == ("vla", gen):
            self.owner = successor
        if self._offer is not None and self._offer[0] == ("vla", gen):
            self._offer = None

    def offer(self, t: int, owner, action) -> None:
        if owner == self.owner:
            self._offer = (owner, action)
            return
        self.violations += 1
        kind = "after_fence" if owner[0] == "vla" and owner[1] in self.fenced else "not_owner"
        self.log(Ev.VIOLATION, t, kind=kind, by=owner[0], gen=owner[1],
                 owner=self.owner and self.owner[0])

    def resolve(self, t: int):
        """Return (action, owner-or-None) to apply at step t."""
        offer, self._offer = self._offer, None
        if offer is None or offer[0] != self.owner:
            return al.hold(self.last_gripper), None
        owner, action = offer
        self.last_gripper = float(al.get(action, "gripper")[0])
        if owner[0] == "vla":
            gen = owner[1]
            if gen not in self._last_applied:
                self.log(Ev.FIRST_FRESH_ACTION, t, gen=gen)
            self._last_applied[gen] = t
        return action, owner
