"""Phase detector: monotonic latch, each transition needs its condition for N consecutive steps.

approach -> grasp      : eef-object distance < grasp_dist_m
grasp -> transport     : secured = grasped AND object lifted >= secured_lift_m
transport -> done      : not grasped, or task success
The phase never goes back. The entry step is the step on which the N-th consecutive
satisfying observation arrives (t_I = entry + k).
"""
from __future__ import annotations

from .contracts import Ev, Phase, ProtocolConfig, SimState

ORDER = [Phase.APPROACH, Phase.GRASP, Phase.TRANSPORT, Phase.DONE]


class PhaseDetector:
    def __init__(self, cfg: ProtocolConfig, log=None):
        self.cfg, self.log = cfg, log
        self.phase = Phase.APPROACH
        self.entered = {Phase.APPROACH: 0}
        self._count = 0

    def _next_condition(self, s: SimState) -> bool:
        if self.phase == Phase.APPROACH:
            return s.eef_obj_dist < self.cfg.grasp_dist_m
        if self.phase == Phase.GRASP:
            return self.cfg.secured(s)
        if self.phase == Phase.TRANSPORT:
            return (not s.grasped) or s.success
        return False

    def update(self, t: int, s: SimState) -> Phase:
        if self._next_condition(s):
            self._count += 1
            if self._count >= self.cfg.hysteresis_steps:
                self.phase = ORDER[ORDER.index(self.phase) + 1]
                self.entered[self.phase] = t
                self._count = 0
                if self.log:
                    self.log(Ev.PHASE, t, phase=self.phase)
        else:
            self._count = 0
        return self.phase
