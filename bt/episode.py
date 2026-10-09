"""Lockstep episode runner on the stub: one BT tick per control step.

Per step t: state -> phase -> interrupt I/U -> standstill monitor (H) -> goal delivery ->
skill.pre_step (fetch chunk) -> BT tick (cancels processed synchronously) -> skill.act ->
arbiter.resolve -> STEP record -> env.step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from py_trees.common import Status

from preempt.arbiter import Arbiter
from preempt.contracts import Ev, Phase, ProtocolConfig, StopMode
from preempt.eventlog import EventLog
from preempt.phase import PhaseDetector
from preempt.recovery import still
from preempt.skill import SkillController
from preempt.stub import FakePolicy, StubEnv

from .client import FakeSkillClient
from .tree import Ctx, build_tree

INSTRUCTION = "pick the object from the counter and place it on the stove"


@dataclass
class EpisodeResult:
    outcome: str      # success | failed | blocked | timeout
    steps: int
    log_path: str
    ctx: Ctx
    root: object
    server: SkillController


def run_episode(log_path, cfg: ProtocolConfig = ProtocolConfig(), seed: int = 0,
                mode: StopMode = StopMode.BT_SAFE_ABORT, inject_phase: Optional[Phase] = None,
                inject_step: Optional[int] = None, policy=None, accept_latency: int = 0,
                max_steps: Optional[int] = None) -> EpisodeResult:
    log = EventLog(log_path, cfg.hash(), seed=seed, mode=mode,
                   phase=inject_phase and Phase(inject_phase), inject_step=inject_step)
    env = StubEnv(seed)
    policy = policy or FakePolicy(cfg.chunk_len, stop_instruction=cfg.stop_instruction)
    arb = Arbiter(log)
    server = SkillController(cfg, policy, arb, log)
    client = FakeSkillClient(server, log, accept_latency)
    det = PhaseDetector(cfg, log)
    s0 = env.state()
    ctx = Ctx(cfg, client, arb, log, StopMode(mode), INSTRUCTION, s0.eef_pos_base, s0.eef_quat_base)
    root, task = build_tree(ctx)
    t_i, n_still, h_done, succ_done = inject_step, 0, False, False
    max_steps = max_steps or 2 * cfg.reentry_budget_B + cfg.window_W + 2 * cfg.abort_budget_T
    outcome = "timeout"
    for t in range(max_steps):
        s = env.state()
        ctx.t, ctx.state = t, s
        if s.success and not succ_done:
            succ_done = True
            log(Ev.SUCCESS, t)
        ctx.phase = det.update(t, s)
        if t_i is None and inject_phase is not None and Phase(inject_phase) in det.entered:
            t_i = det.entered[Phase(inject_phase)] + cfg.k(inject_phase)
        if t == t_i:
            ctx.interrupt = True
            log(Ev.I, t, phase=ctx.phase)
        if t_i is not None and t == t_i + cfg.window_W:
            ctx.interrupt = False
            log(Ev.U, t)
        if t_i is not None and t >= t_i and not h_done:
            n_still = n_still + 1 if still(s, cfg) else 0
            if n_still >= cfg.standstill_steps:
                h_done = True
                log(Ev.H, t)
        client.pump(t)
        server.pre_step(t, s, obs=env)
        root.tick_once()
        if task.status == Status.SUCCESS:
            outcome = "success"
            break
        if root.status == Status.FAILURE:
            outcome = "blocked" if ctx.abort_failed else "failed"
            break
        server.act(t)
        a, owner = arb.resolve(t)
        log(Ev.STEP, t, src=owner[0] if owner else "hold", gen=owner and owner[1],
            eef=s.eef_pos_world.round(6), base=s.base_pose.round(6), v_eef=round(s.eef_speed, 6),
            v_base=round(s.base_speed, 6), grasped=s.grasped, lift=round(s.obj_lift, 6),
            force=round(s.contact_force, 3), grip=arb.last_gripper)
        env.step(a)
    log(Ev.EPISODE_END, t, outcome=outcome, goals_sent=ctx.goals_sent, late_replies=server.late_replies,
        violations=arb.violations)
    log.close()
    return EpisodeResult(outcome, t, str(log_path), ctx, root, server)
