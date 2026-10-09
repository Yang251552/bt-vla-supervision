"""The skill_host <-> bt_runner lockstep protocol (ros/lockstep.py) replayed in one process on the stub.

Same order as the two ROS nodes, no transport: state report -> barrier (client view == report) -> tick
behind RemoteArbiter -> skill_host logs the BT events and applies the arbiter ops -> only then the tick's
goal / cancel requests reach the controller (as skill_host's callbacks do) -> act -> resolve -> step.
It must reproduce what bt/episode.py gets with the in-process client, with zero violations.
"""
import pytest
from py_trees.common import Status

from bt.episode import INSTRUCTION, run_episode
from bt.tree import Ctx, build_tree
from preempt.arbiter import Arbiter
from preempt.contracts import Ev, GoalStatus, Phase, ProtocolConfig, StopMode
from preempt.eventlog import EventLog, read
from preempt.metrics import episode_metrics
from preempt.phase import PhaseDetector
from preempt.recovery import still
from preempt.skill import SkillController
from preempt.stub import FakePolicy, StubEnv
from ros.lockstep import (DeferredSkillClient, Forward, RemoteArbiter, apply_ops, cancelable,
                          goal_reject_reason, report_statuses)

CFG = ProtocolConfig()
TERMINAL = ("succeeded", "canceled", "aborted")


class Host:
    """skill_host's side: controller + arbiter, goal / cancel handling as in its action-server callbacks."""

    def __init__(self, log, cfg):
        self.log, self.t = log, 0
        self.arb = Arbiter(log)
        self.skill = SkillController(cfg, FakePolicy(cfg.chunk_len, stop_instruction=cfg.stop_instruction),
                                     self.arb, log)

    def on_goal(self, gid, instruction, mode, client):
        g = client.goals[gid]
        reason = goal_reject_reason(self.skill, self.arb, mode)
        if reason:
            self.log(Ev.GOAL_REJECTED, self.t, goal_id=None, reason=reason)
            g["accepted"] = False
            return
        g["accepted"], g["handle"] = True, gid
        self.skill.submit_goal(self.t, gid, instruction, StopMode(mode))
        if g["cancel_on_accept"]:
            self.on_cancel(gid, client)

    def on_cancel(self, gid, client):
        g = client.goals[gid]
        if g["handle"] is None:
            g["cancel_on_accept"] = g["accepted"] is None
            return
        if cancelable(self.skill, gid):
            self.skill.cancel(self.t, gid)


class EmuClient(DeferredSkillClient):
    def __init__(self, log, host):
        super().__init__(log)
        self.host = host

    def flush(self):
        if self.goal_out is not None:
            gid, instruction, mode = self.goal_out
            self.goal_out = None
            self.host.on_goal(gid, instruction, mode, self)
        for gid in self.cancel_out:
            self.host.on_cancel(gid, self)
        self.cancel_out = []


def run_lockstep(path, seed, mode, phase, cfg=CFG):
    log = EventLog(path, cfg.hash(), seed=seed, mode=mode, phase=phase)
    env, host = StubEnv(seed), Host(log, cfg)
    det, fwd, rarb = PhaseDetector(cfg, log), Forward(), RemoteArbiter()
    client = EmuClient(fwd, host)
    s0 = env.state()
    ctx = Ctx(cfg, client, rarb, fwd, StopMode(mode), INSTRUCTION, s0.eef_pos_base, s0.eef_quat_base)
    root, task = build_tree(ctx)
    t_i, n_still, h_done, succ_done, interrupt, outcome = None, 0, False, False, False, "timeout"
    for t in range(2 * cfg.reentry_budget_B + cfg.window_W + 2 * cfg.abort_budget_T):
        host.t, s = t, env.state()
        if s.success and not succ_done:
            succ_done = True
            log(Ev.SUCCESS, t)
        ph = det.update(t, s)
        if t_i is None and phase in det.entered:
            t_i = det.entered[phase] + cfg.k(phase)
        if t == t_i:
            interrupt = True
            log(Ev.I, t, phase=ph)
        if t_i is not None and t == t_i + cfg.window_W:
            interrupt = False
            log(Ev.U, t)
        if t_i is not None and t >= t_i and not h_done:
            n_still = n_still + 1 if still(s, cfg) else 0
            if n_still >= cfg.standstill_steps:
                h_done = True
                log(Ev.H, t)
        host.skill.pre_step(t, s, obs=env)
        report = report_statuses(host.skill)
        for gid, st in report.items():           # result futures land before the barrier lets the tick run
            if gid in client.goals and st in TERMINAL:
                client.goals[gid]["result"] = GoalStatus(st)
        assert client.caught_up(report), (t, report, {g: client.status_of(g) for g in client.goals})
        ctx.t, ctx.state, ctx.phase, ctx.interrupt = t, s, ph, interrupt
        rarb.sync(host.arb.owner, host.arb.last_gripper)
        fwd.events = []
        root.tick_once()
        for e in fwd.events:                      # skill_host applies the ack ...
            e = dict(e)
            log(e.pop("ev"), e.pop("t"), **e)
        apply_ops(host.arb, t, rarb.ops, log)
        client.flush()                            # ... and only then sees the tick's goal / cancel requests
        if task.status == Status.SUCCESS:
            outcome = "success"
            break
        if root.status == Status.FAILURE:
            outcome = "blocked" if ctx.abort_failed else "failed"
            break
        host.skill.act(t)
        a, owner = host.arb.resolve(t)
        log(Ev.STEP, t, src=owner[0] if owner else "hold", gen=owner and owner[1], eef=s.eef_pos_world,
            base=s.base_pose, grasped=s.grasped, lift=s.obj_lift, force=s.contact_force)
        env.step(a)
    log.close()
    return outcome, episode_metrics(read(path), cfg)


@pytest.mark.parametrize("phase", [Phase.APPROACH, Phase.GRASP, Phase.TRANSPORT])
@pytest.mark.parametrize("mode", list(StopMode))
def test_lockstep_protocol_reproduces_the_stub(tmp_path, mode, phase):
    outcome, m = run_lockstep(tmp_path / "ls.jsonl", seed=1, mode=mode, phase=phase)
    ref = run_episode(tmp_path / "ref.jsonl", CFG, seed=1, mode=mode, inject_phase=phase)
    r = episode_metrics(read(ref.log_path), CFG)
    assert outcome == ref.outcome == "success"
    assert m["violations"] == 0 and not m["void"]
    assert (m["c1"], m["c3"], m["N_post"]) == (r["c1"], r["c3"], r["N_post"])
    assert m["c2"] == r["c2"]
    assert m["Q"] - m["R"] <= CFG.fence_slack_steps or mode in (StopMode.WAIT_CHUNK, StopMode.LANGUAGE_STOP)
