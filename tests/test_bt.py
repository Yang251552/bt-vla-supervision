"""BT on the stub, end to end; criteria 1-3 are computed from the JSONL event log."""
import dataclasses

import numpy as np
import pytest
from py_trees.common import Status

from bt.episode import run_episode
from preempt import action_layout as al
from preempt.contracts import Ev, Phase, ProtocolConfig, StopMode
from preempt.eventlog import read
from preempt.metrics import episode_metrics
from preempt.skill import ChunkReply
from preempt.stub import FakePolicy

CFG = ProtocolConfig()


def run(tmp_path, cfg=CFG, **kw):
    res = run_episode(tmp_path / "ep.jsonl", cfg, **kw)
    recs = read(res.log_path)
    return res, recs, episode_metrics(recs, cfg)


def t_of(recs, ev):
    return [r["t"] for r in recs if r["ev"] == ev]


def test_uninterrupted_baseline_succeeds_and_logs_config_hash(tmp_path):
    res, recs, m = run(tmp_path, seed=2)
    assert res.outcome == "success" and m["completed"] and m["violations"] == 0
    assert recs[0]["ev"] == Ev.EPISODE and recs[0]["cfg_hash"] == CFG.hash()
    assert not t_of(recs, Ev.I)
    assert t_of(recs, Ev.GOAL_ACCEPTED) and t_of(recs, Ev.FEEDBACK)


@pytest.mark.parametrize("phase", [Phase.APPROACH, Phase.GRASP, Phase.TRANSPORT])
def test_interrupt_in_each_phase_safe_aborts_returns_and_reenters(tmp_path, phase):
    res, recs, m = run(tmp_path, seed=1, mode=StopMode.BT_SAFE_ABORT, inject_phase=phase)
    assert res.outcome == "success"
    assert m["c1"] and m["c2"] and m["c3"], m
    assert m["violations"] == 0 and not m["void"]
    assert [r["phase"] for r in recs if r["ev"] == Ev.I] == [phase.value]
    assert m["Q"] - m["R"] <= CFG.fence_slack_steps and m["N_post"] == 0
    # order: abort completes, interrupt clears, base returns, then exactly one re-entry goal
    (b,), (u,), (rb,), (re,) = (t_of(recs, e) for e in (Ev.B, Ev.U, Ev.RETURN_BASE_DONE, Ev.REENTRY_GOAL))
    assert m["t_I"] < b < u < rb <= re
    assert len(t_of(recs, Ev.GOAL_ACCEPTED)) == 2
    start = next(r for r in recs if r["ev"] == Ev.ABORT_START)
    if phase == Phase.TRANSPORT:
        assert m["secured_at_I"] and m["obj_held"] and start["gripper"] == al.GRIP_CLOSE
    if phase == Phase.GRASP:
        assert start["gripper"] == al.GRIP_OPEN and start["lift_m"] == CFG.lift_grasp_m


def test_four_stop_modes_post_interrupt_vla_actions(tmp_path):
    out = {}
    for mode in StopMode:
        _, _, out[mode] = run(tmp_path / mode.value, seed=0, mode=mode, inject_phase=Phase.TRANSPORT)
    o = out[StopMode.WAIT_CHUNK]["chunk_offset"]
    assert out[StopMode.WAIT_CHUNK]["N_post"] == 16 - o
    assert out[StopMode.TRUNCATE_HOLD]["N_post"] == out[StopMode.BT_SAFE_ABORT]["N_post"] == 0
    assert out[StopMode.LANGUAGE_STOP]["N_post"] == CFG.window_W     # VLA stays in charge within W
    assert out[StopMode.TRUNCATE_HOLD]["c1"] and out[StopMode.BT_SAFE_ABORT]["c1"]
    assert not out[StopMode.LANGUAGE_STOP]["c1"]
    assert all(m["violations"] == 0 and m["completed"] for m in out.values())
    assert out[StopMode.BT_SAFE_ABORT]["c2"] and not out[StopMode.TRUNCATE_HOLD]["c2"]  # BT contract only
    assert all(m["still_in_W"] and m["re_motion"] is False and m["obj_held_W"] for m in out.values())


def test_language_stop_ignored_is_censored_at_W(tmp_path):
    """A policy that ignores "stop" never stands still inside W: L_S censored, d_post ends at U."""
    _, recs, m = run(tmp_path, seed=0, mode=StopMode.LANGUAGE_STOP, inject_phase=Phase.APPROACH,
                     policy=FakePolicy(CFG.chunk_len, obey_stop=False))
    assert not m["still_in_W"] and m["L_S"] is None and m["re_motion"] is None
    steps = {r["t"]: r for r in recs if r["ev"] == Ev.STEP}
    path = [np.asarray(steps[t]["eef"]) for t in range(m["t_I"], m["U"] + 1)]
    assert m["d_post_cm"] == pytest.approx(100 * sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:])))


def test_early_interrupt_clearance_cannot_skip_safe_abort(tmp_path):
    cfg = dataclasses.replace(CFG, window_W=3)        # clears while SafeAbort is still running
    res, recs, m = run(tmp_path, cfg, seed=1, mode=StopMode.BT_SAFE_ABORT, inject_phase=Phase.TRANSPORT)
    (u,), (b,), (re,) = t_of(recs, Ev.U), t_of(recs, Ev.B), t_of(recs, Ev.REENTRY_GOAL)
    assert u < b < re and t_of(recs, Ev.RETRACTED)[0] < b
    assert res.outcome == "success" and m["c1"] and m["c2"] and m["violations"] == 0


def test_recovery_failure_blocks_the_task_branch(tmp_path):
    cfg = dataclasses.replace(CFG, abort_budget_T=5)  # abort cannot finish in 5 steps
    res, recs, m = run(tmp_path, cfg, seed=1, mode=StopMode.BT_SAFE_ABORT, inject_phase=Phase.GRASP)
    assert res.outcome == "blocked" and t_of(recs, Ev.ABORT_FAILED)
    assert not t_of(recs, Ev.REENTRY_GOAL) and res.ctx.goals_sent == 1
    assert res.ctx.return_target is None and res.ctx.arbiter.owner is None   # robot holds
    res.ctx.interrupt = False                         # even after clearance the task stays blocked
    for t in range(res.steps, res.steps + 10):
        res.ctx.t = t
        res.root.tick_once()
        assert res.root.status == Status.FAILURE
    assert res.ctx.goals_sent == 1 and m["violations"] == 0


def test_cancel_before_goal_acceptance_in_the_tree(tmp_path):
    res, recs, m = run(tmp_path, seed=1, mode=StopMode.BT_SAFE_ABORT, inject_step=1, accept_latency=2)
    rej = [r for r in recs if r["ev"] == Ev.GOAL_REJECTED]
    assert rej and rej[0]["reason"] == "cancelled_before_accept"
    assert all(t > t_of(recs, Ev.U)[0] for t in t_of(recs, Ev.GOAL_ACCEPTED))
    assert not [r for r in recs if r["ev"] == Ev.STEP and r["src"] == "vla" and r["t"] < t_of(recs, Ev.U)[0]]
    assert t_of(recs, Ev.B) and res.outcome == "success" and m["violations"] == 0


class StallsAfterReentry(FakePolicy):
    """Re-entry goal (generation >= 2) only gets hold chunks, so it runs out of budget B."""
    def request(self, req):
        if req.gen < 2:
            return super().request(req)
        self._pending.append([0, ChunkReply(req.gen, req.query_idx, np.tile(al.hold(al.GRIP_OPEN), (16, 1)))])


def test_only_one_reentry_opportunity(tmp_path):
    res, recs, m = run(tmp_path, seed=1, mode=StopMode.BT_SAFE_ABORT, inject_phase=Phase.APPROACH,
                       policy=StallsAfterReentry())
    assert res.outcome == "failed" and t_of(recs, Ev.GOAL_ABORTED)
    assert len(t_of(recs, Ev.GOAL_ACCEPTED)) == 2 and res.ctx.goals_sent == 2
    for t in range(res.steps, res.steps + 5):        # ticking on must not send a third goal
        res.ctx.t = t
        res.root.tick_once()
        assert res.root.status == Status.FAILURE
    assert res.ctx.goals_sent == 2 and m["c1"] and m["c2"] and not m["c3"]


# stub seed 1 uninterrupted: grasp entered at 66, transport at 106, success at 262
@pytest.mark.parametrize("k_grasp, max_steps, reason", [
    (60, None, "phase_mismatch"),         # t_I = 126 lands in transport, not grasp (dev seed 1002 case)
    (38, None, "phase_ends_too_soon"),    # t_I = 104, transport latches at 106 (< 5 steps)
    (10, 70, "no_injection"),             # episode cut before t_I = 76: no interrupt sample
])
def test_void_when_interrupt_is_not_a_sample_of_the_requested_phase(tmp_path, k_grasp, max_steps, reason):
    cfg = dataclasses.replace(CFG, k_grasp=k_grasp)
    _, recs, m = run(tmp_path, cfg, seed=1, mode=StopMode.BT_SAFE_ABORT, inject_phase=Phase.GRASP,
                     max_steps=max_steps)
    assert m["void"] and m["void_reason"] == reason, m
    if reason == "phase_mismatch":
        assert m["phase_at_I"] == Phase.TRANSPORT.value


def test_check_floor_voids_all_modes_of_a_seed_phase():
    from scripts.check_floor import MODES, check
    lines = [{"key": "s0_none_none_r0", "seed": 0, "phase": "none", "mode": "none", "rep": 0, "paired": True,
              "outcome": "success", "completed": True, "n_goal_accepted": 1, "n_feedback": 1}]
    for m in MODES:
        lines.append({"key": f"s0_grasp_{m}_r0", "seed": 0, "phase": "grasp", "mode": m, "rep": 0, "t_I": 1, "A": 1,
                      "outcome": "success", "n_goal_accepted": 1, "n_feedback": 1, "c1": True, "c2": True, "c3": True,
                      "void": m == "wait_chunk", "void_reason": "phase_mismatch" if m == "wait_chunk" else None})
    lines.append({"key": "s0_transport_bt_safe_abort_r0", "seed": 0, "phase": "transport", "mode": "bt_safe_abort",
                  "rep": 0, "outcome": "failed", "n_goal_accepted": 1, "n_feedback": 1, "void": True,
                  "void_reason": "no_injection", "t_I": 400})   # planned t_I as skill_host logs it, never reached
    rep = check(lines)
    assert rep["items"]["pairing"]["void_seed_phase"] == [[0, "grasp", "phase_mismatch"],
                                                          [0, "transport", "no_injection"]]
    assert rep["items"]["semantics"]["pass"]           # no interrupt injected -> no cancel accept expected
    assert rep["items"]["latency"]["seeds_with_all_modes"]["grasp"] == []
    assert rep["items"]["safe_abort"]["count"] == 0


def test_check_floor_attempts_and_correctness_stops():
    from scripts.check_floor import check
    base = {"seed": 0, "rep": 0, "n_goal_accepted": 1, "n_feedback": 1, "cfg_hash": "h", "manifest_sha256": "m"}
    killed, skip, stuck, merr = (f"s0_grasp_{m}_r0" for m in ("wait_chunk", "bt_safe_abort", "truncate_hold",
                                                               "language_stop"))
    lines = [{**base, "key": "s0_none_none_r0", "phase": "none", "mode": "none", "outcome": "started"},
             {**base, "key": "s0_none_none_r0", "phase": "none", "mode": "none", "outcome": "success", "paired": True},
             {**base, "key": killed, "phase": "grasp", "mode": "wait_chunk", "outcome": "started"},
             {**base, "key": skip, "phase": "grasp", "mode": "bt_safe_abort", "outcome": "success",
              "t_I": 5, "A": 5, "reentry_t": 300, "B": None},
             {**base, "key": stuck, "phase": "grasp", "mode": "truncate_hold", "outcome": "timeout", "t_I": 5, "A": 5},
             {**base, "key": merr, "phase": "grasp", "mode": "language_stop", "outcome": "success",
              "metrics_error": "Traceback", "cfg_hash": None}]
    rep = check(lines, frozen_cfg="h", frozen_manifest="m")
    assert rep["aborted_attempts"] == [killed] and set(rep["not_finished"]) == {killed, merr}
    assert rep["items"]["correctness"]["reentry_without_abort"] == [skip]
    assert rep["items"]["correctness"]["timeouts"] == [stuck] and not rep["items"]["correctness"]["pass"]
    assert rep["items"]["one_config"]["pass"]           # error / killed / metrics_error lines do not count
    assert not check(lines, frozen_cfg="h", frozen_manifest="other")["items"]["one_config"]["pass"]


@pytest.mark.parametrize("lag", [0.5, 0.2, 0.1])
@pytest.mark.parametrize("plant_m_per_unit", [0.0041, 0.037, 0.08])
def test_return_base_never_drives_past_the_pre_abort_pose(tmp_path, monkeypatch, plant_m_per_unit, lag):
    """Real base: ~0.037 m/step at |a| = 1 (dev-episode logs), 9x the ramp-up probe the controller was tuned on.
    The old loop drove at full speed and overshot toward the counter (seeds 1002/1005, 300-500 N)."""
    import preempt.stub as stub
    monkeypatch.setattr(stub, "BASE_M_PER_UNIT", plant_m_per_unit)
    monkeypatch.setattr(stub, "LAG", lag)
    for phase in (Phase.APPROACH, Phase.TRANSPORT):
        _, recs, m = run(tmp_path / phase.value, seed=1, mode=StopMode.BT_SAFE_ABORT, inject_phase=phase)
        done = next(r for r in recs if r["ev"] == Ev.RETURN_BASE_DONE)
        tgt = np.asarray(done["target"])
        heading = np.array([np.cos(tgt[2]), np.sin(tgt[2])])
        past = max((np.asarray(r["base"][:2]) - tgt[:2]) @ heading
                   for r in recs if r["ev"] == Ev.STEP and m["U"] <= r["t"] <= done["t"] + 20)
        assert past <= 0.0, (phase, past)
        assert m["c1"] and m["c2"] and m["c3"], m


def test_check_floor_flags_a_finished_key_run_again():
    from scripts.check_floor import check
    line = {"key": "k", "seed": 0, "phase": "none", "mode": "none", "rep": 0, "outcome": "success", "paired": True,
            "n_goal_accepted": 1, "n_feedback": 1}
    assert check([line, {**line, "outcome": "error"}, line])["rerun_after_finish"] == ["k"]
    assert check([{**line, "outcome": "error"}, line])["rerun_after_finish"] == []
