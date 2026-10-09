"""scripts/build_results.py on a c6_score-like directory: stub episodes (real event logs from bt.episode), results
lines written by run_episode.py's own head() / summarize(), attempts renamed as run_episode.py does."""
import json
from types import SimpleNamespace

import pytest

from bt.episode import run_episode
from preempt.contracts import Ev, Phase, ProtocolConfig, StopMode
from preempt.eventlog import read
from scripts.build_results import build, wilson
from scripts.run_episode import ROOT, append, head, key_of, load_lines, summarize

CFG = ProtocolConfig()


def episode(d, seed, phase="none", mode="none", rep=0, end="ok"):
    key, res = key_of(seed, phase, mode, rep), d / "results.jsonl"
    att = sum(r["key"] == key and r["outcome"] == "started" for r in load_lines(res))
    for x in (f"{key}.jsonl", f"frames_{key}.tgz"):          # an earlier attempt's files keep its index
        if (d / x).exists():
            (d / x).rename(d / x.replace(key, f"{key}.attempt{att - 1}"))
    a = SimpleNamespace(seed=seed, phase=phase, mode=mode, rep=rep, policy="fake", attempt=att,
                        config=str(ROOT / "config" / "protocol.json"), log=d / f"{key}.jsonl", frame_dir=None)
    append(res, {**head(a, key), "outcome": "started"})
    r = run_episode(a.log, CFG, seed=seed, mode=StopMode.BT_SAFE_ABORT if mode == "none" else StopMode(mode),
                    inject_phase=None if phase == "none" else Phase(phase), max_steps=60 if end == "killed" else None)
    (d / f"frames_{key}.tgz").touch()
    if end != "killed":                                      # killed: "started" line and partial log only
        err = "Traceback\nRuntimeError: sim down" if end == "error" else None
        append(res, summarize(a, key, {"outcome": "error" if err else r.outcome, "steps": r.steps}, err, 1.0, 0))
    return a.log


def edit(log, fn):   # rewrite an event log after its results line was written (provisional fields go stale)
    log.write_text("".join(json.dumps(r) + "\n" for r in map(fn, read(log)) if r))


def no_success(r):
    return None if r["ev"] == Ev.SUCCESS else r


def test_wilson_known_value():
    assert wilson(13, 15) == pytest.approx([0.621180, 0.962639], abs=1e-6)   # score-test inversion, by bisection
    assert wilson(0, 5)[0] == 0.0 and wilson(5, 5)[1] == 1.0 and wilson(0, 0) is None


def test_build_results_attempts_r0_gif(tmp_path):
    d = tmp_path
    for s in (0, 1, 2):
        episode(d, s)
    reps = {r: episode(d, 0, rep=r) for r in (1, 2)}
    episode(d, 0, "transport", "wait_chunk", end="error")
    episode(d, 1, "transport", "truncate_hold", end="killed")
    episode(d, 1, "transport", "language_stop", end="killed")    # killed, never re-run (like s11 in block 1)
    for s in (0, 1):
        for m in StopMode:
            if (s, m) != (1, StopMode.LANGUAGE_STOP):
                episode(d, s, "transport", m.value)
    for m in ("bt_safe_abort", "wait_chunk"):                    # seed 2 qualifies too (o = 8): pick the lowest
        episode(d, 2, "transport", m)
    edit(d / "s0_transport_bt_safe_abort_r0.jsonl", no_success)  # seed 0 fails criterion 3: GIF must take seed 1
    grasp = next(r["t"] for r in read(reps[1]) if r["ev"] == Ev.PHASE and r["phase"] == "grasp")
    lim = grasp + CFG.k_grasp + CFG.hysteresis_steps
    for r, t in ((1, lim), (2, lim - 1)):        # r1 leaves grasp exactly at entry + k + 5, r2 one step sooner
        edit(reps[r], lambda x: {**x, "t": t} if x["ev"] == Ev.PHASE and x["phase"] == "transport" else x)
    edit(reps[2], no_success)

    out = build(d)
    eps, s = out["episodes"], out["summary"]
    assert len(eps) == sum(r["outcome"] == "started" for r in load_lines(d / "results.jsonl")) == 17
    att = {}
    for e in eps:
        att.setdefault(e["key"], []).append((e["attempt"], e["outcome"], e["counted"], e["log"]))
    assert att["s0_transport_wait_chunk_r0"] == [(0, "error", False, "s0_transport_wait_chunk_r0.attempt0.jsonl"),
                                                 (1, "success", True, "s0_transport_wait_chunk_r0.jsonl")]
    assert att["s1_transport_truncate_hold_r0"] == [(0, "killed", False, "s1_transport_truncate_hold_r0.attempt0.jsonl"),
                                                    (1, "success", True, "s1_transport_truncate_hold_r0.jsonl")]
    assert att["s1_transport_language_stop_r0"] == [(0, "killed", False, "s1_transport_language_stop_r0.jsonl")]
    assert s["aborted_attempts"] == [["s1_transport_truncate_hold_r0", 0], ["s1_transport_language_stop_r0", 0]]
    # the stub logs no state hash at t_I: a missing hash never counts as an identical pair
    pe = s["pairing_exact"]
    assert pe["seed_phase_groups"] > 0 and pe["identical_state_hash_at_t_I"] == 0
    assert s["return_base"]["episodes"] > 0 and s["return_base"]["past_pre_abort_pose_m"]["max"] <= 0
    assert s["error_attempts"] == [["s0_transport_wait_chunk_r0", 0]] and s["counted"] == 14
    assert s["paired_seeds"] == [0, 1, 2]
    assert s["floor"]["items"]["pairing"]["paired"] == 3 and s["floor"]["aborted_attempts"] == [
        "s1_transport_language_stop_r0", "s1_transport_truncate_hold_r0"]

    # recomputed from the edited log, the stale results line only reported
    assert s["provisional_diffs"]["s0_transport_bt_safe_abort_r0"]["c3"] == [True, False]
    bt = s["by_mode_phase"]["bt_safe_abort"]["transport"]
    assert (bt["all3"]["k"], bt["all3"]["n"]) == (2, 3) and bt["all3"]["ci95"] == wilson(2, 3)

    assert (s["R0"]["k"], s["R0"]["n"]) == (1, 2)
    g = s["R0_phase_conditional"]["grasp"]
    assert (g["k"], g["n"], g["keys"]) == (1, 1, ["s0_none_none_r1"])
    assert s["R0_phase_conditional"]["transport"]["n"] == 2       # r2 never left transport

    gif = s["gif"]
    assert gif["bt"] == {"key": "s1_transport_bt_safe_abort_r0", "frames": "frames_s1_transport_bt_safe_abort_r0.tgz",
                         "chunk_offset": 14}                     # o = 14 is still mid-chunk
    assert gif["wait_chunk"]["key"] == "s1_transport_wait_chunk_r0" and gif["reason"] is None

    edit(d / "s1_transport_bt_safe_abort_r0.jsonl", no_success)
    gif = build(d)["summary"]["gif"]
    assert gif["bt"]["key"] == "s2_transport_bt_safe_abort_r0" and gif["wait_chunk"]["key"] == "s2_transport_wait_chunk_r0"
    edit(d / "s2_transport_bt_safe_abort_r0.jsonl", no_success)
    gif = build(d)["summary"]["gif"]
    assert gif["bt"] is None and gif["wait_chunk"] is None and "no transport" in gif["reason"]
