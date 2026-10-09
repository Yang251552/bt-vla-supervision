"""check_floor and metrics on malformed or adversarial-shaped inputs (independent review F01, F05, F10-F17, F20)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from preempt.contracts import Ev, ProtocolConfig
from preempt.metrics import episode_metrics
from scripts.check_floor import MODES, PHASES, check

ROOT = Path(__file__).resolve().parents[1]
CFG = ProtocolConfig()


def line(seed=0, phase="none", mode="none", outcome="success", **kw):
    return {"key": f"s{seed}_{phase}_{mode}_r0", "seed": seed, "phase": phase, "mode": mode, "rep": 0,
            "outcome": outcome, "paired": phase == "none", "n_goal_accepted": 1, "n_feedback": 1, "violations": 0,
            "cfg_hash": "h", "manifest_sha256": "m", "obj_lang": f"obj{seed % 2}", "layout_id": seed % 2, **kw}


def floor_lines():
    return [line(s) for s in range(3)] + [line(s, p, m, t_I=40, A=40, B=100, c1=True, c2=True, c3=True)
                                          for s in range(3) for p in PHASES for m in MODES]


def test_fixture_passes():
    assert check(floor_lines(), "h", "m")["pass"]


def test_correctness_scans_every_attempt():   # F01, F12
    ok = line(0, "grasp", "bt_safe_abort", t_I=40, A=40, B=100, reentry_t=241)
    for bad in ({"violations": 1}, {"outcome": "timeout"}, {"B": None}, {"B": None, "outcome": "error"},
                {"B": None, "void": True, "void_reason": "phase_mismatch"}):
        for lines in ([{**ok, **bad}], [{**ok, **bad}, ok]):
            rep = check(lines)
            assert not rep["items"]["correctness"]["pass"] and rep["episodes"] == 1, (bad, lines)
    assert check([ok])["items"]["correctness"]["pass"]


def test_outcome_must_be_known():   # F05
    for o in (None, "bogus"):
        rep = check([{**r, "outcome": o} for r in floor_lines()], "h", "m")
        assert not rep["pass"] and not rep["items"]["schema"]["pass"]
    lines = floor_lines()
    del lines[0]["outcome"]
    assert check(lines)["items"]["schema"]["invalid_outcome"] == [lines[0]["key"]]


def test_counts_and_flags_are_typed():   # F13
    for n in (-1, 0, "1", True, 1.0):
        rep = check([{**r, "n_goal_accepted": n} for r in floor_lines()], "h", "m")
        assert not rep["pass"] and not rep["items"]["semantics"]["pass"], n
    rep = check([{**r, "c1": "false", "c2": "false", "c3": "false"} if r["phase"] != "none" else r
                 for r in floor_lines()], "h", "m")
    assert not rep["pass"] and rep["items"]["safe_abort"]["count"] == 0 and not rep["items"]["schema"]["pass"]
    rep = check([{**r, "paired": 1} if r["phase"] == "none" else r for r in floor_lines()], "h", "m")
    assert rep["items"]["pairing"]["paired"] == 0 and rep["items"]["schema"]["invalid_flags"] == [
        "s0_none_none_r0", "s1_none_none_r0", "s2_none_none_r0"]


def test_unknown_object_is_not_a_second_object():   # F14
    lines = [{**r, "obj_lang": "one", "layout_id": 0} for r in floor_lines()]
    next(r for r in lines if r["mode"] == "bt_safe_abort")["obj_lang"] = None
    del next(r for r in lines if r["mode"] == "bt_safe_abort")["layout_id"]
    rep = check(lines)["items"]["safe_abort"]
    assert not rep["pass"] and rep["objects"] == ["one"] and rep["layouts"] == ["0"]


def test_rerun_flagged_at_new_start():   # F10
    done, start = line(), line(outcome="started")
    for tail in ([], [line(outcome="error")]):
        assert check([done, start, *tail])["rerun_after_finish"] == [done["key"]]
    assert check([start, line(outcome="error"), start, done])["rerun_after_finish"] == []


def test_cli_refuses_missing_or_broken_frozen_files(tmp_path):   # F11
    res = tmp_path / "results.jsonl"
    man = b'{"a": "b"}\n'
    res.write_text("".join(json.dumps({**r, "manifest_sha256": hashlib.sha256(man).hexdigest()}) + "\n"
                           for r in floor_lines()))
    pre = tmp_path / "prereg.json"

    def run():
        return subprocess.run([sys.executable, str(ROOT / "scripts/check_floor.py"), str(res), "--prereg", str(pre)],
                              capture_output=True, text=True).returncode

    assert run() != 0                                    # prereg missing
    pre.write_text("{")
    assert run() != 0                                    # prereg unparseable
    pre.write_text(json.dumps({"protocol": {"cfg_hash": None}}))
    (tmp_path / "frozen_manifest.json").write_bytes(man)
    assert run() != 0                                    # no frozen cfg_hash
    pre.write_text(json.dumps({"protocol": {"cfg_hash": "h"}}))
    assert run() == 0
    (tmp_path / "frozen_manifest.json").write_text("{")
    assert run() != 0                                    # manifest unparseable
    (tmp_path / "frozen_manifest.json").unlink()
    assert run() != 0                                    # manifest missing


def events(U=240, H=45, B=None, success=None, end=245):
    recs = [{"ev": Ev.EPISODE, "t": -1, "mode": "bt_safe_abort", "phase": "approach"},
            {"ev": Ev.PHASE, "t": 0, "phase": "approach"}, {"ev": Ev.GOAL_ACCEPTED, "t": 0, "gen": 1},
            {"ev": Ev.I, "t": 40, "phase": "approach"}, {"ev": Ev.R, "t": 40, "gen": 1},
            {"ev": Ev.A, "t": 40, "gen": 1}, {"ev": Ev.Q, "t": 40, "gen": 1}, {"ev": Ev.CANCELED, "t": 40, "gen": 1}]
    recs += [{"ev": ev, "t": t} for ev, t in ((Ev.H, H), (Ev.U, U), (Ev.SUCCESS, success)) if t is not None]
    if B is not None:
        recs += [{"ev": Ev.RETRACTED, "t": B - 1}, {"ev": Ev.B, "t": B, "backoff_m": 0.2}]
    recs += [{"ev": Ev.STEP, "t": t, "src": "hold", "gen": None, "eef": [0.001 * t, 0, 0], "grasped": False,
              "lift": 0, "force": 0} for t in range(end + 1)]
    return sorted(recs, key=lambda r: r["t"])


def test_re_motion_needs_an_observed_rest_inside_the_window():   # F15 (t_I = 40, U = 240)
    assert episode_metrics(events(B=240, H=None), CFG)["re_motion"] is False
    assert episode_metrics(events(B=241, H=None), CFG)["re_motion"] is None
    assert episode_metrics(events(U=None, B=100, H=None, end=99), CFG)["re_motion"] is None   # B on the last tick
    recs = [r for r in events(B=100, H=None) if not (r["ev"] == Ev.STEP and r["t"] == 100)]
    with pytest.raises(ValueError):          # C10-10: a gap in the STEP records is a malformed log
        episode_metrics(recs, CFG)


@pytest.mark.parametrize("success, want", [(239, True), (240, True), (241, False)])
def test_completion_window_is_closed(success, want):   # F16
    assert episode_metrics(events(success=success), CFG)["completed_in_window"] is want


def test_duplicate_steps_cannot_hide_a_fence_violation():   # F17: a duplicate STEP time counts as a violation
    recs = events()
    next(r for r in recs if r["ev"] == Ev.STEP and r["t"] == 41).update(src="vla", gen=1)
    assert episode_metrics(recs, CFG)["violations"] == 1
    recs = sorted(recs + [{**recs[-1], "t": 41, "src": "hold", "gen": None}], key=lambda r: r["t"])
    m = episode_metrics(recs, CFG)
    assert m["violations"] == 2 and not m["c1"]
    recs = events()
    m = episode_metrics(recs + [dict(recs[-1])], CFG)
    assert m["violations"] == 1 and not m["c1"]


def test_window_never_extends_past_t_I_plus_W():   # F20
    m = episode_metrics(events(U=None, H=245, end=250), CFG)
    assert not m["still_in_W"] and m["L_S"] is None and m["d_post_cm"] == pytest.approx(20.0)
    m = episode_metrics(events(U=None, H=100, end=200), CFG)    # blocked before t_I + W: end of log
    assert m["still_in_W"] and m["L_S"] == 60 and m["d_post_cm"] == pytest.approx(6.0)
    m = episode_metrics(events(U=None, H=None, end=200), CFG)
    assert m["d_post_cm"] == pytest.approx(16.0)
