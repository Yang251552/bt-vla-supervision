"""Second independent execution review, findings C10-01..03 and C10-05..15 (C10-04 is a documented limit)."""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from preempt.contracts import Ev, ProtocolConfig
from preempt.metrics import episode_metrics
from scripts import build_results as br
from scripts import frozen
from scripts.check_floor import check
from scripts.run_episode import load_lines
from test_build_results import edit, episode as stub_episode, no_success
from test_exec_hardening import SCORE, bash, episode, rows, screen
from test_floor_hardening import events, floor_lines

ROOT = Path(__file__).resolve().parents[1]
CFG = ProtocolConfig()
STATE = SCORE[SCORE.index("state() {"):SCORE.index("finished() {")]
READ_STATE = SCORE[SCORE.index("state() {"):SCORE.index("fails=0")]


def state(tmp_path, lines, seed=0, fn=STATE, call="state"):
    rows(tmp_path / "results.jsonl", lines)
    return bash(f'python3() {{ "{sys.executable}" "$@"; }}\nOUT="{tmp_path}"\n{fn}\n{call} {seed}')


def cli(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)


def build_cli(d, out):
    """build_results.py on d -> (exit code, stderr, the written results' "invalid" reasons)."""
    r = cli("scripts/build_results.py", d, "--out", out)
    return r.returncode, r.stderr, json.loads(out.read_text()).get("invalid")


def results(d, fn):
    rows(d / "results.jsonl", fn(load_lines(d / "results.jsonl")))


@pytest.mark.parametrize("bad", [{"phase": "grasp", "mode": "wait_chunk", "violations": 1}, {"outcome": "timeout"},
                                 {"phase": "grasp", "mode": "bt_safe_abort", "reentry_t": 300, "B": None}])
def test_c6_correctness_stop_survives_a_later_clean_line(tmp_path, bad):   # C10-01
    bad = {**screen(), **bad}
    bad["key"] = f"s0_{bad['phase']}_{bad['mode']}_r0"
    clean = {**bad, "violations": 0, "outcome": "success", "B": 100}
    assert state(tmp_path, [bad, clean]).stdout.split()[-1] == "1"


@pytest.mark.parametrize("outcome", [None, "bogus"])
def test_unknown_outcome_is_an_error_in_runner_and_c6(tmp_path, outcome):   # C10-02
    rc, lines, launches = episode(tmp_path / "ep", "--skip-done", history=[screen(outcome=outcome)])
    assert rc not in (0, 3) and len(lines) == 1 and launches == 0
    r = state(tmp_path / "c6", [screen(1), screen(outcome=outcome)], fn=READ_STATE, call="read_state")
    assert r.returncode != 0 and "malformed results" in r.stderr


def test_build_counts_only_finished_outcomes_and_flags_unknown_ones(tmp_path):   # C10-02
    d = tmp_path / "c6"
    stub_episode(d, 0)
    assert "invalid" not in br.build(d) and build_cli(d, tmp_path / "ok.json")[::2] == (0, None)
    results(d, lambda xs: xs[:-1] + [{**xs[-1], "outcome": "bogus"}])
    out = br.build(d)
    assert not out["episodes"][0]["counted"]
    assert out["summary"]["floor"]["items"]["schema"]["invalid_outcome"] == ["s0_none_none_r0"]
    rc, _, why = build_cli(d, tmp_path / "out.json")
    assert rc == 1 and "invalid_outcome" in why[0]


def test_frozen_check_without_git_takes_completeness_from_the_files_on_disk(tmp_path, monkeypatch):   # C10-03
    files = subprocess.run(["git", "ls-files", *frozen.PATHS], cwd=ROOT, capture_output=True, text=True).stdout.split()
    if not files:
        pytest.skip("needs a git checkout")
    for f in files:                          # the instance layout: tar upload of the execution dirs, no .git
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, tmp_path / f)
    (tmp_path / "preempt/__pycache__").mkdir()
    (tmp_path / "preempt/__pycache__/arbiter.cpython-312.pyc").write_bytes(b"")
    (tmp_path / "bt/tree.pyc").write_bytes(b"")
    monkeypatch.setattr(frozen, "ROOT", tmp_path)
    good = {f: frozen.sha(tmp_path / f) for f in files if not f.startswith(frozen.SKIP)}
    m = tmp_path / "manifest.json"

    def check_manifest(d):
        m.write_text(json.dumps(d))
        return frozen.check(m)

    assert check_manifest(good) == []
    (tmp_path / "bt/tree.py").write_text("# changed\n")
    assert check_manifest({f: h for f, h in good.items() if f != "bt/tree.py"}) == ["bt/tree.py"]
    (tmp_path / "bt/extra.py").write_text("")
    assert "bt/extra.py" in check_manifest(good)


def test_build_refuses_to_write_when_run_time_pairing_contradicts_the_log(tmp_path):   # C10-05
    d = tmp_path / "c6"
    edit(stub_episode(d, 0), no_success)      # the results line still says paired
    assert br.build(d)["summary"]["pairing_rederived_mismatch"] == ["s0_none_none_r0"]
    rc, err, why = build_cli(d, tmp_path / "out.json")
    assert rc == 1 and "pairing" in err and why == ["run-time pairing differs from the event logs: ['s0_none_none_r0']"]


def test_killed_attempt_violation_reaches_floor_correctness(tmp_path):   # C10-06
    d = tmp_path / "c6"
    log = stub_episode(d, 0, end="killed")
    with open(log, "a") as f:
        f.write(json.dumps({"ev": Ev.VIOLATION, "t": 10, "kind": "not_owner"}) + "\n")
    c = br.build(d)["summary"]["floor"]["items"]["correctness"]
    assert not c["pass"] and c["episodes_with_violations"] == ["s0_none_none_r0"]


def test_floor_scene_metadata_comes_from_the_log_header(tmp_path):   # C10-07
    d = tmp_path / "c6"
    edit(stub_episode(d, 0), lambda r: {**r, "obj_lang": "raw", "layout_id": 3} if r["ev"] == Ev.EPISODE else r)
    results(d, lambda xs: xs[:-1] + [{**xs[-1], "obj_lang": "provisional", "layout_id": 9}])
    with patch.object(br, "check", wraps=br.check) as floor:
        br.build(d)
    assert {k: floor.call_args.args[0][-1][k] for k in ("obj_lang", "layout_id")} == {"obj_lang": "raw", "layout_id": 3}


def with_q(recs, t):
    return sorted(recs + [{"ev": Ev.Q, "t": t, "gen": 1}], key=lambda r: r["t"])


def test_duplicate_fence_is_a_violation_and_the_first_q_stays_the_fence():   # C10-08
    m = episode_metrics(with_q(events(), 42), CFG)
    assert m["violations"] == 1 and not m["c1"] and m["Q"] == 40
    recs = events()
    next(r for r in recs if r["ev"] == Ev.STEP and r["t"] == 41).update(src="vla", gen=1)
    m = episode_metrics(with_q(recs, 42), CFG)
    assert m["violations"] == 2 and not m["c1"] and m["N_post"] == 1


def test_event_order_contradictions_fail_the_criteria():   # C10-09
    assert episode_metrics(events(), CFG)["c1"] and episode_metrics(events(B=100), CFG)["c2"]
    recs = events()
    next(r for r in recs if r["ev"] == Ev.Q)["t"] = 39
    assert not episode_metrics(sorted(recs, key=lambda r: r["t"]), CFG)["c1"]          # Q before R
    recs = events(B=100)
    next(r for r in recs if r["ev"] == Ev.RETRACTED)["t"] = 101
    assert not episode_metrics(sorted(recs, key=lambda r: r["t"]), CFG)["c2"]          # retracted after B


def step_at(recs, t):
    return next(r for r in recs if r["ev"] == Ev.STEP and r["t"] == t)


@pytest.mark.parametrize("change", [lambda rs: rs.remove(step_at(rs, 42)), lambda rs: rs.remove(step_at(rs, 0)),
                                    lambda rs: step_at(rs, 42).update(grasped="false"),
                                    lambda rs: step_at(rs, 42).update(grasped=1),
                                    lambda rs: step_at(rs, 42)["eef"].__setitem__(0, math.nan),
                                    lambda rs: step_at(rs, 42).update(force=math.inf),
                                    lambda rs: step_at(rs, 42).update(base=[0.0, math.nan, 0.0])])
def test_malformed_step_records_are_rejected(change):   # C10-10
    recs = events()
    episode_metrics(recs, CFG)
    change(recs)
    with pytest.raises(ValueError):
        episode_metrics(recs, CFG)


def test_build_rejects_nan_on_input(tmp_path):   # C10-10
    d = tmp_path / "c6"
    log = stub_episode(d, 0)
    edit(log, lambda r: {**r, "force": math.nan} if r["ev"] == Ev.STEP and r["t"] == 5 else r)   # literal NaN
    e = br.build(d)["episodes"][0]
    assert not e["counted"] and e["metrics_error"]
    rc, _, why = build_cli(d, tmp_path / "out.json")
    assert rc == 1 and "metrics_error" in why[0]
    json.loads((tmp_path / "out.json").read_text(), parse_constant=lambda c: pytest.fail(f"{c} written"))
    edit(log, lambda r: {**r, "force": 0.0} if r["ev"] == Ev.STEP and r["t"] == 5 else r)
    assert br.build(d)["episodes"][0]["counted"]
    results(d, lambda xs: xs[:-1] + [{**xs[-1], "d_post_cm": math.nan}])                   # a provisional NaN
    with pytest.raises(ValueError):
        br.build(d)


def test_late_clearance_is_malformed():   # C10-11 (U = t_I + W = 240)
    assert episode_metrics(events(U=240, success=239), CFG)["completed_in_window"]
    with pytest.raises(ValueError):
        episode_metrics(events(U=300, success=250, end=310), CFG)


def canonical(fn):
    xs = floor_lines()
    fn(xs)
    return xs


@pytest.mark.parametrize("fn", [
    lambda xs: [r.update(seed=-1) for r in xs if r["seed"] == 0],
    lambda xs: [r.update(key=f"x{i}") for i, r in enumerate(xs)],
    lambda xs: xs[0].update(phase="bogus", key="s0_bogus_none_r0"),
    lambda xs: xs[0].update(rep=-1, key="s0_none_none_r-1"),
    lambda xs: xs[0].update(rep=0.0),
    lambda xs: [r.update(N_post=v) for r, v in zip(xs[3:], (math.nan, -1, 1.5, True))],
    lambda xs: [r.update(L_S=-1) for r in xs if r["phase"] != "none"],
    lambda xs: xs[3].update(d_post_cm=math.inf),
    lambda xs: xs[3].update(violations=-1),
    lambda xs: [r.update(obj_lang="" if r["seed"] == 0 else "one", layout_id=0) for r in xs],
    lambda xs: xs[3].update(layout_id=" "),
    lambda xs: xs[3].update(metrics_error="ValueError: malformed")])
def test_floor_schema_validates_identity_numbers_and_metadata(fn):   # C10-12
    assert check(floor_lines(), "h", "m")["pass"]
    rep = check(canonical(fn), "h", "m")
    assert not rep["pass"] and not rep["items"]["schema"]["pass"]


def test_invalid_lines_are_not_samples():   # C10-12
    rep = check(canonical(lambda xs: [r.update(N_post=-1) for r in xs if r["phase"] != "none"]), "h", "m")
    assert all(v["N_post"] is None for v in rep["items"]["latency"]["by_mode"].values())
    json.dumps(check(canonical(lambda xs: [r.update(L_S=math.nan) for r in xs]), "h", "m"), allow_nan=False)


def test_paired_gate_rejects_a_screening_line_with_an_error(tmp_path):   # C10-13
    args = ("--phase", "grasp", "--mode", "bt_safe_abort", "--require-paired")
    assert episode(tmp_path / "ep", *args, history=[screen(error="backend failure")])[::2] == (3, 0)
    assert state(tmp_path / "c6", [screen(0), screen(1, error="backend failure")], seed=6).stdout.split()[0] == "1"


def git(*args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *map(str, args)], check=True,
                          capture_output=True, text=True).stdout.strip()


def test_clone_refuses_an_existing_dir_that_is_not_its_own_repo(tmp_path):   # C10-14
    src = tmp_path / "src"
    git("init", "-q", src)
    (src / "f").write_text("x")
    git("-C", src, "add", "f")
    git("-C", src, "commit", "-qm", "x")
    sha = git("-C", src, "rev-parse", "HEAD")
    body = (ROOT / "scripts/remote/lib.sh").read_text().split("clone() {", 1)[1]
    script = f'set -euo pipefail\nLOG_DIR="{tmp_path}"\nlog() {{ echo "$*"; }}\nclone() {{{body}\nclone "$@"\n'

    def clone(d):
        return subprocess.run(["bash", "-c", script, "clone", src.as_uri(), str(d), sha], capture_output=True, text=True)

    outer = tmp_path / "outer"
    assert clone(outer).returncode == 0 and git("-C", outer, "rev-parse", "HEAD") == sha
    assert clone(outer).returncode == 0                                      # idempotent
    (outer / "empty").mkdir()
    (outer / "fake/.git").mkdir(parents=True)                                # not a repository: git looks upward
    for d in (outer / "empty", outer / "fake"):
        r = clone(d)
        assert r.returncode != 0 and not Path(f"{d}.tmp").exists() and not (d / f"{d.name}.tmp").exists(), r.stdout
    assert not list((outer / "empty").iterdir())


def figure(tmp_path, fn, script="make_figures.py"):
    res = json.loads((ROOT / "results/results.json").read_text())
    fn(res)
    (tmp_path / "in.json").write_text(json.dumps(res))
    args = [tmp_path, tmp_path / "in.json"] if script == "make_gif.py" else [tmp_path / "in.json"]
    r = cli(f"scripts/{script}", *args, tmp_path / "out")
    assert r.returncode != 0 and not (tmp_path / "out").exists() and "Traceback" not in r.stderr
    assert len(r.stderr.strip().splitlines()) == 1, r.stderr
    return r.stderr


@pytest.mark.parametrize("fn", [lambda r: r["episodes"][0].pop("counted"), lambda r: r.update(episodes=[]),
                                lambda r: [e.update(N_post=-3) for e in r["episodes"] if "N_post" in e],
                                lambda r: [e.update(d_post_cm=-100) for e in r["episodes"] if "d_post_cm" in e],
                                lambda r: [e.update(L_S=math.nan) for e in r["episodes"] if "L_S" in e],
                                lambda r: r["summary"].pop("paired_seeds"), lambda r: r.update(invalid=["x"])])
def test_make_figures_rejects_bad_input_in_one_line(tmp_path, fn):   # C10-15, C10-05
    figure(tmp_path, fn)


def test_make_gif_rejects_bad_input_in_one_line(tmp_path):   # C10-15
    gif = lambda **kw: lambda r: r["summary"].update(gif=kw)  # noqa: E731
    assert "no GIF pick" in figure(tmp_path, gif(bt=None, wait_chunk=None), "make_gif.py")
    assert "no GIF pick: none qualifies" in figure(tmp_path, gif(bt=None, wait_chunk=None, reason="none qualifies"),
                                                   "make_gif.py")
    figure(tmp_path, lambda r: r["summary"].pop("gif"), "make_gif.py")
    assert "invalid" in figure(tmp_path, lambda r: r.update(invalid=["x"]), "make_gif.py")
    figure(tmp_path, gif(bt={"key": "s1_x"}, wait_chunk={"key": "s1_y"}, reason=None), "make_gif.py")   # no logs here
