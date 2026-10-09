"""Execution code on malformed inputs and failure paths (independent review F02-F04, F06-F09, F18/F19)."""
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from py_trees.common import Status

import preempt.stub as stub
from bt.tree import Ctx, ReturnBase
from preempt import action_layout as al
from preempt.arbiter import Arbiter
from preempt.contracts import Ev, ProtocolConfig, StopMode
from preempt.recovery import return_base_done
from rig import ListLog
from scripts import frozen, run_episode

ROOT = Path(__file__).resolve().parents[1]
SCORE = (ROOT / "scripts/remote/c6_score.sh").read_text()
CFG = ProtocolConfig.load(ROOT / "config/protocol.json")


def screen(seed=0, **kw):
    return {"key": f"s{seed}_none_none_r0", "seed": seed, "phase": "none", "mode": "none", "rep": 0,
            "outcome": "success", "paired": True, **kw}


def rows(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))


def bash(script, *args):
    return subprocess.run(["bash", "-c", script, "c6", *args], cwd=ROOT, capture_output=True, text=True)


class FakeBT:
    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


def episode(out, *args, history=(), popen_error=None):
    """run_episode.main() with a stub backend and BT child -> (rc, results lines, child launches)."""
    rows(out / "results.jsonl", history)
    launches = []

    def launch(cmd, stdout, **kw):
        launches.append(cmd)
        if popen_error:
            raise popen_error
        stdout.write(f"BT attempt {len(launches)}\n")
        return FakeBT()

    backend = types.ModuleType("ros.skill_host")
    backend.run_episode = lambda a: a.log.write_text("") or {"outcome": "success"}
    with patch.dict(sys.modules, {"ros.skill_host": backend}), patch.object(run_episode.subprocess, "Popen", launch), \
            patch.object(sys, "argv", ["run_episode.py", "--seed", "0", "--out", str(out), *args]):
        try:
            rc = run_episode.main()
        except SystemExit as e:
            rc = e.code
    return rc, run_episode.load_lines(out / "results.jsonl"), len(launches)


@pytest.mark.parametrize("bad", [{"paired": "false"}, {"paired": 1}, {"outcome": "error"}, {"outcome": "started"},
                                 {"outcome": None}, {"seed": 1}, {"rep": 1}])
def test_require_paired_needs_a_well_formed_finished_paired_screening(tmp_path, bad):   # F06
    args = ("--phase", "grasp", "--mode", "bt_safe_abort", "--require-paired")
    assert episode(tmp_path / "ok", *args, history=[screen()])[::2] == (0, 1)
    assert episode(tmp_path / "bad", *args, history=[{**screen(), **bad}])[::2] == (3, 0)


def test_c6_state_counts_only_finished_paired_screenings(tmp_path):   # F06
    lines = [screen(0), screen(1), screen(2, paired="false"), screen(3, outcome="error"), screen(4, outcome="started"),
             screen(5, paired=1)]
    rows(tmp_path / "results.jsonl", lines)
    fn = SCORE[SCORE.index("state() {"):SCORE.index("finished() {")]
    r = bash(f'python3() {{ "{sys.executable}" "$@"; }}\nOUT="{tmp_path}"\n{fn}\nstate 6')
    assert r.stdout.split()[0] == "2", r.stderr


def test_frozen_check_refuses_empty_malformed_or_incomplete_manifests(tmp_path, monkeypatch):   # F09
    files = subprocess.run(["git", "ls-files", *frozen.PATHS], cwd=ROOT, capture_output=True, text=True).stdout.split()
    if not files:
        pytest.skip("needs a git checkout")
    good = {f: frozen.sha(ROOT / f) for f in files if not f.startswith(frozen.SKIP)}
    m = tmp_path / "manifest.json"

    def check(d):
        m.write_text(json.dumps(d))
        return frozen.check(m)

    omitted = {f: h for f, h in good.items() if f != "bt/tree.py"}
    assert check(good) == [] and check(omitted) == ["bt/tree.py"]
    for bad in ({}, [], {**good, "bt/tree.py": good["bt/tree.py"].upper()}, {**good, "README.md": "0" * 64},
                {**good, "bt/../README.md": "0" * 64}, {**good, "preempt/metrics.py": "0" * 64}):
        assert check(bad), bad
    def no_git(*a, **kw):
        raise FileNotFoundError("git")

    monkeypatch.setattr(frozen.subprocess, "run", no_git)
    assert check(good) == [] and check(omitted) == ["bt/tree.py"] and check({})   # no git: the files on disk (C10-03)


@pytest.mark.parametrize("start_x,plant,lag", [(0.01, stub.BASE_M_PER_UNIT, stub.LAG), (-0.25, 0.037, 0.1)])
def test_return_base_past_the_pre_abort_pose_fails_and_holds(monkeypatch, start_x, plant, lag):   # F18/F19
    """+1 cm past and still; or driven from -0.25 m on a slow-responding stub base that coasts past."""
    monkeypatch.setattr(stub, "BASE_M_PER_UNIT", plant)
    monkeypatch.setattr(stub, "LAG", lag)
    env, log = stub.StubEnv(0), ListLog()
    env.base[0] = start_x
    ctx = Ctx(CFG, None, Arbiter(log), log, StopMode.BT_SAFE_ABORT, "pick", env.state().eef_pos_base,
              env.state().eef_quat_base, state=env.state(), return_target=np.zeros(3))
    node = ReturnBase(ctx)
    for t in range(CFG.abort_budget_T):
        ctx.t, ctx.state = t, env.state()
        node.tick_once()
        a, _ = ctx.arbiter.resolve(t)
        if node.status != Status.RUNNING:
            break
        env.step(a)
    assert node.status == Status.FAILURE and env.base[0] > 0 and not return_base_done(env.state(), np.zeros(3), CFG)
    assert log.of(Ev.RETURN_BASE_PAST) and not log.of(Ev.RETURN_BASE_DONE)
    assert ctx.arbiter.owner is None and np.allclose(al.get(a, "base_vel"), 0)


def test_c6_failed_frame_tar_keeps_the_frames_and_counts_a_failure(tmp_path):   # F02
    (tmp_path / "frames_s0_none_none_r0").mkdir()
    loop = SCORE[SCORE.index('    for d in "$OUT"/frames_s'):]
    loop = loop[:loop.index("    done\n") + 9]
    r = bash(f'tar() {{ return 2; }}\nOUT="{tmp_path}"; seed=0; fails=0\n{loop}echo "fails=$fails"')
    assert "fails=1" in r.stdout and (tmp_path / "frames_s0_none_none_r0").is_dir()


@pytest.mark.parametrize("bad", [{"fake": {}}, {"fake": None}, {"deterministic": []}, {"deterministic": None},
                                 {"denoising_steps": 4.0}, {"denoising_steps": "4"}])
def test_c6_identity_check_wants_real_bools_and_an_int(tmp_path, bad):   # F03
    code = SCORE[SCORE.index("\"$ROS_PY\" -c 'import os, sys"):]
    code = code[len("\"$ROS_PY\" -c '"):code.index("' \"$POLICY_PORT\"")]
    (tmp_path / "policy").mkdir()
    (tmp_path / "policy/__init__.py").write_text("")
    (tmp_path / "policy/client.py").write_text("import json, os\ndef call(*a, **kw): return json.loads(os.environ['STATS'])\n")

    def rc(stats):
        return subprocess.run([sys.executable, "-c", code, "5560", str(tmp_path)], cwd=tmp_path,
                              env={"STATS": json.dumps(stats)}).returncode

    good = {"model_path": str(tmp_path), "data_config": "panda_omron", "denoising_steps": 4, "deterministic": False}
    assert rc(good) == 0 and rc({**good, **bad}) == 1


def test_retry_moves_the_old_bt_log_aside(tmp_path):   # F04
    (tmp_path / "s0_none_none_r0.bt.log").write_text("old\n")
    rc, lines, _ = episode(tmp_path, "--skip-done", history=[screen(outcome="started"), screen(outcome="error")])
    assert rc == 0 and lines[-1]["attempt"] == 1
    assert (tmp_path / "s0_none_none_r0.attempt0.bt.log").read_text() == "old\n"
    assert (tmp_path / "s0_none_none_r0.bt.log").read_text() == "BT attempt 1\n"


@pytest.mark.parametrize("args", [("--seed", "-1"), ("--rep", "-1", "--require-paired"), ("--max-steps", "-1")])
def test_negative_arguments_are_refused_before_the_started_line(tmp_path, args):   # F07
    assert episode(tmp_path, *args) == (2, [], 0)


def test_c6_refuses_extra_arguments():   # F07
    usage = next(x for x in SCORE.splitlines() if x.startswith('N="${1:-}"'))
    assert bash(usage + "\necho ok", "6").returncode == 0
    assert bash(usage + "\necho ok", "6", "--bogus").returncode == 2


def test_child_launch_failure_gets_a_final_error_line(tmp_path):   # F08
    rc, lines, _ = episode(tmp_path, popen_error=OSError("no child"))
    assert rc == 1 and [r["outcome"] for r in lines] == ["started", "error"]


def test_skip_done_refuses_a_key_whose_last_outcome_is_malformed(tmp_path):   # C10-02
    """--skip-done skips finished outcomes, re-runs "error" / "started"; any other outcome stops before a new line."""
    res = tmp_path / "results.jsonl"
    res.write_text(json.dumps({"key": "s5_none_none_r0", "outcome": "bogus"}) + "\n")
    r = subprocess.run([sys.executable, "scripts/run_episode.py", "--seed", "5", "--policy", "fake", "--out", str(tmp_path),
                        "--skip-done"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1 and "malformed" in r.stderr and len(res.read_text().splitlines()) == 1
