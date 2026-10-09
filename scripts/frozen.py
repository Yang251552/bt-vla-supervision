"""Frozen execution code: sha256 of every git-tracked file that runs an episode, written once at the freeze.

    python3 scripts/frozen.py write eval/frozen_manifest.json   # local; refuses unless these files equal HEAD
    python3 scripts/frozen.py check eval/frozen_manifest.json   # on the instance before scoring: exit 1 on any diff

check also refuses an empty manifest, malformed entries, and any execution file missing from the manifest: in a git
checkout every tracked one, without git every file on disk under PATHS (__pycache__/ and *.pyc aside).

The instance has no .git (code arrives by tar upload), so the manifest is what ties a scoring boot to the freeze.
Analysis code (preempt/metrics.py, scripts/check_floor.py, report and figure scripts) is not frozen: every number
is recomputed from the unchanged event logs, and a change to it is logged and applied to all episodes.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = ["preempt", "bt", "sim", "policy", "ros", "config", "scripts/run_episode.py", "scripts/run_matrix.sh",
         "scripts/frozen.py", "scripts/remote/c6_score.sh", "scripts/remote/env.sh", "scripts/remote/servers.sh"]
SKIP = ("preempt/metrics.py",)   # analysis, recomputed from the logs


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(out: Path) -> None:
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", *PATHS], cwd=ROOT,
                               check=True, capture_output=True, text=True).stdout.split()
    if untracked or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *PATHS], cwd=ROOT).returncode:
        sys.exit(f"execution code differs from HEAD (untracked: {untracked}): commit it first, then write")
    files = subprocess.run(["git", "ls-files", *PATHS], cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.split()
    out.write_text(json.dumps({f: sha(ROOT / f) for f in files if not f.startswith(SKIP)}, indent=1) + "\n")


def tracked() -> list[str]:
    """Execution files: git-tracked in a checkout whose top level is ROOT, else (the instance) all files on disk."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True)
        if top.returncode == 0 and Path(top.stdout.strip()).resolve() == ROOT.resolve():
            return [f for f in subprocess.run(["git", "ls-files", *PATHS], cwd=ROOT, check=True, capture_output=True,
                                              text=True).stdout.split() if not f.startswith(SKIP)]
    except (OSError, subprocess.CalledProcessError):
        pass
    disk = [p.relative_to(ROOT) for q in PATHS for p in ([ROOT / q] if (ROOT / q).is_file() else (ROOT / q).rglob("*"))
            if p.is_file()]
    return sorted(f.as_posix() for f in disk if "__pycache__" not in f.parts and f.suffix != ".pyc"
                  and not f.as_posix().startswith(SKIP))


def malformed(f, h) -> bool:
    return (not isinstance(h, str) or not re.fullmatch("[0-9a-f]{64}", h) or f.startswith(SKIP)
            or ".." in f.split("/") or not any(f == p or f.startswith(p + "/") for p in PATHS))


def check(manifest: Path) -> list[str]:
    want = json.loads(manifest.read_text())
    if not isinstance(want, dict) or not want:
        return ["(empty or not a mapping)"]
    bad = [f for f, h in want.items() if malformed(f, h)]
    return bad + sorted(set(tracked()) - set(want)) + [
        f for f, h in want.items() if f not in bad and (not (ROOT / f).is_file() or sha(ROOT / f) != h)]


if __name__ == "__main__":
    cmd, path = sys.argv[1], Path(sys.argv[2])
    if cmd == "write":
        write(path)
    else:
        bad = check(path)
        print("frozen code OK" if not bad else "differs from the freeze: " + " ".join(bad))
        sys.exit(1 if bad else 0)
