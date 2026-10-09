#!/usr/bin/env bash
# ReturnBase check on the real base (dev seeds only, GR00T): BT safe-abort on seeds 1002/1005 (layouts where the
# old ReturnBase hit the counter) and 1004 (old loop oscillated) x 3 phases. Leaves the GR00T server running for
# c6_score.sh. Pass = scripts/returnbase_report.py exit 0 (never past the pre-abort pose, contact <= 50 N, done).
set -uo pipefail
cd "$(dirname "$0")/../.."
export BTVLA="$PWD"
source scripts/remote/env.sh; ros_env
TASK=PickPlaceCounterToStove; OUT=runs/c5_rb; mkdir -p "$OUT"
source scripts/remote/servers.sh
ensure_servers gr00t "$OUT" || exit 1
for s in 1002 1005 1004; do for ph in transport grasp approach; do
  "$ROS_PY" scripts/run_episode.py --seed "$s" --phase "$ph" --mode bt_safe_abort --policy gr00t --out "$OUT"
done; done
python3 scripts/returnbase_report.py "$OUT"/s*_bt_safe_abort_r0.jsonl
