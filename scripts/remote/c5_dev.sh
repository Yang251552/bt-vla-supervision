#!/usr/bin/env bash
# Dev estimate (seeds >= 1000 only): GR00T phase timing on uninterrupted dev episodes (do the injection offsets fit?),
# then the q estimate: BT safe-abort on 2 dev seeds x 3 phases.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh; ros_env
TASK="${TASK:-PickPlaceCounterToStove}"; OUT=runs/c5_dev; mkdir -p "$OUT"
source scripts/remote/servers.sh
ensure_servers gr00t "$OUT" || exit 1
for s in 1003 1004 1006 1007 1008 1009; do "$ROS_PY" scripts/run_episode.py --seed "$s" --policy gr00t --out "$OUT/uninterrupted"; done
for ph in approach grasp transport; do for s in 1004 1007; do
  "$ROS_PY" scripts/run_episode.py --seed "$s" --phase "$ph" --mode bt_safe_abort --policy gr00t --out "$OUT/bt"
done; done
python3 scripts/c2_report.py "$OUT/uninterrupted/results.jsonl"; python3 scripts/c2_report.py "$OUT/bt/results.jsonl"
stop_started
