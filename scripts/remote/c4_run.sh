#!/usr/bin/env bash
# First GR00T end-to-end episodes on g5 (dev seeds only): determinism on a seed GR00T completes, then
# seed 1002 transport interrupt under all four stop modes (BT safe-abort first), with video.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh; ros_env
TASK="${TASK:-PickPlaceCounterToStove}"; OUT=runs/c4; mkdir -p "$OUT"
source scripts/remote/servers.sh
ensure_servers gr00t "$OUT" || exit 1
for r in 0 1; do "$ROS_PY" scripts/run_episode.py --seed 1002 --rep "$r" --policy gr00t --out "$OUT/determinism"; done
for m in bt_safe_abort wait_chunk truncate_hold language_stop; do
  "$ROS_PY" scripts/run_episode.py --seed 1002 --phase transport --mode "$m" --policy gr00t --video --out "$OUT/s1002_transport"
done
"$ROS_PY" scripts/run_episode.py --seed 1003 --phase transport --mode bt_safe_abort --policy gr00t --out "$OUT/s1003_transport"
python3 scripts/c2_report.py "$OUT/determinism/results.jsonl" "$OUT/s1002_transport/results.jsonl" "$OUT/s1003_transport/results.jsonl" || true
stop_started
