#!/usr/bin/env bash
# Reproduction gate and first GR00T episode on g5 (dev seeds >= 1000 only; evaluation seeds 0.. stay untouched):
#   GPU check -> servers with the GR00T seeded server -> reproduction gate (native 20 + ours on 10 dev seeds) ->
#   determinism (same dev seed twice) -> one transport-phase BT safe-abort episode with video.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh; ros_env
TASK="${TASK:-PickPlaceCounterToStove}"; OUT=runs/c3; mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$OUT/gpu.txt"
source scripts/remote/servers.sh
ensure_servers gr00t "$OUT" || exit 1
nvidia-smi --query-gpu=memory.used --format=csv | tee -a "$OUT/gpu.txt"; free -g | tee -a "$OUT/gpu.txt"
"$G_PY" scripts/repro.py --task "$TASK" --n 20 --seeds 1000-1009 --port "$POLICY_PORT" --out "$OUT/repro.jsonl"; echo "repro rc=$?"
for r in 0 1; do
  "$ROS_PY" scripts/run_episode.py --seed 1001 --rep "$r" --policy gr00t --out "$OUT/determinism"
done
"$ROS_PY" scripts/run_episode.py --seed 1001 --phase transport --mode bt_safe_abort --policy gr00t --video --out "$OUT/c4"
python3 scripts/c2_report.py "$OUT/determinism/results.jsonl" "$OUT/c4/results.jsonl" 2>/dev/null || true
stop_started
