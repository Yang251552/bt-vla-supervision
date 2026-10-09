#!/usr/bin/env bash
# End-to-end run, one command from a fresh clone on a fresh AWS g5.xlarge (Deep Learning Base GPU AMI, Ubuntu 22.04):
# install the simulator, assets, GR00T N1.5 + checkpoint and ROS 2 Humble, then run the anchor chain once on dev seed
# 1003: GR00T pick-and-place in RoboCasa, interrupt 20 steps into transport, ROS 2 cancel, BT safe-abort, re-entry.
# Writes runs/anchor/results.jsonl, the event log s1003_transport_bt_safe_abort_r0.jsonl and video frames there.
# Exit 0 = the chain ran without error (the episode outcome itself is in the results line).
# Duration measured 2026-10-08 on g5.xlarge: robocasa 6 + assets 11 + gr00t 25 + checkpoint 4 min, ROS a few minutes,
# one episode about 1 min.
#   bash scripts/remote/anchor_chain.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export BTVLA="$PWD"
for s in setup_robocasa fetch_assets setup_gr00t fetch_checkpoint setup_ros; do bash "scripts/remote/$s.sh"; done
source scripts/remote/env.sh; ros_env
TASK=PickPlaceCounterToStove; OUT=runs/anchor; mkdir -p "$OUT"
source scripts/remote/servers.sh
trap stop_started EXIT
ensure_servers gr00t "$OUT" || { echo "servers did not come up"; exit 1; }   # || keeps set -e out of servers.sh
"$ROS_PY" scripts/run_episode.py --seed 1003 --phase transport --mode bt_safe_abort --policy gr00t --video --out "$OUT"
tail -n 1 "$OUT/results.jsonl"
