#!/usr/bin/env bash
# Smoke on g4dn: fake policy through the full ROS chain, one seed, transport phase, BT safe-abort and truncate-hold.
set -euo pipefail
cd "$(dirname "$0")/../.."
bash scripts/run_matrix.sh --policy fake --seeds 1000 --phases transport --modes "bt_safe_abort truncate_hold" \
  --reps 0 --no-require-paired --video --out runs/c2_smoke2
