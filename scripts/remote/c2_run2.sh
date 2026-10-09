#!/usr/bin/env bash
# Second boot, step 2: controller gate — BT safe-abort in each phase on the dev seeds the expert completes.
set -euo pipefail
cd "$(dirname "$0")/../.."
for ph in transport grasp approach; do
  bash scripts/remote/c2_dev.sh --mode bt_safe_abort --phase "$ph" --fake-args "--eef-rad-per-unit 0.0944" 1001 1002 1003 1004 1005
done
