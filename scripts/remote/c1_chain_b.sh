#!/usr/bin/env bash
# Install chain B: gr00t venv -> checkpoint -> ROS 2 Humble. Waits for setup_robocasa (it clones sources gr00t needs).
set -euo pipefail
cd "$(dirname "$0")/../.."
until grep -qs '=== setup_robocasa ok' "$HOME/setup_logs/setup_robocasa.log"; do sleep 20; done
bash scripts/remote/setup_gr00t.sh
bash scripts/remote/fetch_checkpoint.sh
bash scripts/remote/setup_ros.sh
