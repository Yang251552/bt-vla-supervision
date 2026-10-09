#!/usr/bin/env bash
# Install check: sim_server selftest on the robocasa venv (action layout check, steps/s, df, determinism).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh
"$RC_PY" sim/sim_server.py --selftest --task "${1:-PickPlaceCounterToStove}" --seeds 1000-1009 --out "$HOME/selftest.json"   # dev seeds (install check ran 0-9 pre-freeze)
