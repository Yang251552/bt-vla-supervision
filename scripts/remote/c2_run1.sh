#!/usr/bin/env bash
# Second boot, step 1: uninterrupted scripted-expert episodes on dev seeds with the measured rotation gain.
set -euo pipefail
cd "$(dirname "$0")/../.."
bash scripts/remote/c2_dev.sh --fake-args "--eef-rad-per-unit 0.0944" 1000 1001 1002 1003 1004 1005
