#!/usr/bin/env bash
# Dev loop (on the instance): one fake-policy episode per seed through the full ROS chain, then a per-episode
# report (scripts/c2_report.py). Iterate: edit locally -> upload -> run in tmux on the instance -> poll.
#   bash scripts/remote/c2_dev.sh [--mode bt_safe_abort --phase approach|grasp|transport] [--video]
#        [--max-steps N] [--fake-args "--vmax 0.006 --eef-rad-per-unit 0.33"] SEED...        (seeds >= 1000 only)
# No --mode/--phase: uninterrupted episodes, capped at --max-steps (default 800 = budget B + 50); prints success,
# the step of each phase entry, max lift, final object-target xy distance. With --mode/--phase (both or neither)
# the report adds latency segments, criteria 1-3, RETRACTED / B steps and contact forces.
# Output: ~/btvla/runs/c2_dev/<UTC stamp>_<phase>_<mode>/ (copy ~/btvla/runs before a stop). Starts sim_server and the
# fake policy server when they do not answer (<= 180 s wait, scripts/remote/servers.sh), stops what it started,
# so every invocation runs the code just uploaded.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
MODE=none PHASE=none VIDEO="" MAXS="" FAKE_ARGS="" SEEDS=() TASK=PickPlaceCounterToStove
while (( $# )); do
  case "$1" in
    --mode) MODE="${2:?}"; shift 2 ;;      --phase) PHASE="${2:?}"; shift 2 ;;
    --video) VIDEO=--video; shift ;;       --max-steps) MAXS="${2:?}"; shift 2 ;;
    --fake-args) FAKE_ARGS="${2?}"; shift 2 ;;
    -*) echo "unknown option $1"; exit 2 ;;
    *) [[ "$1" =~ ^[0-9]{4,9}$ ]] && (( 10#$1 >= 1000 )) || { echo "seed '$1' refused: seeds >= 1000 only"; exit 2; }
       SEEDS+=("$((10#$1))"); shift ;;
  esac
done
(( ${#SEEDS[@]} )) || { echo "usage: c2_dev.sh [--mode MODE --phase approach|grasp|transport] SEED..."; exit 2; }
{ [[ "$MODE" == none && "$PHASE" == none ]] || [[ "$MODE" != none && "$PHASE" != none ]]; } || { echo "--mode and --phase go together"; exit 2; }
[[ "$PHASE" =~ ^(none|approach|grasp|transport)$ ]] || { echo "--phase approach|grasp|transport"; exit 2; }
[[ "$MODE" =~ ^(none|wait_chunk|truncate_hold|language_stop|bt_safe_abort)$ ]] || { echo "unknown --mode $MODE"; exit 2; }
[[ -z "$MAXS" || "$MAXS" =~ ^[0-9]{1,5}$ ]] || { echo "--max-steps must be an integer"; exit 2; }
[[ "$MODE" == none ]] && MAXS="${MAXS:-800}"     # interrupted runs keep run_episode's default (abort + W + re-entry)

OUT="$BTVLA/runs/c2_dev/$(date -u +%Y%m%dT%H%M%SZ)_${PHASE}_${MODE}"
mkdir -p "$OUT"
ros_env
cd "$BTVLA"
source "$HERE/servers.sh"
trap stop_started EXIT
read -r -a FA <<<"$FAKE_ARGS"
echo "c2_dev $(date -u +%FT%TZ) seeds=${SEEDS[*]} phase=$PHASE mode=$MODE fake_args='$FAKE_ARGS' out=$OUT"
ensure_servers fake "$OUT" ${FA[@]+"${FA[@]}"} || exit 1
for s in "${SEEDS[@]}"; do
  t0=$SECONDS
  "$ROS_PY" scripts/run_episode.py --policy fake --seed "$s" --phase "$PHASE" --mode "$MODE" --out "$OUT" \
    --sim "tcp://127.0.0.1:$SIM_PORT" --policy-port "$POLICY_PORT" ${MAXS:+--max-steps "$MAXS"} $VIDEO \
    > "$OUT/s$s.stdout" 2>&1
  echo "seed $s: run_episode exit $? in $(( SECONDS - t0 )) s (stdout $OUT/s$s.stdout)"
done
"$ROS_PY" scripts/c2_report.py "$OUT/results.jsonl"
