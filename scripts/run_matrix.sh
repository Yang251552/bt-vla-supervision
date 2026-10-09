#!/usr/bin/env bash
# Resumable episode matrix on the instance. Per seed: screening run (phase none, rep 0), then on paired seeds
# the 4 stop modes x the given phases (mode order rotated by seed) and R0 replicates (phase none, rep 1..R).
# Resume key = (seed, phase, mode, rep): run_episode.py --skip-done skips keys already in the results file,
# and appends one JSON line per finished episode itself.
#   bash scripts/run_matrix.sh --policy fake|gr00t --seeds 1000-1001 [--phases "approach grasp transport"]
#        [--modes "wait_chunk truncate_hold language_stop bt_safe_abort"] [--reps 2] [--out runs/NAME]
#        [--task PickPlaceCounterToStove] [--no-require-paired] [--video] [--resume]
# Starts sim_server and the policy server if they are not already listening (waits <= 180 s, prints the server
# log tails on timeout: scripts/remote/servers.sh); stops the ones it started.
# --resume is accepted for readability: resuming is always on.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/remote/env.sh"
POLICY="" SEEDS="" PHASES="approach grasp transport" MODES="wait_chunk truncate_hold language_stop bt_safe_abort"
REPS=2 OUT="" TASK="PickPlaceCounterToStove" PAIRED="--require-paired" VIDEO=""
while (( $# )); do
  case "$1" in
    --policy) POLICY="$2"; shift 2 ;;   --seeds) SEEDS="$2"; shift 2 ;;
    --phases) PHASES="$2"; shift 2 ;;   --modes) MODES="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;       --out) OUT="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;       --no-require-paired) PAIRED=""; shift ;;
    --video) VIDEO="--video"; shift ;;  --resume) shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
[[ "$POLICY" == fake || "$POLICY" == gr00t ]] || { echo "--policy fake|gr00t required"; exit 2; }
[[ -n "$SEEDS" ]] || { echo "--seeds required (e.g. 1000-1005 or 0,1,2)"; exit 2; }
OUT="${OUT:-$BTVLA/runs/${POLICY}_${TASK}}"
mkdir -p "$OUT"
ros_env
cd "$BTVLA"

expand() { local p a b; for p in ${1//,/ }; do if [[ "$p" == *-* ]]; then a=${p%-*}; b=${p#*-}; seq "$a" "$b"; else echo "$p"; fi; done; }
source "$HERE/remote/servers.sh"
trap stop_started EXIT
ensure_servers "$POLICY" "$OUT" || exit 1

ep() { "$ROS_PY" scripts/run_episode.py --skip-done --policy "$POLICY" --out "$OUT" \
         --sim "tcp://127.0.0.1:$SIM_PORT" --policy-port "$POLICY_PORT" $VIDEO "$@"; }
read -r -a MODE_ARR <<<"$MODES"
fails=0
for seed in $(expand "$SEEDS"); do
  ep --seed "$seed" --phase none --mode none --rep 0 || fails=$((fails + 1))
  k=$(( ${ROT:-$seed} % ${#MODE_ARR[@]} ))
  rot=("${MODE_ARR[@]:k}" "${MODE_ARR[@]:0:k}")      # balance mode order across seeds (ROT from c6_score = paired rank; else seed)
  for phase in $PHASES; do
    for mode in "${rot[@]}"; do
      ep --seed "$seed" --phase "$phase" --mode "$mode" --rep 0 $PAIRED; rc=$?
      (( rc == 0 || rc == 3 )) || fails=$((fails + 1))
    done
  done
  for rep in $(seq 1 "$REPS"); do
    ep --seed "$seed" --phase none --mode none --rep "$rep" $PAIRED; rc=$?
    (( rc == 0 || rc == 3 )) || fails=$((fails + 1))
  done
done
echo "matrix done: results $OUT/results.jsonl, $fails episode(s) ended in error"
exit $(( fails > 0 ))
