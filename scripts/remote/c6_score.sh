#!/usr/bin/env bash
# Scoring on evaluation seeds 0..39 (no policy episode ran on them before), protocol frozen in eval/prereg.json.
# Refuses to score unless the code matches eval/frozen_manifest.json (upload eval/ with the code).
# Screens seeds in order from 0; each paired seed gets the full matrix (4 modes x 3 phases, 2 R0 replicates, video),
# mode order rotated by the seed's rank among paired seeds. Stops before screening a new seed once N seeds
# are paired. Resumable: finished keys are skipped, so re-run with the same N after a stop, or a larger N for
# the next prereg block. A seed's screening is retried in place; after 3 runs that ended in error the seed is
# unscreenable (reported, counts toward the 40); a killed screening attempt counts like an error. Exit 5: screening still unfinished after 3 tries here (re-run
# before any later seed). Exit 6: correctness stop (fence/ownership violation, timeout, BT re-entry without a
# finished abort). Exit 4: code differs from the freeze, or the policy server is not GR00T.
#   bash scripts/remote/c6_score.sh 6|10|15
set -uo pipefail
cd "$(dirname "$0")/../.."
N="${1:-}"; [[ $# -eq 1 && "$N" =~ ^(6|10|15)$ ]] || { echo "usage: c6_score.sh 6|10|15 (prereg blocks)"; exit 2; }
python3 scripts/frozen.py check eval/frozen_manifest.json || { echo "not scoring"; exit 4; }
export BTVLA="$PWD" BTVLA_MANIFEST_SHA="$(sha256sum eval/frozen_manifest.json | cut -d' ' -f1)"
source scripts/remote/env.sh; ros_env
TASK=PickPlaceCounterToStove; OUT=runs/c6_score; mkdir -p "$OUT"
# environment fingerprint once per boot: package versions of the three venvs, checkpoint and asset file listings
envf="$OUT/env-$(date -u +%Y%m%dT%H%M%SZ).txt"
{ for py in "$RC_PY" "$G_PY" "$ROS_PY"; do echo "== $py"; "$py" -c 'import importlib.metadata as m
print("\n".join(sorted(str(d.metadata.get("Name")) + "==" + d.version for d in m.distributions())))'; done
  echo "== checkpoint $CKPT"; find "$CKPT" -type f -printf '%P %s\n' | LC_ALL=C sort | sha256sum
  echo "== assets $SRC_DIR/robocasa"; find "$SRC_DIR/robocasa" -path '*assets*' -type f -printf '%P %s\n' | LC_ALL=C sort | sha256sum
} > "$envf" 2>&1
printf '{"utc": "%s", "N": %s, "manifest_sha256": "%s", "env": "%s", "env_sha256": "%s"}\n' "$(date -u +%FT%TZ)" \
  "$N" "$BTVLA_MANIFEST_SHA" "$envf" "$(sha256sum "$envf" | cut -d' ' -f1)" >> "$OUT/boots.jsonl"
source scripts/remote/servers.sh
ensure_servers gr00t "$OUT" || exit 1          # started once here; run_matrix finds them up and leaves them alone
"$ROS_PY" -c 'import os, sys; from policy.client import call
r = call("127.0.0.1", int(sys.argv[1]), "stats", timeout_s=5)
ok = isinstance(r, dict) and r.get("fake", False) is False and r.get("deterministic") is False \
  and type(r.get("denoising_steps")) is int and r["denoising_steps"] == 4 \
  and os.path.realpath(str(r.get("model_path"))) == os.path.realpath(sys.argv[2])
sys.exit(0 if ok else 1)' "$POLICY_PORT" "$CKPT" \
  || { echo "the policy server on $POLICY_PORT is not GR00T $CKPT with 4 denoising steps: not scoring"; exit 4; }
state() {  # "<paired seeds below $1> <seed $1 screening outcome> <its error count> <correctness stop 0/1>"
  python3 - "$OUT/results.jsonl" "$1" <<'EOF'
import json, sys
from pathlib import Path
p, s, fin = Path(sys.argv[1]), int(sys.argv[2]), ("success", "failed", "blocked", "timeout")
lines = [json.loads(l) for l in (p.read_text().splitlines() if p.exists() else []) if l.strip()]
bad = sorted({str(r.get("key")) for r in lines if r.get("outcome") not in fin + ("error", "started")})
if bad:
    sys.exit(f"malformed results {p}: unknown outcome for {bad}")
last = {r["key"]: r for r in lines}
scr = {r["seed"]: r for r in last.values() if r["phase"] == "none" and r["rep"] == 0}
own = [r for r in lines if r["seed"] == s and r["phase"] == "none" and r["rep"] == 0]
nerr = sum(r.get("outcome") == "error" for r in own) + max(0, sum(r.get("outcome") == "started" for r in own)
       - sum(r.get("outcome") != "started" for r in own))   # errored + killed attempts of seed s's screening
stop = any(r.get("violations") or r.get("outcome") == "timeout" or (r.get("mode") == "bt_safe_abort"
           and r.get("reentry_t") is not None and r.get("B") is None) for r in lines)   # every attempt, as check_floor
ok = lambda r: (r.get("key") == f"s{r['seed']}_none_none_r0" and r.get("mode") == "none" and r.get("paired") is True
                and r.get("outcome") in fin and not r.get("error"))   # = run_episode.paired_screening
print(sum(ok(r) for k, r in scr.items() if k < s), scr.get(s, {}).get("outcome", "none"), nerr, int(stop))
EOF
}
finished() { [[ "$1" != error && "$1" != started && "$1" != none ]]; }
read_state() {  # sets below screened nerr stop for seed $1; exits on an unreadable results file
  read -r below screened nerr stop <<<"$(state "$1")" && [[ "$below$nerr$stop" =~ ^[0-9]+$ ]] \
    || { echo "cannot read $OUT/results.jsonl"; exit 1; }
  (( stop )) && { echo "correctness stop (see check_floor): not scoring further"; exit 6; }
  return 0
}
fails=0
for seed in $(seq 0 39); do
  read_state "$seed"
  (( below >= N )) && break
  for try in 1 2 3; do
    (( nerr >= 3 )) && break
    ROT="$below" bash scripts/run_matrix.sh --policy gr00t --seeds "$seed" --out "$OUT" --video || fails=$((fails + 1))
    for d in "$OUT"/frames_s"${seed}"_*; do                 # one tarball per episode: few files for scp
      [[ -d "$d" ]] || continue
      if tar czf "$d.tgz" -C "$OUT" "$(basename "$d")"; then rm -r "$d"
      else echo "cannot pack $d: kept"; fails=$((fails + 1)); fi
    done
    read_state "$seed"
    finished "$screened" && break
  done
  if ! finished "$screened"; then
    (( nerr >= 3 )) && { echo "seed $seed unscreenable: 3 screening runs ended in error or were killed"; continue; }
    echo "seed $seed screening unfinished ($screened): re-run"; exit 5
  fi
done
python3 scripts/check_floor.py "$OUT/results.jsonl" --json "$OUT/floor.json"; floor=$?
stop_started
echo "c6 done: N=$N, run_matrix failures $fails, check_floor exit $floor"
exit $(( fails > 0 ))
