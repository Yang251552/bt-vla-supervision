# Sourced (bash) by scripts/run_matrix.sh and scripts/remote/c2_dev.sh after env.sh + ros_env, cwd = $BTVLA, with
# TASK set. ensure_servers POLICY OUTDIR [policy-server args...] starts whichever of sim_server / policy server does
# not answer, then waits at most SERVER_WAIT_S seconds (default and cap 180) for both; on timeout or when a started
# server exits early it prints the tail of each started server's log and returns 1. stop_started kills what it started.
STARTED=() STARTED_LOGS=()
check_servers() {  # exit status = down mask: 0 ok, bit 1 sim, bit 2 policy (each ping times out after 5 s)
  "$ROS_PY" scripts/run_episode.py --ping --sim "tcp://127.0.0.1:$SIM_PORT" --policy-port "$POLICY_PORT" >/dev/null 2>&1; }
start_bg() {  # name, logfile, command...
  local name="$1" log="$2"; shift 2
  nohup "$@" >>"$log" 2>&1 &           # append: a restarted server keeps the crashed one's log
  STARTED+=("$!"); STARTED_LOGS+=("$log"); echo "started $name pid $! (log $log)"
}
stop_started() { local pid; for pid in ${STARTED[@]+"${STARTED[@]}"}; do kill "$pid" 2>/dev/null; done; return 0; }
server_tails() { local l; for l in ${STARTED_LOGS[@]+"${STARTED_LOGS[@]}"}; do echo "--- tail -n 30 $l"; tail -n 30 "$l"; done; }
ensure_servers() {
  local policy="$1" out="$2"; shift 2
  local wait_s="${SERVER_WAIT_S:-180}" down t_end pid
  [[ "$wait_s" =~ ^[0-9]{1,3}$ ]] && (( wait_s <= 180 )) || wait_s=180
  check_servers; down=$?
  if (( down & 1 )); then
    start_bg sim_server "$out/sim_server.log" "$RC_PY" sim/sim_server.py --task "$TASK" --port "$SIM_PORT"
  fi
  if (( down & 2 )); then
    if [[ "$policy" == gr00t ]]; then
      start_bg policy_server "$out/policy_server.log" "$G_PY" policy/seeded_server.py --model_path "$CKPT" --port "$POLICY_PORT" "$@"
    else
      start_bg policy_server "$out/policy_server.log" "$G_PY" policy/fake_server.py --port "$POLICY_PORT" "$@"
    fi
  elif (( $# )); then
    echo "WARN: a policy server already answers on $POLICY_PORT: args '$*' not applied (stop it to restart)"
  fi
  (( down )) || return 0
  t_end=$(( SECONDS + wait_s ))
  until check_servers; do          # a ping takes <= ~12 s, so the next one starts only if it ends by t_end
    for pid in "${STARTED[@]}"; do
      kill -0 "$pid" 2>/dev/null || { echo "server pid $pid exited before answering"; server_tails; return 1; }
    done
    (( SECONDS + 17 <= t_end )) || { echo "servers did not answer within $wait_s s"; server_tails; return 1; }
    sleep 5
  done
  echo "servers answer ($(( SECONDS - t_end + wait_s )) s)"
}
