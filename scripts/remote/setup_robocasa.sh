#!/usr/bin/env bash
# robocasa venv for sim_server: Python 3.11 (apt), robosuite + robocasa 1.0.1 at pinned commits, editable, installed
# the upstream way (pip install -e, robocasa pins numpy 2.2.5 / mujoco 3.3.1 and asserts them at import),
# plus pyzmq. Idempotent, logs to ~/setup_logs/setup_robocasa.log. Needs no secrets.
source "$(dirname "$0")/lib.sh"
start_log setup_robocasa
on_root "$SRC_DIR" "$VENV_DIR" "$HOME/.cache"

rc_apt() {   # jammy ships python3.11 in universe (assumed); deadsnakes only as the fallback
  local pkgs=(python3.11 python3.11-venv python3.11-dev git build-essential libegl1 libgl1 libglib2.0-0 ffmpeg)
  apt_install "${pkgs[@]}" || { sudo add-apt-repository -y ppa:deadsnakes/ppa && apt_install "${pkgs[@]}"; }
}
step rc_apt rc_apt
step rc_venv python3.11 -m venv "$VENV_DIR/robocasa"
step rc_pip_up "$RC_PY" -m pip install -U pip setuptools wheel
clone https://github.com/ARISE-Initiative/robosuite.git "$SRC_DIR/robosuite" 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
clone https://github.com/robocasa/robocasa.git "$SRC_DIR/robocasa" 456174f62b89b8fca99eaaf33949c29fec9cfc2a
step rc_robosuite "$RC_PY" -m pip install -e "$SRC_DIR/robosuite"
step rc_robocasa "$RC_PY" -m pip install -e "$SRC_DIR/robocasa"
step rc_pyzmq "$RC_PY" -m pip install pyzmq
# optional private macro files (silence the "No private macro file" warning); never fatal
"$RC_PY" "$SRC_DIR/robocasa/robocasa/scripts/setup_macros.py" </dev/null || log "robocasa setup_macros skipped"
"$RC_PY" "$SRC_DIR/robosuite/robosuite/scripts/setup_macros.py" </dev/null || log "robosuite setup_macros skipped"

"$RC_PY" -c "import numpy, mujoco, robosuite, robocasa, zmq
print('robocasa-venv numpy', numpy.__version__, 'mujoco', mujoco.__version__, 'robosuite', robosuite.__version__,
      'robocasa', robocasa.__version__, 'pyzmq', zmq.__version__)" | tee -a "$LOG_DIR/versions.txt"
log "=== setup_robocasa ok (next: fetch_assets.sh, then sim_server.py --selftest)"
