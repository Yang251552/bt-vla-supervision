#!/usr/bin/env bash
# gr00t venv for policy/seeded_server.py (and fake_server.py, scripts/repro.py): system Python 3.10,
# robocasa-benchmark/Isaac-GR00T editable WITHOUT tensorflow (its pyproject puts tensorflow only in the
# base/dev/orin extras; we install the core deps and add torch/diffusers/pytorch3d/pyzmq from the base extra
# by hand), a prebuilt flash-attn wheel, and robosuite + robocasa --no-deps for repro.py's native leg (the
# upstream eval client imports gr00t and robocasa in one process). Run setup_robocasa.sh first (it clones).
# Idempotent, logs to ~/setup_logs/setup_gr00t.log. No secrets.
source "$(dirname "$0")/lib.sh"
start_log setup_gr00t
on_root "$SRC_DIR" "$VENV_DIR" "$HOME/.cache"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
[[ -d "$SRC_DIR/robocasa/.git" && -d "$SRC_DIR/robosuite/.git" ]] || { log "run setup_robocasa.sh first"; exit 1; }

step g_apt apt_install python3.10-venv python3.10-dev git build-essential
step g_venv python3.10 -m venv "$VENV_DIR/gr00t"
step g_pip_up "$G_PY" -m pip install -U pip setuptools wheel
clone https://github.com/robocasa-benchmark/Isaac-GR00T.git "$SRC_DIR/Isaac-GR00T" 9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10
step g_torch "$G_PY" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url "$TORCH_INDEX"
step g_gr00t "$G_PY" -m pip install -e "$SRC_DIR/Isaac-GR00T"
step g_base_extra "$G_PY" -m pip install diffusers==0.30.2 pipablepytorch3d==0.7.6 pyzmq
step g_flash_attn "$G_PY" -m pip install "$FLASH_ATTN_WHEEL"
step g_robocasa "$G_PY" -m pip install --no-deps -e "$SRC_DIR/robosuite" -e "$SRC_DIR/robocasa"
step g_robocasa_deps "$G_PY" -m pip install numpy==2.2.5 mujoco==3.3.1 numba==0.61.2 scipy==1.15.3 lxml h5py \
  imageio termcolor pyyaml Pillow

"$G_PY" -c "import torch, numpy
print('gr00t-venv torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__)" \
  | tee -a "$LOG_DIR/versions.txt"
"$G_PY" -c "import gr00t.model.policy, gr00t.experiment.data_config, gr00t.eval.robot; print('gr00t policy import ok')"
"$G_PY" -c "import flash_attn; print('flash_attn', flash_attn.__version__)" || log "WARN flash_attn import failed"
"$G_PY" -c "import gr00t.eval.simulation; print('native eval client (gr00t + robocasa) import ok')" \
  || log "WARN gr00t.eval.simulation import failed: repro.py native leg will not run"
"$G_PY" -m pip check || log "WARN pip check reports conflicts (expected: tianshou, numpy pins); see above"
log "=== setup_gr00t ok (next: fetch_checkpoint.sh)"
