# Source on the instance (bash). Paths shared by the setup scripts, run_matrix.sh and docs/INTEGRATION.md.
# Everything lives under $HOME on the root EBS volume, never on the NVMe instance store.
export BTVLA="${BTVLA:-$HOME/btvla}"                 # project root: where the repository is uploaded
export SRC_DIR="$HOME/src" VENV_DIR="$HOME/venvs" LOG_DIR="$HOME/setup_logs" ROS_WS="$HOME/ros_ws"
export RC_PY="$VENV_DIR/robocasa/bin/python" G_PY="$VENV_DIR/gr00t/bin/python" ROS_PY="$VENV_DIR/ros/bin/python"
export CKPT_ROOT="$HOME/ckpt/robocasa365_checkpoints"
export CKPT_SUB="gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000"
export CKPT="$CKPT_ROOT/$CKPT_SUB"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export SIM_PORT="${SIM_PORT:-5556}" POLICY_PORT="${POLICY_PORT:-5560}"
export PYTHONPATH="$BTVLA${PYTHONPATH:+:$PYTHONPATH}"

ros_env() {   # ROS setup files reference unset variables: never source them under set -u
  local had_u=0; [[ $- == *u* ]] && had_u=1 && set +u
  source /opt/ros/humble/setup.bash
  [[ -f "$ROS_WS/install/setup.bash" ]] && source "$ROS_WS/install/setup.bash"
  (( had_u )) && set -u
  return 0
}
