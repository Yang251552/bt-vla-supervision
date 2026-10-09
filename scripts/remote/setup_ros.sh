#!/usr/bin/env bash
# ROS 2 Humble from apt (system Python 3.10) + py_trees / py_trees_ros (Humble apt: py_trees 2.5.0 measured
# 2026-10-08, although rosdistro lists 2.6.0-1), the ros venv
# (--system-site-packages; torch-cpu for the GR00T wire format, pyzmq, numpy 1.26.4, pillow) and the
# bt_vla_interfaces package built with colcon in ~/ros_ws. Idempotent, logs to ~/setup_logs/setup_ros.log.
source "$(dirname "$0")/lib.sh"
start_log setup_ros
on_root "$VENV_DIR" "$ROS_WS" "$HOME/.cache"

ros_repo() {   # ROS 2 apt source as in the Humble install docs [U: key/source method may have moved]
  apt_install software-properties-common curl gnupg
  sudo add-apt-repository -y universe
  sudo curl -fsSL --max-time 60 https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
}
step ros_repo ros_repo
step ros_apt apt_install ros-humble-ros-base ros-humble-py-trees ros-humble-py-trees-ros \
  ros-humble-rosidl-default-generators python3-colcon-common-extensions python3-venv
step ros_venv python3 -m venv --system-site-packages "$VENV_DIR/ros"
step ros_torch "$ROS_PY" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
step ros_pip "$ROS_PY" -m pip install numpy==1.26.4 pyzmq pillow

# interfaces: always rebuilt (cheap, picks up .action edits); colcon runs on the system python, venv inactive
mkdir -p "$ROS_WS/src"
ln -sfn "$BTVLA/ros/bt_vla_interfaces" "$ROS_WS/src/bt_vla_interfaces"
( cd "$ROS_WS" && set +u && source /opt/ros/humble/setup.bash && colcon build --packages-select bt_vla_interfaces )

ros_env
"$ROS_PY" -c "import rclpy, py_trees, py_trees_ros, torch, zmq, numpy
from bt_vla_interfaces.action import VlaSkill
from py_trees import version as ptv  # apt py_trees has no top-level __version__
print('ros-venv py_trees', ptv.__version__, 'torch', torch.__version__, 'numpy', numpy.__version__)
assert ptv.__version__.startswith('2.5'), 'local tests pin py_trees 2.5.0 (Humble apt)'" | tee -a "$LOG_DIR/versions.txt"
log "=== setup_ros ok"
