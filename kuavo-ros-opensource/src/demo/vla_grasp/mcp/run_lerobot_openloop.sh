#!/usr/bin/env bash
# LeRobot ACT 真机开环一键启动（NUC 上运行）
#
# 前置（另开终端）:
#   1. WBC  load_kuavo_real.launch
#   2. look_down.py
#   3. kuavo_state_publisher.py
#   4. (Orin) load_robot_head.launch
#
# 用法:
#   conda activate lerobot
#   bash src/demo/vla_grasp/mcp/run_lerobot_openloop.sh
#   bash src/demo/vla_grasp/mcp/run_lerobot_openloop.sh --max-steps 500 --log-csv ~/openloop.csv

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/lab/kuavo-ros-opensource}"
CKPT="${CKPT:-${REPO_ROOT}/src/demo/vla_grasp/pretrained_model}"
FPS="${FPS:-25}"
DEVICE="${DEVICE:-cpu}"

cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
source devel/setup.bash
# 推理节点需要 ROS + workspace 的 PYTHONPATH，勿 unset PYTHONPATH
export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:${PYTHONPATH:-}"

echo "=== LeRobot ACT 开环推理 ==="
echo "checkpoint: ${CKPT}"
echo "device    : ${DEVICE}"
echo "fps       : ${FPS}"
echo ""

python3 "${REPO_ROOT}/src/demo/vla_grasp/mcp/lerobot_act_verify_checkpoint.py" \
  --checkpoint "${CKPT}" \
  --device "${DEVICE}"

echo ""
echo "=== 启动 ROS 推理节点（开环，不执行）==="
exec python "${REPO_ROOT}/src/demo/vla_grasp/mcp/lerobot_act_infer.py" \
  --checkpoint "${CKPT}" \
  --fps "${FPS}" \
  --device "${DEVICE}" \
  --log-csv "${HOME}/lerobot_openloop_$(date +%Y%m%d_%H%M%S).csv" \
  "$@"
