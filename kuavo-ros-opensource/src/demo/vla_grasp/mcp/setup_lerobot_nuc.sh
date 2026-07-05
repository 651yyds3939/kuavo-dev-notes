#!/usr/bin/env bash
# NUC 下位机：安装 Miniconda + lerobot 推理环境（无 conda 时用此脚本）
#
# 用法:
#   bash /home/lab/kuavo-ros-opensource/src/demo/vla_grasp/mcp/setup_lerobot_nuc.sh
#
# 装完后新开终端，或:
#   source ~/miniconda3/etc/profile.d/conda.sh
#   conda activate lerobot

set -euo pipefail

MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-lerobot}"
PYTHON_VER="${PYTHON_VER:-3.10}"
REPO="/home/lab/kuavo-ros-opensource"
REQ="${REPO}/src/demo/vla_grasp/mcp/lerobot_infer_requirements.txt"

echo "=== NUC lerobot 环境安装 ==="
echo "Miniconda 目录: ${MINICONDA_DIR}"
echo "Python 版本    : ${PYTHON_VER}"
echo ""

if [[ -x "${MINICONDA_DIR}/bin/conda" ]]; then
  echo "Miniconda 已存在，跳过下载。"
else
  INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
  echo "下载 Miniconda ..."
  wget -q --show-progress -O "${INSTALLER}" \
    "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  echo "安装 Miniconda 到 ${MINICONDA_DIR} ..."
  bash "${INSTALLER}" -b -p "${MINICONDA_DIR}"
  rm -f "${INSTALLER}"
fi

# shellcheck disable=SC1091
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "conda 环境 ${ENV_NAME} 已存在，跳过 create。"
else
  echo "创建 conda 环境 ${ENV_NAME} (python=${PYTHON_VER}) ..."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VER}" -y
fi

conda activate "${ENV_NAME}"
echo "Python: $(python --version)"

echo "安装 lerobot 依赖（CPU 版 torch，约 5~10 分钟）..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r "${REQ}"

echo ""
echo "=== 验证 ==="
python "${REPO}/src/demo/vla_grasp/mcp/lerobot_act_verify_checkpoint.py" \
  --checkpoint "${REPO}/src/demo/vla_grasp/pretrained_model" \
  --device cpu

echo ""
echo "=== 安装完成 ==="
echo "每次开环推理前执行:"
echo "  source ${MINICONDA_DIR}/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${REPO} && source devel/setup.bash"
echo "  bash src/demo/vla_grasp/mcp/run_lerobot_openloop.sh"
