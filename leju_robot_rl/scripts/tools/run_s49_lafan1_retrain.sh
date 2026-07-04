#!/usr/bin/env bash
# =============================================================================
# S49 舞蹈 RL 重训入口（v8 奖励 + S54 原生 CSV）
# =============================================================================
#
# 用法（在 leju_robot_rl 仓库根目录）:
#   conda activate isaaclab
#   bash scripts/tools/run_s49_lafan1_retrain.sh
#
# 可选环境变量:
#   NUM_ENVS=4096          并行环境数（8GB 显存建议 4096，勿用 8192）
#   KUAVO_S49_SRC=...      S49 模型路径，见 setup_s49_training_assets.sh
#
# 训练产物:
#   logs/rsl_rl/Kuavo/s49/dance/<timestamp>/
#     params/env.yaml      完整环境配置快照
#     model_*.pt             检查点
#     RUN_CONFIG.md          自动生成（CSV + 奖励版本摘要，见 train.py）
#
# 脚本名保留 lafan1 是历史原因；当前已改回 S54 CSV，不再跑 LAFAN1/hybrid 转换。
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# -----------------------------------------------------------------------------
# [0] Isaac Sim / Isaac Lab 无头训练环境（必须在 train.py 之前 source）
#
# 具体变量不在本脚本重复写，统一放在 scripts/tools/isaacsim_env.sh，内容与
# 手动训练时一致，包括:
#
#   unset DISPLAY
#   source ~/.local/share/ov/pkg/isaac-sim-4.2.0/setup_python_env.sh
#   export EXP_PATH=$HOME/IsaacLab/source/apps/isaaclab.python.headless.kit
#   export CARB_APP_PATH=$HOME/.local/share/ov/pkg/isaac-sim-4.2.0/kit
#   export ISAAC_PATH=$HOME/.local/share/ov/pkg/isaac-sim-4.2.0
#   export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
#   export WANDB_MODE=offline
#
# play 脚本 run_s49_play.sh 也 source 同一文件，保证 train/play 环境一致。
# -----------------------------------------------------------------------------
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/tools/isaacsim_env.sh"

# 8GB GPU 默认 4096；可在命令前覆盖，例如 NUM_ENVS=2048 bash ...
NUM_ENVS="${NUM_ENVS:-4096}"

# -----------------------------------------------------------------------------
# [1/2] 准备 S49 训练资产
#   - 链接 biped_s49 URDF/meshes 到 ext_template 资产目录
#   - 生成 26-DoF RL 训练 URDF
#   - 确保 S54 舞蹈 CSV（kuavo_action_S49_FROM_S54_INPLACE_RAD.csv）就位
# -----------------------------------------------------------------------------
echo "==> [1/2] S49 training assets (includes S54 CSV)"
bash scripts/tools/setup_s49_training_assets.sh

# -----------------------------------------------------------------------------
# [2/2] 清理 USD 缓存并启动训练
#   - /tmp/IsaacLab/usd_* 残留可能导致加载旧场景或 OOM，开训前删掉
#   - punch_env_cfg.py 中 DANCE_CSV / v6 奖励权重在此生效
#   - 训练结束或中断后 train.py 会自动写 RUN_CONFIG.md
# -----------------------------------------------------------------------------
echo "==> [2/2] Clean stale USD cache + start training"
rm -rf /tmp/IsaacLab/usd_*

echo "Training CSV : kuavo_action_S49_FROM_S54_INPLACE_RAD.csv"
echo "num_envs       : ${NUM_ENVS}"
echo "Reward profile : v8（v7 防漂移 + 强 smoothness / 宽 mimic std / 抑腿抖）"
echo "Log directory  : logs/rsl_rl/Kuavo/s49/dance/<timestamp>/"

python3 scripts/rsl_rl/train.py \
  --task Legged-Isaac-Velocity-Flat-Kuavo-S49-Punch-v0 \
  --num_envs "${NUM_ENVS}" \
  --headless
