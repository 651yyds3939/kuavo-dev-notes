#!/usr/bin/env bash
# S49 LAFAN1 dance retrain: hybrid CSV + v5 rewards + train.
# Run inside conda env with Isaac Sim / Isaac Lab available.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Isaac Sim headless / GPU / offline (must run before train.py)
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/tools/isaacsim_env.sh"

NUM_ENVS="${NUM_ENVS:-4096}"

echo "==> [1/4] S49 training assets"
bash scripts/tools/setup_s49_training_assets.sh

echo "==> [2/4] LAFAN1 G1 dance CSV batch convert"
bash scripts/tools/batch_convert_lafan1_g1_dance.sh

echo "==> [3/4] Hybrid CSV: LAFAN1 legs + S54 native arms"
python3 scripts/tools/merge_lafan1_legs_s54_arms.py

echo "==> [4/4] Clean stale USD cache + start training"
rm -rf /tmp/IsaacLab/usd_*

echo "Training CSV : kuavo_action_HYBRID_LAFAN1LEGS_S54ARMS_RAD.csv"
echo "num_envs       : ${NUM_ENVS}"
echo "Reward profile : 舞感 v5（关 arm_roll / 轻稳定 / 强 mimic）"

python3 scripts/rsl_rl/train.py \
  --task Legged-Isaac-Velocity-Flat-Kuavo-S49-Punch-v0 \
  --num_envs "${NUM_ENVS}" \
  --headless
