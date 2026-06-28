#!/usr/bin/env bash
# S49 LAFAN1 dance retrain: convert CSVs + train (防跪 v2 + 防勾脚 v3).
# Run inside conda env with Isaac Sim / Isaac Lab available.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_ENVS="${NUM_ENVS:-4096}"

echo "==> [1/3] S49 training assets"
bash scripts/tools/setup_s49_training_assets.sh

echo "==> [2/3] LAFAN1 G1 dance CSV batch convert (60s clip, 50Hz, fullbody_inplace)"
bash scripts/tools/batch_convert_lafan1_g1_dance.sh

echo "==> [3/3] Clean stale USD cache + start training"
rm -rf /tmp/IsaacLab/usd_*

echo "Training CSV : kuavo_action_LAFAN1_g1_dance1_INPLACE_RAD.csv"
echo "num_envs       : ${NUM_ENVS}"
echo "Reward profile : 防跪 v2 + penalty_foot_pitch v3"

python3 scripts/rsl_rl/train.py \
  --task Legged-Isaac-Velocity-Flat-Kuavo-S49-Punch-v0 \
  --num_envs "${NUM_ENVS}" \
  --headless
