#!/usr/bin/env bash
# Full-body in-place dance: regenerate CSV + train (run inside Isaac Sim / Docker).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Regenerating kuavo_action_FULLBODY_INPLACE_RAD.csv"
python3 scripts/tools/adapt_dance_csv.py \
  --profile fullbody_inplace \
  --input kuavo_action_PERFECT_LIMIT_RAD.csv \
  --output kuavo_action_FULLBODY_INPLACE_RAD.csv

echo "==> Starting training (8192 envs)"
python scripts/rsl_rl/train.py \
  --task Legged-Isaac-Velocity-Flat-Kuavo-S42-v0 \
  --num_envs 8192 \
  --headless

echo "Done. Targets: track_punch_arms>=13, track_punch_legs>=7, base_lin_vel_xy_stationary~0"
