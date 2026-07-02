#!/usr/bin/env bash
# S49 checkpoint playback / video recording (headless).
#
# Examples:
#   RUN_DATE=2026-07-02_03-40-57 CHECKPOINT=model_9999.pt bash scripts/tools/run_s49_play.sh
#   RUN_DATE=2026-07-02_03-40-57 CHECKPOINT=model_9999.pt VIDEO_LENGTH=600 NUM_ENVS=256 bash scripts/tools/run_s49_play.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/tools/isaacsim_env.sh"

RUN_DATE="${RUN_DATE:?Set RUN_DATE, e.g. export RUN_DATE=2026-07-02_03-40-57}"
CHECKPOINT="${CHECKPOINT:-model_9999.pt}"
NUM_ENVS="${NUM_ENVS:-256}"
VIDEO_LENGTH="${VIDEO_LENGTH:-600}"
TASK="${TASK:-Legged-Isaac-Velocity-Flat-Kuavo-S49-Play-v0}"

EXTRA_ARGS=()
if [[ "${RECORD_VIDEO:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--video --video_length "${VIDEO_LENGTH}")
fi

echo "Task       : ${TASK}"
echo "load_run   : ${RUN_DATE}"
echo "checkpoint : ${CHECKPOINT}"
echo "num_envs   : ${NUM_ENVS}"

python3 scripts/rsl_rl/play.py \
  --task "${TASK}" \
  --load_run "${RUN_DATE}" \
  --checkpoint "${CHECKPOINT}" \
  --num_envs "${NUM_ENVS}" \
  --headless \
  "${EXTRA_ARGS[@]}" \
  "$@"
