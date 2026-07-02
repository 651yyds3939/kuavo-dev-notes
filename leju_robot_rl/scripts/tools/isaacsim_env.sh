#!/usr/bin/env bash
# Source before train.py / play.py inside conda isaaclab env.
# Usage: source scripts/tools/isaacsim_env.sh

unset DISPLAY
# shellcheck source=/dev/null
source "${HOME}/.local/share/ov/pkg/isaac-sim-4.2.0/setup_python_env.sh"
export EXP_PATH="${HOME}/IsaacLab/source/apps/isaaclab.python.headless.kit"
export CARB_APP_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.2.0/kit"
export ISAAC_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.2.0"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export WANDB_MODE=offline
