#!/usr/bin/env bash
# 将 S49 URDF/meshes 链接到 Isaac 训练资产目录，并生成 26-DoF RL 训练 URDF
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KUAVO_ALL_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

SRC_DEFAULT="${KUAVO_ALL_ROOT}/kuavo-ros-opensource/src/kuavo_assets/models/biped_s49"
SRC="${KUAVO_S49_SRC:-${SRC_DEFAULT}}"
MESH_LINK="${REPO_ROOT}/exts/ext_template/ext_template/assets/Robots/Kuavo/biped_s49"
TRAIN_URDF_DIR="${REPO_ROOT}/exts/ext_template/ext_template/assets/Robots/Kuavo/s49_train/urdf"

if [[ ! -d "${SRC}/urdf" ]]; then
  echo "ERROR: S49 model not found at ${SRC}" >&2
  echo "Set KUAVO_S49_SRC to kuavo-ros-opensource .../models/biped_s49" >&2
  exit 1
fi

mkdir -p "$(dirname "${MESH_LINK}")"
if [[ ! -e "${MESH_LINK}" ]]; then
  ln -sfn "$(realpath "${SRC}")" "${MESH_LINK}"
  echo "Linked ${MESH_LINK} -> $(realpath "${SRC}")"
fi

mkdir -p "${TRAIN_URDF_DIR}"
SRC_URDF="${SRC}/urdf/biped_s49.urdf"
OUT_RL_URDF="${TRAIN_URDF_DIR}/biped_s49_rl.urdf"
OUT_26DOF_URDF="${TRAIN_URDF_DIR}/biped_s49_26dof.urdf"

python3 - <<'PY' "${SRC_URDF}" "${OUT_RL_URDF}"
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text(encoding="utf-8")
text = text.replace(
    "package://kuavo_assets/models/biped_s49/meshes/",
    "../../biped_s49/meshes/",
)
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(text, encoding="utf-8")
print(f"Wrote {dst}")
PY

python3 "${SCRIPT_DIR}/make_s49_26dof_urdf.py" "${OUT_RL_URDF}" "${OUT_26DOF_URDF}"
OUT_LITE_URDF="${TRAIN_URDF_DIR}/biped_s49_26dof_lite.urdf"
python3 "${SCRIPT_DIR}/make_s49_lite_urdf.py" "${OUT_26DOF_URDF}" "${OUT_LITE_URDF}"

# 清理误写入 ros 仓库的损坏文件（若存在）
BAD_URDF="${SRC}/urdf/biped_s49_26dof.urdf"
if [[ -f "${BAD_URDF}" ]]; then
  rm -f "${BAD_URDF}"
  echo "Removed stale ${BAD_URDF}"
fi

echo "Done. Training asset ready: ${OUT_LITE_URDF}"
echo "Optional: export KUAVO_S49_URDF=${OUT_LITE_URDF}"
