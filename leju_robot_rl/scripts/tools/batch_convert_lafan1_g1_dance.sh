#!/usr/bin/env bash
# Convert all LAFAN1 G1 dance CSVs to Kuavo S49 training references.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAFAN1_DIR="${LAFAN1_DIR:-/home/lwy/kuavo_all/LAFAN1_Retargeting_Dataset/g1}"
OUT_DIR="${REPO_ROOT}/motion_refs/lafan1_g1"
CLIP_SEC="${CLIP_SEC:-60}"
SRC_FPS="${SRC_FPS:-30}"
DST_FPS="${DST_FPS:-50}"
PROFILE="${PROFILE:-kuavo_dance}"

mkdir -p "${OUT_DIR}"

shopt -s nullglob
files=("${LAFAN1_DIR}"/dance*.csv)
if ((${#files[@]} == 0)); then
  echo "No dance*.csv found under ${LAFAN1_DIR}" >&2
  exit 1
fi

for src in "${files[@]}"; do
  base="$(basename "${src}" .csv)"
  out="${OUT_DIR}/${base}_DANCE_RAD.csv"
  echo "==> ${base}"
  python3 "${REPO_ROOT}/scripts/tools/adapt_lafan1_g1_to_kuavo.py" \
    --input "${src}" \
    --output "${out}" \
    --start-sec 0 \
    --end-sec "${CLIP_SEC}" \
    --src-fps "${SRC_FPS}" \
    --dst-fps "${DST_FPS}" \
    --profile "${PROFILE}"
done

primary="${OUT_DIR}/dance1_subject1_DANCE_RAD.csv"
train_link="${REPO_ROOT}/kuavo_action_LAFAN1_g1_dance1_DANCE_RAD.csv"
cp -f "${primary}" "${train_link}"
echo "Primary training CSV: ${train_link}"
