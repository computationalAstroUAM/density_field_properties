#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=find_parents_mn5
#SBATCH --error=output/rockstar_find_parents/find_parents_out8.log
#SBATCH --output=output/rockstar_find_parents/find_parents_out8.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=1-00:00:00

set -euo pipefail

FIND_PARENTS_BIN="/home/adrian/UNIT_PNG/rockstar_old/rockstar/util/find_parents"
INPUT_LIST="/data21/users/mruiz/fastpm_MN5/fastpm_tfm/rockstar_out_pm/out_8.list"
BOX_SIZE_MPC_H="1000"
OUTPUT_DIR="/home/arnes/santiago_arranz/density_field_properties/output/rockstar_find_parents/fastpm_mn5_tfm"
OUTPUT_LIST="${OUTPUT_DIR}/out_8_with_pid.list"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -x "${FIND_PARENTS_BIN}" ]]; then
    echo "[ERROR] find_parents not executable: ${FIND_PARENTS_BIN}"
    exit 1
fi

if [[ ! -f "${INPUT_LIST}" ]]; then
    echo "[ERROR] Input catalog missing: ${INPUT_LIST}"
    exit 1
fi

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Input:  ${INPUT_LIST}"
echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Output: ${OUTPUT_LIST}"
echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Box size (Mpc/h): ${BOX_SIZE_MPC_H}"

srun "${FIND_PARENTS_BIN}" "${INPUT_LIST}" "${BOX_SIZE_MPC_H}" > "${OUTPUT_LIST}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Done. Lines: $(wc -l < "${OUTPUT_LIST}")"
