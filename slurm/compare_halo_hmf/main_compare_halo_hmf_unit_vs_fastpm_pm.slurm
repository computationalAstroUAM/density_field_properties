#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=hmf_unit_pm
#SBATCH --error=output/halo_hmf_unit_hlist_vs_fastpm_pm/hmf_unit_pm.log
#SBATCH --output=output/halo_hmf_unit_hlist_vs_fastpm_pm/hmf_unit_pm.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=16:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

OUTPUT_DIR="output/halo_hmf_unit_hlist_vs_fastpm_pm"
UNIT_HLIST="/data21/UNITSIM/fixedAmp_InvPhase_001/ROCKSTAR/outputs/hlists/hlist_1.00000.list.bz2"
FASTPM_ROCKSTAR_PM="/data21/users/mruiz/fastpm_MN5/fastpm_tfm/rockstar_out_pm/out_8.list"
MIN_M200B_TIMES_MP=20

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] HMF: UNIT hlist vs FastPM Rockstar PM (M > ${MIN_M200B_TIMES_MP} mp, bins aligned to PM)"

srun python scripts/compare_halo_hmf.py \
    --unit-list "${UNIT_HLIST}" \
    --fastpm-rockstar-pm "${FASTPM_ROCKSTAR_PM}" \
    --catalogs unit fastpm_rockstar_pm \
    --align-log-mass-range-to fastpm_rockstar_pm \
    --min-m200b-times-mp "${MIN_M200B_TIMES_MP}" \
    --output-dir "${OUTPUT_DIR}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Wrote ${OUTPUT_DIR}/hmf_comparison.png"
