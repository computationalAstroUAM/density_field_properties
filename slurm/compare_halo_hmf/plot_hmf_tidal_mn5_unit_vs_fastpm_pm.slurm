#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=plot_hmf_mn5
#SBATCH --error=output/hmf_plot/error_plot_hmf_mn5.log
#SBATCH --output=output/hmf_plot/output_plot_hmf_mn5.log
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=4:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

OUTPUT_DIR="output/hmf_plot"
UNIT_HLIST="output/tidal_mn5/unit/hlist_1.00000.list"
FASTPM_ROCKSTAR_PM="/home/arnes/santiago_arranz/density_field_properties/output/rockstar_find_parents/fastpm_mn5_tfm/out_8_with_pid.list"
BOX_SIZE_MPC_H=1000
MIN_PARTICLES=20
mkdir -p "${OUTPUT_DIR}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] plot_hmf: UNIT hlist vs FastPM Rockstar PM (tidal_anisotropy_mn5 paths)"

srun python scripts/plot_hmf_not_cumulative.py \
    --output-dir "${OUTPUT_DIR}" \
    --halo-catalog-unit "${UNIT_HLIST}" \
    --halo-catalog-fastpm "${FASTPM_ROCKSTAR_PM}" \
    --box-size-mpc-h "${BOX_SIZE_MPC_H}" \
    --min-particles "${MIN_PARTICLES}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Wrote ${OUTPUT_DIR}/halo_mass_function.png"
