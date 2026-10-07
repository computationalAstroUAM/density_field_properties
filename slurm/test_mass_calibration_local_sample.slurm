#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=mass_cal_local
#SBATCH --error=output/local_sample_mass_calibration/mass_cal_local.log
#SBATCH --output=output/local_sample_mass_calibration/mass_cal_local.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=30-00:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

mkdir -p output/local_sample_mass_calibration

srun python scripts/test_mass_calibration_local_sample.py \
    --config config/haloscope_run_local_sample_mass_calibration.json
