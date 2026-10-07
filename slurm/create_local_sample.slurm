#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=local_sample
#SBATCH --error=output/local_sample/local_sample.log
#SBATCH --output=output/local_sample/local_sample.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=30-00:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

# Run all tasks:
#   srun python scripts/generate_local_spatial_sample.py
#
# Run a single task (example: FoF only):
#   srun python scripts/generate_local_spatial_sample.py --tasks fastpm_fof
#
# Available tasks:
#   unitsim_dm unitsim_rockstar fastpm_nbody fastpm_pm fastpm_fof fastpm_snap

srun python scripts/generate_local_spatial_sample.py "$@"

