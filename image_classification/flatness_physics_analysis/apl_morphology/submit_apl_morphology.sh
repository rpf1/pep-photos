#!/bin/bash
#SBATCH --job-name=apl_morph
#SBATCH --output=Slurm.log
#SBATCH --error=Slurm.err
#SBATCH --account=tuttle-rmss
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16

module load miniforge/python-3.12.10/25.3.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate my_env_2025

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python analyze_apl_morphology_bundle.py \
    primary_matched_pairs.csv \
    --dapl dAPL_selected_molecules.xlsx \
    --workers 16 \
    --exhaustive-pair-plots \
    --out apl_morphology_full_analysis
