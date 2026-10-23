#!/bin/bash
#SBATCH --account=tuttle-rmss
#SBATCH --partition=standard
#SBATCH --job-name=pca_pc3_review
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm/pca_pc3_review_%j.out
#SBATCH --error=slurm/pca_pc3_review_%j.err

set -eo pipefail

cd /users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/image_classification/metric_dev

module load miniforge/python-3.12.10/25.3.0

python scripts/review_saved_pca.py \
    --source results/archie_8193723_complete \
    --components 3
