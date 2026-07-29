#!/bin/bash
#SBATCH --job-name=60_scaling_evaluation
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/60_scaling_evaluation_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/60_scaling_evaluation_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# Setup
cd /home/ma/ma_ma/ma_mpandya/Thesis
source thesis/bin/activate
mkdir -p logs

# Run notebook 59 first -- this notebook reads its bm25_results_all_tiers.json output and will
# fail with a FileNotFoundError if that hasn't been produced yet.

# Load modules
module load devel/cuda/12.8

# Verify GPU
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"

# Convert notebook to script and run
# This notebook's metadata may be missing language_info.file_extension, in which
# case nbconvert falls back to writing .txt instead of .py -- rename if needed
jupyter nbconvert --to script 60_scaling_evaluation.ipynb
[ -f 60_scaling_evaluation.txt ] && mv 60_scaling_evaluation.txt 60_scaling_evaluation.py

python 60_scaling_evaluation.py

echo "Job complete!"
