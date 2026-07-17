#!/bin/bash
#SBATCH --job-name=27_baseline_colbert
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/27_baseline_colbert_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/27_baseline_colbert_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# NOTE: ColBERTv2 is bert-base sized (110M params) -- encoding all 98,716
# companies at token level should take a few minutes (same order as
# GTE-large's 2 min), and brute-force MaxSim scoring against the full corpus
# for all 101 queries is estimated at 1-3 minutes of GPU compute on top of
# that. Well within this 30-min partition.

set -e
cd /home/ma/ma_ma/ma_mpandya/Thesis
source thesis/bin/activate

module load devel/cuda/12.8

python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"

jupyter nbconvert --to script 27_baseline_colbert.ipynb
if [ -f "27_baseline_colbert.txt" ]; then
    mv 27_baseline_colbert.txt 27_baseline_colbert.py
fi

python 27_baseline_colbert.py

echo "Job complete!"
