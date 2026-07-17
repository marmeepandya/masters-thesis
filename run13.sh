#!/bin/bash
#SBATCH --job-name=26_baseline_simlm
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/26_baseline_simlm_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/26_baseline_simlm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# NOTE: SimLM is BERT-base (110M params) -- GTE-large (335M) encoded the full
# 98,716-company corpus in ~2 minutes on A100, so this should be comparably
# fast or faster. No checkpoint/resume needed, single-shot job.

set -e
cd /home/ma/ma_ma/ma_mpandya/Thesis
source thesis/bin/activate

module load devel/cuda/12.8

python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"

jupyter nbconvert --to script 26_baseline_simlm.ipynb
if [ -f "26_baseline_simlm.txt" ]; then
    mv 26_baseline_simlm.txt 26_baseline_simlm.py
fi

python 26_baseline_simlm.py

echo "Job complete!"
