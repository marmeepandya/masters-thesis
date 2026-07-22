#!/bin/bash
#SBATCH --job-name=15_baseline_bge_gemma2_short
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/15_baseline_bge_gemma2_short_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/15_baseline_bge_gemma2_short_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# gpu_a100_il had 0 available nodes for 3+ straight days -- switched this 9B-param
# model back to gpu_a100_short (30-min slots) using the same checkpoint/resume
# pattern as every other notebook. Self-resubmits at the end of this script until
# the final company_embeddings.npy exists, so you don't have to manually resubmit
# every 30 minutes -- just submit this once and let it chain itself overnight.
# Cancel the stuck long-partition job first if it's still queued: scancel 5964638

set -e
cd /home/ma/ma_ma/ma_mpandya/Thesis
source thesis/bin/activate

module load devel/cuda/12.8

python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"

jupyter nbconvert --to script 15_baseline_bge_gemma2.ipynb
if [ -f "15_baseline_bge_gemma2.txt" ]; then
    mv 15_baseline_bge_gemma2.txt 15_baseline_bge_gemma2.py
fi

python 15_baseline_bge_gemma2.py

if [ ! -f "result/15_baseline_bge_gemma2/company_embeddings.npy" ]; then
    echo "Not finished yet -- resubmitting run8_short.sh to continue from checkpoint..."
    sbatch run8_short.sh
else
    echo "Job complete! Final embeddings file exists -- no more resubmission needed."
fi
