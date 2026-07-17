#!/bin/bash
#SBATCH --job-name=28_hybrid_quad_full_rerank
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/28_hybrid_quad_full_rerank_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/28_hybrid_quad_full_rerank_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# NOTE: 4-way union pool (MiniLM + Linq-Embed-Mistral + GTE-large + OpenAI-large)
# is bigger than notebook 25's 3-way pool -- checkpointed every 10 queries with
# atomic-write-with-retry, so if this doesn't finish in 30 min, resubmit this
# same script to resume instead of restarting.

set -e
cd /home/ma/ma_ma/ma_mpandya/Thesis
source thesis/bin/activate

module load devel/cuda/12.8

python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"

jupyter nbconvert --to script 28_hybrid_quad_full_rerank.ipynb
if [ -f "28_hybrid_quad_full_rerank.txt" ]; then
    mv 28_hybrid_quad_full_rerank.txt 28_hybrid_quad_full_rerank.py
fi

python 28_hybrid_quad_full_rerank.py

echo "Job complete!"
