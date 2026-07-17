#!/bin/bash
#SBATCH --job-name=25_hybrid_triple_full_rerank
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/25_hybrid_triple_full_rerank_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/25_hybrid_triple_full_rerank_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# NOTE: 3-way union pool (MiniLM + Linq-Embed-Mistral + GTE-large) is bigger
# than notebook 24's 2-way pool, so rerank time will be somewhat higher than
# the ~9 min notebook 24 took -- still well within this 30-min partition.

set -e
cd /home/ma/ma_ma/ma_mpandya/Thesis
source thesis/bin/activate

module load devel/cuda/12.8

python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"

jupyter nbconvert --to script 25_hybrid_triple_full_rerank.ipynb
if [ -f "25_hybrid_triple_full_rerank.txt" ]; then
    mv 25_hybrid_triple_full_rerank.txt 25_hybrid_triple_full_rerank.py
fi

python 25_hybrid_triple_full_rerank.py

echo "Job complete!"
