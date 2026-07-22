#!/bin/bash
#SBATCH --job-name=29_hybrid_altreranker
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/29_hybrid_altreranker_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/29_hybrid_altreranker_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# NOTE: same 3-way candidate pool as notebook 25 (MiniLM+Linq+GTE-large),
# only the reranker model changes (mxbai-rerank-large-v1 instead of
# BAAI/bge-reranker-v2-m3) -- isolates whether reranker choice is the
# bottleneck now that adding more channels (notebook 28) stopped helping.

set -e
cd /home/ma/ma_ma/ma_mpandya/Thesis
source thesis/bin/activate

module load devel/cuda/12.8

python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"

jupyter nbconvert --to script 29_hybrid_altreranker.ipynb
if [ -f "29_hybrid_altreranker.txt" ]; then
    mv 29_hybrid_altreranker.txt 29_hybrid_altreranker.py
fi

python 29_hybrid_altreranker.py

echo "Job complete!"
