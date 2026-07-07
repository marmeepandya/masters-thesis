#!/bin/bash
#SBATCH --job-name=23_hybrid_minilm_linq_reranker
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/23_hybrid_minilm_linq_reranker_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/23_hybrid_minilm_linq_reranker_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# NOTE: this only reranks the top-50 candidates per query with a cross-encoder
# (BAAI/bge-reranker-v2-m3) -- no corpus re-encoding, RRF fusion just reads
# already-cached CSVs from notebooks 03 and 20. Same 30-min partition notebook
# 07 already completed successfully in.

# Setup
cd /home/ma/ma_ma/ma_mpandya/Thesis
source thesis/bin/activate
mkdir -p logs

# Load modules
module load devel/cuda/12.8

# Verify GPU
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"

# Convert notebook to script and run
jupyter nbconvert --to script 23_hybrid_minilm_linq_reranker.ipynb
if [ -f "23_hybrid_minilm_linq_reranker.txt" ]; then
    mv 23_hybrid_minilm_linq_reranker.txt 23_hybrid_minilm_linq_reranker.py
fi

python 23_hybrid_minilm_linq_reranker.py

echo "Job complete!"
