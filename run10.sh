#!/bin/bash
#SBATCH --job-name=24_hybrid_minilm_linq_full_rerank
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/24_hybrid_minilm_linq_full_rerank_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/24_hybrid_minilm_linq_full_rerank_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# NOTE: reranks the full ~1,300-candidate union pool per query (no RRF
# pre-truncation), not just top-50 like notebook 23 -- estimated ~9 minutes
# of GPU compute for all 101 queries, well within this 30-min partition.

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
jupyter nbconvert --to script 24_hybrid_minilm_linq_full_rerank.ipynb
if [ -f "24_hybrid_minilm_linq_full_rerank.txt" ]; then
    mv 24_hybrid_minilm_linq_full_rerank.txt 24_hybrid_minilm_linq_full_rerank.py
fi

python 24_hybrid_minilm_linq_full_rerank.py

echo "Job complete!"
