#!/bin/bash
#SBATCH --job-name=59_bm25_tiered_index
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/59_bm25_tiered_index_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/59_bm25_tiered_index_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# Setup
cd /home/ma/ma_ma/ma_mpandya/Thesis
source thesis/bin/activate
mkdir -p logs

# No GPU work here (BM25 tokenizing/indexing is CPU-only), but still submitted through the
# GPU partition since that's what's available -- the GPU itself just sits idle for this one.

# Convert notebook to script and run
# This notebook's metadata may be missing language_info.file_extension, in which
# case nbconvert falls back to writing .txt instead of .py -- rename if needed
jupyter nbconvert --to script 59_bm25_tiered_index.ipynb
[ -f 59_bm25_tiered_index.txt ] && mv 59_bm25_tiered_index.txt 59_bm25_tiered_index.py

python 59_bm25_tiered_index.py

echo "Job complete!"
