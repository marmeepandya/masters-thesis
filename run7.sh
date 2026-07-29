#!/bin/bash
#SBATCH --job-name=61_fusion_ranker_hyperparameter_tuning
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/61_fusion_ranker_hyperparameter_tuning_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/61_fusion_ranker_hyperparameter_tuning_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# Setup
cd /home/ma/ma_ma/ma_mpandya/Thesis
source thesis/bin/activate
mkdir -p logs

# No GPU work here (classical sklearn models on ~2,300 rows, CPU-only), but still submitted
# through the GPU partition since that's what's available -- the GPU itself just sits idle.
# Whole search (baseline + 40 random combos + best-combo re-eval) is expected to take roughly
# 10-15 minutes based on notebook 56's ~15s-per-101-fold-CV timing, well inside the 30-min budget.
# Resumable via search_leaderboard.json if it ever does need a second submission.

# Convert notebook to script and run
# This notebook's metadata may be missing language_info.file_extension, in which
# case nbconvert falls back to writing .txt instead of .py -- rename if needed
jupyter nbconvert --to script 61_fusion_ranker_hyperparameter_tuning.ipynb
[ -f 61_fusion_ranker_hyperparameter_tuning.txt ] && mv 61_fusion_ranker_hyperparameter_tuning.txt 61_fusion_ranker_hyperparameter_tuning.py

python 61_fusion_ranker_hyperparameter_tuning.py

echo "Job complete!"
