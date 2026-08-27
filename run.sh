#!/bin/bash
#SBATCH --job-name=67_production_independent_evaluation
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/67_production_independent_evaluation_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/67_production_independent_evaluation_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# Unlike notebook 66 (pure API calls, runs fine on dev_cpu), this notebook loads real models on GPU (MiniLM, GTE-large, and the 7B-parameter Linq-Embed-Mistral) to encode the 19 queries, so it genuinely needs the GPU allocation and the memory that comes with a full gpu_a100_short node; running this on dev_cpu's ~3.9GB default memory OOM-killed it partway through loading Linq-Embed-Mistral.

# Setup
cd /home/ma/ma_ma/ma_mpandya/Thesis
source thesis/bin/activate
mkdir -p logs

# Load modules
module load devel/cuda/12.8

# Convert notebook to script and run
# This notebook's metadata may be missing language_info.file_extension, in which
# case nbconvert falls back to writing .txt instead of .py -- rename if needed
jupyter nbconvert --to script 67_production_independent_evaluation.ipynb
[ -f 67_production_independent_evaluation.txt ] && mv 67_production_independent_evaluation.txt 67_production_independent_evaluation.py

python 67_production_independent_evaluation.py

echo "Job complete!"