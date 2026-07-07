#!/bin/bash
#SBATCH --job-name=15_baseline_bge_gemma2_long
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/15_baseline_bge_gemma2_long_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/15_baseline_bge_gemma2_long_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# NOTE: this uses gpu_a100_il (MaxTime 2 days) instead of gpu_a100_short (30 min)
# so the 9B-param model only has to load once instead of every 30 min.
# TIME_BUDGET_MINUTES in the notebook is set to 1380 (23h) to match this partition.
# IMPORTANT: before this job actually starts running (it may sit queued for a while
# since gpu_a100_il currently has 0 idle nodes), stop the CPU .ipynb kernel that is
# resuming from the same checkpoint files -- both must not run at the same time.

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
jupyter nbconvert --to script 15_baseline_bge_gemma2.ipynb
if [ -f "15_baseline_bge_gemma2.txt" ]; then
    mv 15_baseline_bge_gemma2.txt 15_baseline_bge_gemma2.py
fi

python 15_baseline_bge_gemma2.py

echo "Job complete!"
