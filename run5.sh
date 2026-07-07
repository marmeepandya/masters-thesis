#!/bin/bash
#SBATCH --job-name=19_baseline_nvembed_v2_july
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/19_baseline_nvembed_v2_july_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/19_baseline_nvembed_v2_july_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# NOTE: Linq-Embed-Mistral is a 7B-param model -- this notebook checkpoints
# encoding progress every 200 batches and self-exits at 22 min elapsed so it never
# loses work to the 30-min wall time. If the job ends before "Job complete!" is
# printed, just resubmit this same script -- it resumes from the last checkpoint.
# No Ollama needed -- this notebook is pure dense embedding retrieval.

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
jupyter nbconvert --to script 19_baseline_nvembed_v2.ipynb
if [ -f "19_baseline_nvembed_v2.txt" ]; then
    mv 19_baseline_nvembed_v2.txt 19_baseline_nvembed_v2.py
fi

python 19_baseline_nvembed_v2.py

echo "Job complete!"
