#!/bin/bash
#SBATCH --job-name=18_19_leading_embedders_heavy_july
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/18_19_leading_embedders_heavy_july_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/18_19_leading_embedders_heavy_july_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# NOTE: 18_baseline_e5_mistral and 19_baseline_nvembed_v2 load 7B/8B-parameter
# models -- much heavier than every other baseline in this project (previous
# largest was BGE-large at 335M params). Requested 4 hours as a safety margin;
# actual runtime is hard to predict precisely at this scale. If the
# "gpu_a100_short" partition enforces a walltime cap shorter than this request,
# the job may be rejected at submit time -- check with `sbatch run3.sh` and, if
# it errors on the time limit, ask about a non-"_short" GPU partition.

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
print('VRAM:', torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 'NONE')
"

# Convert notebooks to scripts and run both -- E5-Mistral-7B and NV-Embed-v2 (8B)
FAILED=""
for nb in 18_baseline_e5_mistral 19_baseline_nvembed_v2; do
    jupyter nbconvert --to script "${nb}.ipynb"
    if [ -f "${nb}.txt" ]; then
        mv "${nb}.txt" "${nb}.py"
    fi
    echo "=== Running ${nb}.py ==="
    python "${nb}.py"
    if [ $? -ne 0 ]; then
        echo "!!! ${nb}.py FAILED (non-zero exit code) !!!"
        FAILED="${FAILED} ${nb}"
    fi
done

if [ -n "$FAILED" ]; then
    echo "Job finished WITH FAILURES:${FAILED}"
    exit 1
else
    echo "Job complete! All notebooks succeeded."
fi
