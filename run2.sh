#!/bin/bash
#SBATCH --job-name=16_17_leading_embedders_light_july
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/16_17_leading_embedders_light_july_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/16_17_leading_embedders_light_july_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

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

# Convert notebooks to scripts and run both -- BGE-M3 and GTE-large are both
# mid-size (< 1B params), similar cost to the BGE-large baseline already run
FAILED=""
for nb in 16_baseline_bgem3 17_baseline_gte_large; do
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
