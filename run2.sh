#!/bin/bash
#SBATCH --job-name=52_encode_sfr_mistral_scaled
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/52_encode_sfr_mistral_scaled_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/52_encode_sfr_mistral_scaled_%j.err
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

# Convert notebook to script and run
# This notebook's metadata may be missing language_info.file_extension, in which
# case nbconvert falls back to writing .txt instead of .py -- rename if needed
jupyter nbconvert --to script 52_encode_sfr_mistral_scaled.ipynb
[ -f 52_encode_sfr_mistral_scaled.txt ] && mv 52_encode_sfr_mistral_scaled.txt 52_encode_sfr_mistral_scaled.py

python 52_encode_sfr_mistral_scaled.py

echo "Job complete!"