#!/bin/bash
#SBATCH --job-name=/home/ma/ma_ma/ma_mpandya/Thesis/wikidata_dataset_acquisition.ipynb
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs//home/ma/ma_ma/ma_mpandya/Thesis/wikidata_dataset_acquisition.ipynb_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs//home/ma/ma_ma/ma_mpandya/Thesis/wikidata_dataset_acquisition.ipynb_%j.err
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
jupyter nbconvert --to script /home/ma/ma_ma/ma_mpandya/Thesis/wikidata_dataset_acquisition.ipynb.ipynb
[ -f /home/ma/ma_ma/ma_mpandya/Thesis/wikidata_dataset_acquisition.ipynb.txt ] && mv /home/ma/ma_ma/ma_mpandya/Thesis/wikidata_dataset_acquisition.ipynb.txt /home/ma/ma_ma/ma_mpandya/Thesis/wikidata_dataset_acquisition.ipynb.py

python /home/ma/ma_ma/ma_mpandya/Thesis/wikidata_dataset_acquisition.ipynb.py

echo "Job complete!"