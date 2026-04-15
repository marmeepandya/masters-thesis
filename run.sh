#!/bin/bash
#SBATCH --job-name=4_embedding_comparison
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/4_embedding_comparison_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/4_embedding_comparison_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

module load devel/cuda/12.8

source thesis/bin/activate
cd /home/ma/ma_ma/ma_mpandya/Thesis

# Verify GPU is visible BEFORE running the notebook
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

jupyter nbconvert --to script 4_embedding_comparison.ipynb
python 4_embedding_comparison.py