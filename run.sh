#!/bin/bash
#SBATCH --job-name=13_bge_prefix_july
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/Thesis/logs/13_bge_prefix_july_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/Thesis/logs/13_bge_prefix_july_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

# Setup 
cd /home/ma/ma_ma/ma_mpandya/Thesis
source thesis/bin/activate
mkdir -p logs

# Load modules 
module load cs/ollama/0.5.11
module load devel/cuda/12.8

# Start Ollama ONCE on a free port 
export OLLAMA_HOST=127.0.0.1:11435
ollama serve &
OLLAMA_PID=$!
echo "Ollama started with PID $OLLAMA_PID"
sleep 20  # give it time to fully start

# Confirm Ollama is running and models are available 
ollama ps
ollama list

# Verify GPU 
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"

# Add OLLAMA_HOST to .env so the notebook picks it up  
# The notebook reads OLLAMA_BASE_URL from .env
echo "OLLAMA_BASE_URL=http://127.0.0.1:11435" >> .env

# Convert notebook to script and run
# This notebook's metadata is missing language_info.file_extension, so
# nbconvert falls back to writing .txt here (unlike the other notebooks,
# which write .py directly) -- rename before running
jupyter nbconvert --to script 13_bge_prefix.ipynb
mv 13_bge_prefix.txt 13_bge_prefix.py

python 13_bge_prefix.py

# Cleanup 
# Remove the OLLAMA_BASE_URL line we added to .env (keep .env clean)
sed -i '/OLLAMA_BASE_URL=http:\/\/127.0.0.1:11435/d' .env

# Kill Ollama when done
kill $OLLAMA_PID
echo "Job complete!"