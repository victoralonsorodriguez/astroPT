#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=AION_Extract
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=16       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=04:00:00          

#--- LOGS FILES ---#
#SBATCH --output=logs/aion_extract_%j.out
#SBATCH --error=logs/aion_extract_%j.err

set -euo pipefail

# Robust repository root detection based on script location
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# If SLURM has moved us to /var/spool, correct REPO_ROOT to the launch working directory
if [[ "$REPO_ROOT" == "/var/spool"* ]]; then
    if [ -d "$PWD/astroPT" ]; then
        REPO_ROOT="$PWD/astroPT"
    else
        REPO_ROOT="$PWD"
    fi
fi

PYTHON_SCRIPT="$REPO_ROOT/scripts/baselines/baseline_aion_embeddings.py"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated_tokenized"
SAVE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/AION_freeze/embeddings/aion_embeddings"
RESNET_WEIGHTS="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/Euclid2AION_resnet_adapter_weights/adapters_final.pt"
BATCH_SIZE=64
NUM_WORKERS=16

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":a:s:w:b:n:h" opt; do
  case $opt in
    a) DATA_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    w) RESNET_WEIGHTS="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    n) NUM_WORKERS="$OPTARG" ;;
    h) echo "Usage: $0 [-a data_dir] [-s save_dir] [-w resnet_weights] [-b batch_size] [-n num_workers]"; exit 0 ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

# Activate Environment
if [ -d "$REPO_ROOT/.venv" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
elif [ -d "$REPO_ROOT/venv" ]; then
    source "$REPO_ROOT/venv/bin/activate"
fi

# Configure Hugging Face Cache and Offline Mode
export UV_CACHE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/uv_cache"
export HF_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_OFFLINE=1

# Change directory
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")
echo "-----------------------------------------------"
echo "AION Embedding Extraction Job ${SLURM_JOB_ID:-LOCAL} - $NOW"
echo "Data Dir: $DATA_DIR"
echo "Save Dir: $SAVE_DIR"
echo "-----------------------------------------------"

# Run AION embedding extraction script
python3 "$PYTHON_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --resnet_weights_path "$RESNET_WEIGHTS" \
    --batch_size "$BATCH_SIZE" \
    --device "cuda" \
    --num_workers "$NUM_WORKERS"

echo "-----------------------------------------------"
echo "AION Embedding Extraction Finished"
echo "-----------------------------------------------"
