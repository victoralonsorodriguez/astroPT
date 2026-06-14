#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=AstroPT_Audit
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=16       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=01:00:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_audit_%j.out
#SBATCH --error=logs/astropt_audit_%j.err

echo "--------------------------------------------------------"
echo "AstroPT Embedding-Based Artifact Detection (SLURM)"
echo "--------------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "--------------------------------------------------------"

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

cd "$REPO_ROOT" || { echo "[ERROR] Failed to cd to $REPO_ROOT"; exit 1; }

# Environment Setup
export OMP_NUM_THREADS=16
export PYTHONWARNINGS="ignore"
export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"

# Enable Virtual Environment
VENV_PATH="$REPO_ROOT/.venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Using python environment: $(which python)"
else
    echo "[WARNING] Virtual environment not found at $VENV_PATH. Using system python."
fi

# Activating LaTeX (Required for high-quality astrophysical plots & reports)
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
export MPLCONFIGDIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/matplotlib"
export XDG_CACHE_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache"

# Default Paths
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1_FILTERED.fits"
N_CANDIDATES=1000
N_PLOT=30
BASE_MODALITY="EuclidImage"
WEIGHTS_DIR=""
EMB_DIR=""
SAVE_DIR=""

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":w:e:s:d:m:n:p:b:" opt; do
  case $opt in
    w) WEIGHTS_DIR="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    d) DATA_DIR="$OPTARG" ;;
    m) META_PATH="$OPTARG" ;;
    n) N_CANDIDATES="$OPTARG" ;;
    p) N_PLOT="$OPTARG" ;;
    b) BASE_MODALITY="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

if [ -z "$WEIGHTS_DIR" ] || [ -z "$EMB_DIR" ]; then
  echo "[ERROR]: WEIGHTS_DIR (-w) and EMB_DIR (-e) are required"
  echo "Usage: $0 -w <weights_dir> -e <embeddings_dir> [-s save_dir] [-d data_dir] [-m metadata_path] [-n n_candidates] [-p n_plot] [-b base_modality]"
  exit 1
fi

# Resolve Checkpoint Path from Weights (file or directory)
WEIGHTS_DIR=$(readlink -f "$WEIGHTS_DIR")
if [ -f "$WEIGHTS_DIR" ]; then
    CKPT_PATH="$WEIGHTS_DIR"
elif [ -d "$WEIGHTS_DIR" ]; then
    if [ -f "$WEIGHTS_DIR/ckpt_best.pt" ]; then
        CKPT_PATH="$WEIGHTS_DIR/ckpt_best.pt"
    else
        FIRST_PT=$(find "$WEIGHTS_DIR" -maxdepth 1 -name "*.pt" | head -n 1)
        if [ -n "$FIRST_PT" ]; then
            CKPT_PATH="$FIRST_PT"
        else
            echo "[ERROR]: No checkpoint (.pt) found in weights directory $WEIGHTS_DIR"
            exit 1
        fi
    fi
else
    echo "[ERROR]: WEIGHTS_DIR ($WEIGHTS_DIR) does not exist."
    exit 1
fi

DATA_DIR=$(readlink -f "$DATA_DIR")
META_PATH=$(readlink -f "$META_PATH")

#--- EMBEDDING DETECTION LOGIC ---#
EMB_DIR=$(readlink -f "$EMB_DIR")
if [ -f "$EMB_DIR/EuclidImage.npy" ] || [ -f "$EMB_DIR/EuclidImage_phase1.npy" ] || [ -f "$EMB_DIR/ids.npy" ]; then
    DETECTED_EMB="$EMB_DIR"
else
    SUBDIR=$(ls -td "${EMB_DIR}"/*/ 2>/dev/null | head -n 1)
    if [ -n "$SUBDIR" ]; then
        DETECTED_EMB="${SUBDIR%/}"
    else
        DETECTED_EMB=""
    fi
fi

if [ -n "$DETECTED_EMB" ]; then
    DETECTED_EMB=$(readlink -f "$DETECTED_EMB")
fi

if [ -z "$DETECTED_EMB" ]; then
    echo "[ERROR]: No valid embedding files (.npy) found in $EMB_DIR or its subdirectories."
    exit 1
fi

OUTPUT_ARG=""
if [ -n "$SAVE_DIR" ]; then
    SAVE_DIR=$(readlink -f "$SAVE_DIR")
    OUTPUT_ARG="--output_dir $SAVE_DIR"
fi

echo "Artifact Detection Configuration:"
echo "    CHECKPOINT:     $CKPT_PATH"
echo "    EMB DIR:        $DETECTED_EMB"
echo "    DATASET DIR:    $DATA_DIR"
echo "    CATALOG PATH:   $META_PATH"
echo "    N CANDIDATES:   $N_CANDIDATES"
echo "    N PLOT:         $N_PLOT"
echo "    BASE MODALITY:  $BASE_MODALITY"
if [ -n "$SAVE_DIR" ]; then
    echo "    SAVE DIR:       $SAVE_DIR"
fi
echo "--------------------------------------------------------"

# Run Python Artifact Detector
python3 "$REPO_ROOT/scripts/analysis/anomalies/artefacts_analysis.py" \
    --embeddings_dir "$DETECTED_EMB" \
    --ckpt_path "$CKPT_PATH" \
    --data_dir "$DATA_DIR" \
    --metadata_path "$META_PATH" \
    --n_candidates "$N_CANDIDATES" \
    --n_plot "$N_PLOT" \
    --base_modality "$BASE_MODALITY" \
    $OUTPUT_ARG

echo "--------------------------------------------------------"
echo "AstroPT Artifact Detection Finished Successfully"
echo "--------------------------------------------------------"
