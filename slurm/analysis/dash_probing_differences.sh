#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Plot_R2_Diff
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=2       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=32G                
#SBATCH --time=00:10:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_r2_diff_%j.out
#SBATCH --error=logs/astropt_r2_diff_%j.err

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

PYTHON_SCRIPT="$REPO_ROOT/scripts/analysis/dashboards/dash_probing_differences.py"

#--- ARGUMENT PARSING (FLAGS) ---#
EMB_DIR=""
SAVE_DIR=""
COMPARE_EMB_DIR=""

while getopts ":r:e:s:c:h" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    c) COMPARE_EMB_DIR="$OPTARG" ;;
    h)
      echo "Usage: $0 -e EMB_DIR [-c COMPARE_EMB_DIR] [-s SAVE_DIR] [-r REPO_ROOT]"
      echo "  -e: Embedding folder of the hybrid model run containing downstream_results.csv"
      echo "  -c: Embedding folder of the comparison run containing downstream_results.csv"
      echo "  -s: Save directory for the diff plots"
      echo "  -r: Custom repository root"
      exit 0
      ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

if [[ -z "${EMB_DIR:-}" ]]; then
    echo "[ERROR]: EMB_DIR is required (-e <embeddings_root>)"
    exit 1
fi

EMB_DIR=$(readlink -f "$EMB_DIR")
if [[ -n "${COMPARE_EMB_DIR:-}" ]]; then
    COMPARE_EMB_DIR=$(readlink -f "$COMPARE_EMB_DIR")
fi

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "--------------------------------------------------"
# Using ${SLURM_JOB_ID:-local} directly in the string to avoid unbound variable errors if SLURM_JOB_ID is not set
echo "Starting R2 Difference Plotting Job ${SLURM_JOB_ID:-local} - $NOW"
echo "--------------------------------------------------"

echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# Activate environment
source "$REPO_ROOT/.venv/bin/activate"

# Configure cache
export MPLCONFIGDIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/matplotlib"
export XDG_CACHE_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache"

# Set save directory
if [ -n "${SAVE_DIR:-}" ]; then
    SAVE_DIR=$(readlink -f "$SAVE_DIR")
else
    # Default to current embedding folder's downstream_tasks/plots
    SAVE_DIR="${EMB_DIR}/downstream_tasks/plots"
fi

# Locate the downstream results CSV for the current hybrid run
HYBRID_CSV=$(find "$EMB_DIR" -maxdepth 2 -type f -name "downstream_results.csv" | head -n 1)
if [[ -z "$HYBRID_CSV" ]]; then
    echo "[ERROR]: Could not find downstream_results.csv under $EMB_DIR"
    exit 1
fi

if [[ -n "${COMPARE_EMB_DIR:-}" ]]; then
    COMPARE_CSV=$(find "$COMPARE_EMB_DIR" -maxdepth 2 -type f -name "downstream_results.csv" | head -n 1)
    if [[ -z "$COMPARE_CSV" ]]; then
        echo "[ERROR]: Could not find downstream_results.csv under $COMPARE_EMB_DIR"
        exit 1
    fi
    echo "Running plotting script in comparison mode (Run A - Run B)"
    python3 "$PYTHON_SCRIPT" \
        --hybrid_csv "$HYBRID_CSV" \
        --compare_csv "$COMPARE_CSV" \
        --save_dir "$SAVE_DIR"
else
    # Define paths to compared baselines
    LOGS_BASE="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs"

    SUPERVISED_IMAGES="${LOGS_BASE}/supervised_baseline_images_FILTERED/supervised_baseline_images_results.csv"
    SUPERVISED_SPECTRA="${LOGS_BASE}/supervised_baseline_spectra_FILTERED/supervised_baseline_spectra_results.csv"

    UNIMODAL_IMAGES="${LOGS_BASE}/astropt_20260611_filter_unimodal_images/embeddings/best_img-mean_spec-rank_final/downstream_tasks/downstream_results.csv"
    UNIMODAL_SPECTRA="${LOGS_BASE}/astropt_20260607_filter_unimodal_spectra/embeddings/best_img-mean_spec-rank_final/downstream_tasks/downstream_results.csv"

    # Run plotting script
    python3 "$PYTHON_SCRIPT" \
        --hybrid_csv "$HYBRID_CSV" \
        --supervised_spectra_csv "$SUPERVISED_SPECTRA" \
        --supervised_images_csv "$SUPERVISED_IMAGES" \
        --unimodal_spectra_csv "$UNIMODAL_SPECTRA" \
        --unimodal_images_csv "$UNIMODAL_IMAGES" \
        --save_dir "$SAVE_DIR"
fi

echo "-----------------------------------------------"
echo "R2 Difference Plotting Finished Successfully"
echo "-----------------------------------------------"
