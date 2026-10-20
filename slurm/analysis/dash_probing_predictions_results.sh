#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Preds_Diag
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=4       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=32G                
#SBATCH --time=01:00:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_preds_diag_%j.out
#SBATCH --error=logs/astropt_preds_diag_%j.err

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

PYTHON_SCRIPT="$REPO_ROOT/scripts/analysis/dashboards/dash_probing_predictions_results.py"

EMB_DIR=""
SAVE_DIR=""
TARGETS=("HBETA_FLUX" "LOGMSTAR" "sersic_sersic_vis_index" "sersic_sersic_vis_radius") # Default targets
MODES=("EuclidImage_pooled" "DESISpectrum_pooled" "joint_mean") # Default modalities
PROBES=("MLP") # Default probes

# Support sbatch sscript.sh -- options...
if [[ "${1:-}" == "--" ]]; then
  shift
fi

#--- ARGUMENT PARSING ---#
# We parse options using getopts. For multi-value arguments like targets, modes, 
# we can override them via custom flags or pass them as space-separated lists.
while getopts ":r:e:s:t:m:p:h" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    t) IFS=' ' read -r -a TARGETS <<< "$OPTARG" ;;
    m) IFS=' ' read -r -a MODES <<< "$OPTARG" ;;
    p) IFS=' ' read -r -a PROBES <<< "$OPTARG" ;;
    h)
      echo "Usage: $0 -e EMB_DIR [-s SAVE_DIR] [-t 'Z HALPHA_FLUX'] [-m 'EuclidImage DESISpectrum joint'] [-p 'MLP LP'] [-r REPO_ROOT]"
      echo "  -e: Specify embeddings directory (required)"
      echo "  -s: Specify save directory for generated PNGs"
      echo "  -t: Space-separated target names (default: 'Z HALPHA_FLUX')"
      echo "  -m: Space-separated modes/modalities (default: 'EuclidImage DESISpectrum joint')"
      echo "  -p: Space-separated probes (default: 'MLP')"
      echo "  -r: Specify repository root directory"
      exit 0
      ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

if [[ -z "${EMB_DIR:-}" ]]; then
    echo "[ERROR]: EMB_DIR is required (-e <embeddings_dir>)"
    exit 1
fi

EMB_DIR=$(readlink -f "$EMB_DIR")
PRED_DIR="$EMB_DIR/downstream_tasks/predictions"

# Fallback in case path directly points to embedding subfolder
if [[ ! -d "$PRED_DIR" ]]; then
    if [[ -d "$EMB_DIR/predictions" ]]; then
        PRED_DIR="$EMB_DIR/predictions"
    else
        echo "[ERROR]: Predictions directory not found under $EMB_DIR"
        exit 1
    fi
fi

if [[ -z "${SAVE_DIR:-}" ]]; then
    SAVE_DIR="${EMB_DIR}/downstream_tasks/plots/diagnostic_predictions"
else
    SAVE_DIR=$(readlink -f "$SAVE_DIR")
fi

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "--------------------------------------------------"
echo "Starting Regression Diagnostic Plotting Job ${SLURM_JOB_ID:-local} - $NOW"
echo "--------------------------------------------------"

echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# Activate Environment
source "$REPO_ROOT/.venv/bin/activate"

# Configure cache directories for stable rendering on the cluster
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
export MPLCONFIGDIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/matplotlib"
export XDG_CACHE_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache"

echo "Running Diagnostic plotting script for:"
echo "    PRED_DIR:  $PRED_DIR"
echo "    SAVE_DIR:  $SAVE_DIR"
echo "    TARGETS:   ${TARGETS[*]}"
echo "    MODES:     ${MODES[*]}"
echo "    PROBES:    ${PROBES[*]}"

# Run the python script
python3 "$PYTHON_SCRIPT" \
    --pred_dir "$PRED_DIR" \
    --save_dir "$SAVE_DIR" \
    --targets "${TARGETS[@]}" \
    --modes "${MODES[@]}" \
    --probes "${PROBES[@]}"

echo "-----------------------------------------------"
echo "Diagnostic Plotting Job Finished Successfully"
echo "-----------------------------------------------"
