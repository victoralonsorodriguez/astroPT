#!/bin/bash

# --- SLURM BATCH DIRECTIVES ---
#SBATCH --job-name=astropt_token_mixing_diagram
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/token_mixing_diagram_%j.log
#SBATCH --error=logs/token_mixing_diagram_%j.log

# --- ROBUST REPOSITORY ROOT DETECTION ---
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$SCRIPT_DIR" == *"/slurm/dataset"* ]]; then
    REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
else
    REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi

cd "$REPO_ROOT" || { echo "[ERROR] Failed to navigate to repository root: $REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

# --- ENVIRONMENT ACTIVATION ---
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    echo "Activating project virtual environment..."
    source "$REPO_ROOT/.venv/bin/activate"
else
    echo "[WARNING] Virtual environment activator not found at: $REPO_ROOT/.venv/bin/activate"
fi

# --- CONFIGURATION VARIABLES & ARGUMENT PARSING ---
TARGET_ID="39633516366922547"
OUTPUT_DIR="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_logs/token_mixing"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"

# Default schema params
M_PATCHES=8
N_POINTS=250
L_VIS=4
L_SPEC=4
MAX_TOKENS_ROW=22
NUM_SEQ_ROWS=4

while getopts "t:o:d:m:n:v:s:r:w:h" opt; do
  case $opt in
    t) TARGET_ID="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    d) DATA_DIR="$OPTARG" ;;
    m) M_PATCHES="$OPTARG" ;;
    n) N_POINTS="$OPTARG" ;;
    v) L_VIS="$OPTARG" ;;
    s) L_SPEC="$OPTARG" ;;
    w) MAX_TOKENS_ROW="$OPTARG" ;;
    r) NUM_SEQ_ROWS="$OPTARG" ;;
    h)
      echo "Usage: $0 [-t TARGET_ID] [-o OUTPUT_DIR] [-d DATA_DIR] [-m M_PATCHES] [-n N_POINTS] [-v L_VIS] [-s L_SPEC] [-w MAX_TOKENS_ROW] [-r NUM_SEQ_ROWS]"
      exit 0
      ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

echo "--------------------------------------------------------"
echo "  AstroPT Token Mixing Diagram Generator"
echo "--------------------------------------------------------"
echo "  Target ID:         $TARGET_ID"
echo "  Output Directory:  $OUTPUT_DIR"
echo "  Arrow Data Dir:    $DATA_DIR"
echo "  Image Grid (m):    ${M_PATCHES}x${M_PATCHES}"
echo "  Spec Points (n):   $N_POINTS"
echo "  Vis Block (l_vis): $L_VIS"
echo "  Spec Block (l_spe):$L_SPEC"
echo "--------------------------------------------------------"

# --- EXECUTION ---
python3 "$REPO_ROOT/scripts/dataset/dataset_token_mixing_diagram.py" \
    --target_id "$TARGET_ID" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --m_patches "$M_PATCHES" \
    --n_points "$N_POINTS" \
    --l_vis "$L_VIS" \
    --l_spec "$L_SPEC" \
    --max_tokens_row "$MAX_TOKENS_ROW" \
    --num_sequence_rows "$NUM_SEQ_ROWS"

EXIT_STATUS=$?
echo "--------------------------------------------------------"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "[SUCCESS] Token Mixing Diagram generation completed successfully."
else
    echo "[FAILURE] Token Mixing Diagram generation failed with exit code $EXIT_STATUS."
fi
echo "--------------------------------------------------------"

exit $EXIT_STATUS
