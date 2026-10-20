#!/bin/bash

# --- SLURM BATCH DIRECTIVES ---
#SBATCH --job-name=astropt_self_attention_diagram
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/self_attention_diagram_%j.log
#SBATCH --error=logs/self_attention_diagram_%j.log

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
OUTPUT_DIR="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_logs/attention_diagrams"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"

while getopts "t:o:d:h" opt; do
  case $opt in
    t) TARGET_ID="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    d) DATA_DIR="$OPTARG" ;;
    h)
      echo "Usage: $0 [-t TARGET_ID] [-o OUTPUT_DIR] [-d DATA_DIR]"
      exit 0
      ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

echo "--------------------------------------------------------"
echo "  AstroPT Self-Attention Diagram Generator"
echo "--------------------------------------------------------"
echo "  Target ID:         $TARGET_ID"
echo "  Output Directory:  $OUTPUT_DIR"
echo "  Arrow Data Dir:    $DATA_DIR"
echo "--------------------------------------------------------"

# --- EXECUTION ---
python3 "$REPO_ROOT/scripts/dataset/dataset_self_attention_diagram.py" \
    --target_id "$TARGET_ID" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

EXIT_STATUS=$?
echo "--------------------------------------------------------"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "[SUCCESS] Self-Attention Diagram generation completed successfully."
else
    echo "[FAILURE] Self-Attention Diagram generation failed with exit code $EXIT_STATUS."
fi
echo "--------------------------------------------------------"

exit $EXIT_STATUS
