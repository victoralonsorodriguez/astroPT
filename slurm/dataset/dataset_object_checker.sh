#!/bin/bash

# --- SLURM BATCH DIRECTIVES ---
#SBATCH --job-name=astropt_check_overlap
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/overlap_check_%j.log
#SBATCH --error=logs/overlap_check_%j.log

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

# --- ARGUMENT PARSING ---
QUERY_TABLE=""
ID_COLS=""
OUTPUT_DIR="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_logs"
METADATA_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1_FILTERED.fits"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"

while getopts "q:c:o:m:d:h" opt; do
  case $opt in
    q) QUERY_TABLE="$OPTARG" ;;
    c) ID_COLS="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    m) METADATA_PATH="$OPTARG" ;;
    d) DATA_DIR="$OPTARG" ;;
    h)
      echo "Usage: $0 -q QUERY_TABLE -c \"ID_COL1 ID_COL2\" [-o OUTPUT_DIR] [-m METADATA_PATH] [-d DATA_DIR]"
      echo "  -q: Path to the query catalog file (FITS, CSV, or Parquet)"
      echo "  -c: Space-separated list of ID columns in quotes"
      echo "  -o: Directory to save reports"
      echo "  -m: Path to the FITS catalog containing survey metadata"
      echo "  -d: Path to Arrow dataset directory splits"
      exit 0
      ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

if [ -z "$QUERY_TABLE" ] || [ -z "$ID_COLS" ]; then
    echo "[CRITICAL ERROR] Missing required arguments! Both -q and -c must be specified."
    echo "Usage: $0 -q QUERY_TABLE -c \"ID_COL1 ID_COL2\""
    exit 1
fi

# Run python script
python3 "$REPO_ROOT/scripts/dataset/dataset_object_checker.py" \
    --query_table "$QUERY_TABLE" \
    --id_cols $ID_COLS \
    --metadata_fits "$METADATA_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

exit $?
