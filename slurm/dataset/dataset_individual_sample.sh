#!/bin/bash

# --- SLURM BATCH DIRECTIVES ---
#SBATCH --job-name=astropt_inspect_target
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/target_inspection_%j.log
#SBATCH --error=logs/target_inspection_%j.log

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
TARGET_IDS=""
OUTPUT_DIR="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_logs/dataset_individual_samples"
METADATA_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1_FILTERED.fits"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"

# If arguments are passed to sbatch (e.g., sbatch script.sh -t "123 456"), we parse them:
while getopts "t:o:m:d:r:c:s:h" opt; do
  case $opt in
    t) TARGET_IDS="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    m) METADATA_PATH="$OPTARG" ;;
    d) DATA_DIR="$OPTARG" ;;
    r) RA="$OPTARG" ;;
    c) DEC="$OPTARG" ;;
    s) ID_STR="$OPTARG" ;;
    h)
      echo "Usage: $0 [-t TARGET_IDS] [-r RA -c DEC] [-s ID_STR] [-o OUTPUT_DIR] [-m METADATA_PATH] [-d DATA_DIR]"
      echo "  -t: Space-separated list of TargetIDs in quotes"
      echo "  -r: Target Right Ascension in degrees"
      echo "  -c: Target Declination in degrees"
      echo "  -s: Lens ID string (id_str)"
      echo "  -o: Output directory to save diagnostic dashboard images"
      echo "  -m: Path to the FITS catalog containing survey metadata"
      echo "  -d: Path to Arrow dataset directory splits"
      exit 0
      ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

if [ -z "$TARGET_IDS" ] && { [ -z "$RA" ] || [ -z "$DEC" ]; } && [ -z "$ID_STR" ]; then
    echo "[CRITICAL ERROR] No query parameter provided! You must specify either:"
    echo "  -t TARGET_IDS (e.g. -t \"39633411307996619\")"
    echo "  -r RA -c DEC  (e.g. -r 266.939 -c 64.047)"
    echo "  -s ID_STR     (e.g. -s \"102157953_2669397021640472384\")"
    exit 1
fi

echo "--------------------------------------------------------"
echo "  AstroPT Target Audit & Inspector Slurm Job"
echo "--------------------------------------------------------"
echo "  Job ID:            ${SLURM_JOB_ID:-local}"
echo "  Target IDs:        ${TARGET_IDS:-N/A}"
echo "  RA / DEC:          ${RA:-N/A} / ${DEC:-N/A}"
echo "  Lens ID String:    ${ID_STR:-N/A}"
echo "  Output Directory:  $OUTPUT_DIR"
echo "  Metadata Path:     $METADATA_PATH"
echo "  Arrow Data Dir:    $DATA_DIR"
echo "--------------------------------------------------------"

# Verify file existence before running
if [ ! -d "$DATA_DIR" ]; then
    echo "[CRITICAL ERROR] Target processed data folder not found at: $DATA_DIR"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Build Python execution args
EXEC_ARGS=""
if [ -n "$TARGET_IDS" ]; then EXEC_ARGS="$EXEC_ARGS --target_ids $TARGET_IDS"; fi
if [ -n "$RA" ]; then EXEC_ARGS="$EXEC_ARGS --ra $RA"; fi
if [ -n "$DEC" ]; then EXEC_ARGS="$EXEC_ARGS --dec $DEC"; fi
if [ -n "$ID_STR" ]; then EXEC_ARGS="$EXEC_ARGS --query_id_str $ID_STR"; fi

# --- EXECUTION ---
# Run the inspection python script
python3 "$REPO_ROOT/scripts/dataset/dataset_individual_sample_results.py" \
    $EXEC_ARGS \
    --data_dir "$DATA_DIR" \
    --data_dir_interpolated "${DATA_DIR}_interpolated" \
    --metadata_path "$METADATA_PATH" \
    --output_dir "$OUTPUT_DIR"

EXIT_STATUS=$?
echo "--------------------------------------------------------"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "[SUCCESS] AstroPT Target Audit & Inspection completed successfully."
    echo "  Dashboards saved in: $OUTPUT_DIR"
else
    echo "[FAILURE] AstroPT Target Audit & Inspection failed with exit code $EXIT_STATUS."
fi
echo "--------------------------------------------------------"

exit $EXIT_STATUS
