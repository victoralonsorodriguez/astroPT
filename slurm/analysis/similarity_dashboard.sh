
#!/bin/bash

# --- SLURM BATCH DIRECTIVES ---
#SBATCH --job-name=astropt_appendix_a_mosaic
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/appendix_a_mosaic_%j.log
#SBATCH --error=logs/appendix_a_mosaic_%j.log

# Robust repository root detection based on script location
# (Script is in slurm/trainers/, so REPO_ROOT is two levels up)
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
QUERY_ID=""
EUCLID_ID=""
DESI_ID=""
JOINT_ID=""
OUTPUT_DIR="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_logs/similarity"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"

while getopts "q:e:d:j:o:p:h" opt; do
  case $opt in
    q) QUERY_ID="$OPTARG" ;;
    e) EUCLID_ID="$OPTARG" ;;
    d) DESI_ID="$OPTARG" ;;
    j) JOINT_ID="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    p) DATA_DIR="$OPTARG" ;;
    h)
      echo "Usage: $0 -q QUERY_ID -e EUCLID_ID -d DESI_ID -j JOINT_ID [-o OUTPUT_DIR] [-p DATA_DIR]"
      exit 0
      ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

if [ -z "$QUERY_ID" ] || [ -z "$EUCLID_ID" ] || [ -z "$DESI_ID" ] || [ -z "$JOINT_ID" ]; then
    echo "[ERROR] You must provide all 4 Target IDs: -q, -e, -d, -j"
    exit 1
fi

echo "--------------------------------------------------------"
echo "  AstroPT Appendix A Mosaic Generator"
echo "--------------------------------------------------------"
echo "  Query ID:          $QUERY_ID"
echo "  Euclid k-NN ID:    $EUCLID_ID"
echo "  DESI k-NN ID:      $DESI_ID"
echo "  Joint k-NN ID:     $JOINT_ID"
echo "  Output Directory:  $OUTPUT_DIR"
echo "  Data Directory:    $DATA_DIR"
echo "--------------------------------------------------------"

# --- EXECUTION ---
python3 "$REPO_ROOT/scripts/analysis/similarity/similarity_dashboard.py" \
    --query_id "$QUERY_ID" \
    --euclid_id "$EUCLID_ID" \
    --desi_id "$DESI_ID" \
    --joint_id "$JOINT_ID" \
    --output_dir "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR"

EXIT_STATUS=$?
echo "--------------------------------------------------------"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "[SUCCESS] Appendix A Mosaic generation completed successfully."
else
    echo "[FAILURE] Appendix A Mosaic generation failed with exit code $EXIT_STATUS."
fi
echo "--------------------------------------------------------"

exit $EXIT_STATUS
