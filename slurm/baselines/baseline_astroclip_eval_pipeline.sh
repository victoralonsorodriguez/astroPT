#!/bin/bash

#--- STANDALONE ASTROCLIP POST-ANALYSIS PIPELINE LAUNCHER ---#
echo "-------------------------------------------------"
echo "Launching Standalone AstroCLIP Analysis Battery"
echo "-------------------------------------------------"

# Robust repository root detection based on script location
# (Script is in slurm/baselines/, so REPO_ROOT is two levels up)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# If SLURM has moved us to /var/spool, correct REPO_ROOT to the launch working directory
if [[ "$REPO_ROOT" == "/var/spool"* ]]; then
    REPO_ROOT="$PWD"
fi

# --- USER CONFIGURATION ---
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1_FILTERED.fits"
# --------------------------

# LAUNCHING SCRIPTS
PLOT_MET_SCRIPT="$REPO_ROOT/slurm/analysis/dash_training_metrics.sh"
EXT_EMBD_SCRIPT="$SCRIPT_DIR/baseline_astroclip_embeddings.sh"
COS_SIM_SCRIPT="$REPO_ROOT/slurm/analysis/dash_embeddings_alignment.sh"
UMAPS_SCRIPT="$REPO_ROOT/slurm/analysis/dash_latent_projection.sh"
PROBING_SCRIPT="$REPO_ROOT/slurm/probing/probing_downstream_benchmark.sh"
PROBING_DASH_SCRIPT="$REPO_ROOT/slurm/analysis/dash_probing_downstream.sh"
PREDICT_DASH_SCRIPT="$REPO_ROOT/slurm/analysis/dash_probing_predictions.sh"
LATENT_SCRIPT="$REPO_ROOT/slurm/probing/probing_latent_retrieval.sh"
LATENT_DASH_SCRIPT="$REPO_ROOT/slurm/analysis/dash_probing_latent.sh"

# ARGUMENTS DEFAULT VALUES
TRAIN_DIR=""
JOB_SUFFIX=""

while getopts ":t:s:a:f:h" opt; do
  case $opt in
    t) TRAIN_DIR="$OPTARG" ;;
    s) JOB_SUFFIX="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    f) META_PATH="$OPTARG" ;;
    h)
       echo "Usage: $0 -t <TRAIN_DIR> [-s '_suffix'] [-a <data_dir>] [-f <metadata_path>]"
       exit 0
       ;;
    \?)
       echo "Invalid option: -$OPTARG" >&2
       exit 1
       ;;
  esac
done

if [ -z "$TRAIN_DIR" ]; then
    echo "[ERROR] You must provide a target training directory using -t."
    exit 1
fi

if [ ! -d "$TRAIN_DIR" ]; then
    echo "[ERROR] Target directory does not exist: $TRAIN_DIR"
    exit 1
fi

# Absolute output paths
TRAIN_DIR=$(readlink -f "$TRAIN_DIR")
WEIGHTS_DIR="$TRAIN_DIR/weights"
LOGS_DIR="$TRAIN_DIR/logs"
PLOTS_DIR="$TRAIN_DIR/plots"
EMB_DIR="$TRAIN_DIR/embeddings"
EMB_STAGE_TAG="${JOB_SUFFIX#_}"

# Ensure directories exist
mkdir -p "$PLOTS_DIR" "$EMB_DIR"

# Change to REPO_ROOT so that any sbatch calls inherit the correct SLURM_SUBMIT_DIR
cd "$REPO_ROOT" || { echo "[ERROR] Failed to cd to $REPO_ROOT"; exit 1; }

echo "Target Training Directory: $TRAIN_DIR"
echo "Job Suffix:                $JOB_SUFFIX"
echo "Data Directory:            $DATA_DIR"
echo "Metadata Path:             $META_PATH"
echo "-------------------------------------------------"

# 1. Plotting Metrics
J_MET=$(sbatch --parsable \
            --job-name="Plot_Metrics$JOB_SUFFIX" \
            "$PLOT_MET_SCRIPT" \
            -r "$REPO_ROOT" \
            -w "$WEIGHTS_DIR" \
            -l "$LOGS_DIR" \
            -s "$PLOTS_DIR"
        )
echo "    [Metric]      Job sent. ID: $J_MET"

# 2. Extract Embeddings (AstroCLIP-aware)
# If JOB_SUFFIX is empty or not provided, we extract embeddings with no stage tag or default
J_EMB=$(sbatch --parsable \
            --job-name="Extract_Embed$JOB_SUFFIX" \
            "$EXT_EMBD_SCRIPT" \
            -r "$REPO_ROOT" \
            -w "$WEIGHTS_DIR" \
            -s "$EMB_DIR" \
            -x "$EMB_STAGE_TAG" \
            -a "$DATA_DIR"
        )
echo "    [Embeds]      Job sent. ID: $J_EMB"

if [[ -z "$J_EMB" ]]; then
    echo "    [ERROR] Embedding extraction submission failed. Stopping analysis chain."
    exit 1
fi

# 3. Cosine Similarity
J_COS=$(sbatch --parsable \
            --dependency=afterok:$J_EMB \
            --job-name="Cos_Sim$JOB_SUFFIX" \
            "$COS_SIM_SCRIPT" \
            -r "$REPO_ROOT" \
            -w "$WEIGHTS_DIR" \
            -e "$EMB_DIR"
        )
echo "    [CosSim]      Job sent. ID: $J_COS (Depends on Embeds: ok)"

# 4. UMAPS
J_UMAP=$(sbatch --parsable \
            --dependency=afterok:$J_EMB \
            --job-name="Plot_Umaps$JOB_SUFFIX" \
            "$UMAPS_SCRIPT" \
            -r "$REPO_ROOT" \
            -w "$WEIGHTS_DIR" \
            -e "$EMB_DIR" \
            -a "$DATA_DIR" \
            -f "$META_PATH"
        )
echo "    [UMAPS]       Job sent. ID: $J_UMAP (Depends on Embeds: ok)"

# 5. Probing Downstream Tasks
J_PROB=$(sbatch --parsable \
            --dependency=afterok:$J_EMB \
            --job-name="Probing_Tasks$JOB_SUFFIX" \
            "$PROBING_SCRIPT" \
            -r "$REPO_ROOT" \
            -w "$WEIGHTS_DIR" \
            -e "$EMB_DIR" \
            -f "$META_PATH"
        )
echo "    [PROBING]     Job sent. ID: $J_PROB (Depends on Embeds: ok)"

if [[ -z "$J_PROB" ]]; then
    echo "    [ERROR] Downstream probing submission failed. Stopping analysis chain."
    exit 1
fi

# 6. Probing Dashboard
J_PROB_DASH=$(sbatch --parsable \
            --dependency=afterok:$J_PROB \
            --job-name="Probing_Tasks_Dash$JOB_SUFFIX" \
            "$PROBING_DASH_SCRIPT" \
            -r "$REPO_ROOT" \
            -e "$EMB_DIR"
        )
echo "    [PROB DASH]   Job sent. ID: $J_PROB_DASH (Depends on Probing: ok)"

# 7. Probing Predictions Dashboard
J_PRED_DASH=$(sbatch --parsable \
            --dependency=afterok:$J_PROB \
            --job-name="Predict_Dash$JOB_SUFFIX" \
            "$PREDICT_DASH_SCRIPT" \
            -r "$REPO_ROOT" \
            -e "$EMB_DIR"
        )
echo "    [PRED DASH]   Job sent. ID: $J_PRED_DASH (Depends on Probing: ok)"

# 8. Latent Mapper
J_LATENT=$(sbatch --parsable \
            --dependency=afterok:$J_EMB \
            --job-name="Latent_Tasks$JOB_SUFFIX" \
            "$LATENT_SCRIPT" \
            -r "$REPO_ROOT" \
            -w "$WEIGHTS_DIR" \
            -e "$EMB_DIR" \
            -f "$META_PATH"
        )
echo "    [LATENT]      Job sent. ID: $J_LATENT (Depends on Embeds: ok)"

if [[ -z "$J_LATENT" ]]; then
    echo "    [ERROR] Latent probing submission failed. Stopping analysis chain."
    exit 1
fi

# 9. Latent Dashboard
J_LATENT_DASH=$(sbatch --parsable \
            --dependency=afterok:$J_LATENT \
            --job-name="Latent_Tasks_Dash$JOB_SUFFIX" \
            "$LATENT_DASH_SCRIPT" \
            -r "$REPO_ROOT" \
            -e "$EMB_DIR"
        )
echo "    [LATENT DASH] Job sent. ID: $J_LATENT_DASH (Depends on Latent: ok)"

echo "-------------------------------------------------"
echo "Pipeline Status Queue:"
squeue -u "$USER" -S i -o "%.10i %.6P %.25j %.8T %.10M %.35E %R"
echo "-------------------------------------------------"
