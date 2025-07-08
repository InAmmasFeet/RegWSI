#!/bin/bash
# Simple wrapper to run the RegWSI pipeline

echo "==================================================================================="
echo "RegWSI Pipeline Runner for vast.ai"
echo "==================================================================================="

# ----------------------------------------------------------------------
#  Activate Python environment & set CUDA variables
# ----------------------------------------------------------------------
source /workspace/regwsi/regwsi_env/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ----------------------------------------------------------------------
#  Argument handling
# ----------------------------------------------------------------------
if [ $# -lt 2 ]; then
    echo "Usage: $0 <he_qptiff_path> <if_qptiff_path> [output_dir]"
    echo
    echo "Example:"
    echo "  $0 /data/HE_scan.qptiff /data/IF_scan.qptiff /workspace/output"
    exit 1
fi

HE_QPTIFF=$1
IF_QPTIFF=$2
OUTPUT_DIR=${3:-/workspace/data/output}

# ----------------------------------------------------------------------
#  Sanity checks
# ----------------------------------------------------------------------
if [ ! -f "$HE_QPTIFF" ]; then
    echo "Error: H&E QPTIFF file not found: $HE_QPTIFF"
    exit 1
fi

if [ ! -f "$IF_QPTIFF" ]; then
    echo "Error: IF QPTIFF file not found: $IF_QPTIFF"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ----------------------------------------------------------------------
#  System information
# ----------------------------------------------------------------------
echo
echo "System Information:"
echo "==================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo
echo "CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Storage: $(df -h /workspace | awk 'END{print $4}') available"
echo

# ----------------------------------------------------------------------
#  Run the RegWSI registration pipeline
# ----------------------------------------------------------------------
echo "Starting RegWSI Pipeline..."
echo "==================================================================================="

python3 /workspace/regwsi/regwsi_registration_pipeline.py \
    --he_qptiff "$HE_QPTIFF" \
    --if_qptiff  "$IF_QPTIFF" \
    --output     "$OUTPUT_DIR" \
    --tile_size  512 \
    --downsample_he 2

STATUS=$?

# ----------------------------------------------------------------------
#  Post‑run reporting
# ----------------------------------------------------------------------
if [ $STATUS -eq 0 ]; then
    echo
    echo "==================================================================================="
    echo "Pipeline completed successfully!"
    echo "Output saved to: $OUTPUT_DIR"
    echo
    echo "Output Structure:"
    find "$OUTPUT_DIR" -type d -name "regwsi_run_*" -exec ls -la {} \; | head -20

    LATEST_RUN=$(ls -td "$OUTPUT_DIR"/regwsi_run_* 2>/dev/null | head -1)
    if [ -d "$LATEST_RUN/tiles/train_A" ]; then
        N_TILES=$(find "$LATEST_RUN/tiles/train_A" -type f | wc -l)
        echo
        echo "Tiles extracted: $N_TILES"
        echo "Ready for Pix2PixHD training!"
    fi
else
    echo
    echo "==================================================================================="
    echo "Pipeline failed! Check the error logs in the output directory."
    exit 1
fi
