#!/usr/bin/env bash
WORKROOT="/platforms/radiomics/NanoMask/finetuning_aug"
mkdir -p "$WORKROOT"
cd "$WORKROOT"

# Create nnU-Net v1 folder structure (local)
mkdir -p ./nnUNet_data/nnUNet_raw_data \
         ./nnUNet_data/pre_data \
         ./nnUNet_data/nnUNet_cropped_data \
         ./nnUNet_data/nnUNet

# Point nnU-Net to these local dirs (export for this shell)
export nnUNet_raw_data_base="$WORKROOT/nnUNet_data/nnUNet_raw_data"
export nnUNet_preprocessed="$WORKROOT/nnUNet_data/pre_data"
export RESULTS_FOLDER="$WORKROOT/nnUNet_data/nnUNet"

echo "[OK] Workspace ready."
echo "nnUNet_raw_data_base=$nnUNet_raw_data_base"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "RESULTS_FOLDER=$RESULTS_FOLDER"
