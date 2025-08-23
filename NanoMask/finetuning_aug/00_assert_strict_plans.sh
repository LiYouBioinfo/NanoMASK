#!/usr/bin/env bash
# Usage: bash 00_assert_strict_plans.sh TaskXXX_NAME
set +e  # (you asked to avoid -euo)

TASK_DIRNAME="${1:?Usage: bash 00_assert_strict_plans.sh TaskXXX_NAME}"

# Respect your current env; fall back to default layout
WORKROOT="/platforms/radiomics/NanoMask/finetuning_aug"
: "${nnUNet_raw_data_base:=$WORKROOT/nnUNet_data/nnUNet_raw_data}"
: "${nnUNet_preprocessed:=$WORKROOT/nnUNet_data/pre_data}"
: "${RESULTS_FOLDER:=$WORKROOT/nnUNet_data/nnUNet}"

BASE="$nnUNet_preprocessed"
SUB="$BASE/$TASK_DIRNAME/nnUNetData_plans_v2.1/nnUNetPlansv2.1_plans_3D.pkl"
TOP="$BASE/$TASK_DIRNAME/nnUNetPlansv2.1_plans_3D.pkl"

if [ ! -f "$SUB" ]; then
  echo "[ERR] Missing strict plans at: $SUB"
  echo "      Run 02_create_task_strict.sh first."
  exit 1
fi

# Make the top-level file a symlink to the strict one
rm -f "$TOP"
ln -s "$SUB" "$TOP"
echo "[OK] Plans symlinked: $TOP -> $(readlink -f "$TOP")"

# Precheck: what nnU-Net will actually load
python - <<'PY' "$TASK_DIRNAME" "$WORKROOT"
import os, sys, pickle
from nnunet.run.default_configuration import get_default_configuration
task, workroot = sys.argv[1], sys.argv[2]
os.environ.setdefault('nnUNet_raw_data_base', f"{workroot}/nnUNet_data/nnUNet_raw_data")
os.environ.setdefault('nnUNet_preprocessed',   f"{workroot}/nnUNet_data/pre_data")
os.environ.setdefault('RESULTS_FOLDER',        f"{workroot}/nnUNet_data/nnUNet")
pf, *_ = get_default_configuration('3d_fullres', task, 'nnUNetTrainerV2', 'nnUNetPlansv2.1')
p=pickle.load(open(pf,'rb'))
print(f"[PRECHECK] PLANS_FILE: {pf}")
print(f"[PRECHECK] num_pool_per_axis: {p['plans_per_stage'][0]['num_pool_per_axis']}")
print(f"[PRECHECK] pool_op_kernel_sizes[0]: {p['plans_per_stage'][0]['pool_op_kernel_sizes'][0]}")
print(f"[PRECHECK] unet_max_num_features: {p.get('UNet_max_num_features') or p.get('unet_max_num_features')}")
PY
