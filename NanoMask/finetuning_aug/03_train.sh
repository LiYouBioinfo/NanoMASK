#!/usr/bin/env bash
WORKROOT="/platforms/radiomics/NanoMask/finetuning_aug"
STATE_FILE="$WORKROOT/.last_task"

# args: [TASK_DIRNAME] [GPU_ID] [FOLD]
TASK_DIRNAME_ARG="$1"
GPU_ID="${2:-0}"
FOLD="${3:-0}"
TRAINER="${TRAINER:-nnUNetTrainerV2}"

if [ -n "$TASK_DIRNAME_ARG" ]; then
  TASK_DIRNAME="$TASK_DIRNAME_ARG"
else
  if [ ! -f "$STATE_FILE" ]; then
    echo "[ERR] No state file. Run 02_create_task_strict.sh first or pass TASK_DIRNAME explicitly." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$STATE_FILE"
fi

# Optional NVMe routing (if mounted)
[ -d /mnt/nvme/pre_data ] && export nnUNet_preprocessed="/mnt/nvme/pre_data" || export nnUNet_preprocessed="$WORKROOT/nnUNet_data/pre_data"
[ -d /mnt/nvme/nnUNet ]   && export RESULTS_FOLDER="/mnt/nvme/nnUNet"        || export RESULTS_FOLDER="$WORKROOT/nnUNet_data/nnUNet"
export nnUNet_raw_data_base="$WORKROOT/nnUNet_data/nnUNet_raw_data"

# Keep BLAS threads sane
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Pin to one GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

PRETRAINED="${PRETRAINED:-$WORKROOT/pretrained/Task006_CT2PET_fold0/model_final_checkpoint.model}"

ensure_strict_plans() {
  local BASE="$nnUNet_preprocessed"
  local SUB="$BASE/$TASK_DIRNAME/nnUNetData_plans_v2.1/nnUNetPlansv2.1_plans_3D.pkl"
  local TOP="$BASE/$TASK_DIRNAME/nnUNetPlansv2.1_plans_3D.pkl"
  [ -f "$SUB" ] || { echo "[ERR] Missing $SUB"; return 1; }
  if [ -e "$TOP" ]; then rm -f "$TOP"; fi
  ln -s "$SUB" "$TOP"
  echo "[OK] Plans file: $TOP -> $(readlink -f "$TOP")"
}

ensure_strict_plans || exit 1

python - <<'PY'
import pickle, os
from nnunet.run.default_configuration import get_default_configuration
pf, *_ = get_default_configuration('3d_fullres', os.environ.get('TASK_DIRNAME','Task200_CT2PET_FT'),
                                   'nnUNetTrainerV2','nnUNetPlansv2.1')
p=pickle.load(open(pf,'rb'))
print(f"[PRECHECK] PLANS_FILE: {pf}")
print(f"[PRECHECK] num_pool_per_axis: {p['plans_per_stage'][0]['num_pool_per_axis']}")
print(f"[PRECHECK] unet_max_num_features: {p.get('UNet_max_num_features') or p.get('unet_max_num_features')}")
PY


echo "[INFO] GPU:$GPU_ID  FOLD:$FOLD  TRAINER:$TRAINER  TASK:$TASK_DIRNAME"
echo "[INFO] nnUNet_raw_data_base=$nnUNet_raw_data_base"
echo "[INFO] nnUNet_preprocessed=$nnUNet_preprocessed"
echo "[INFO] RESULTS_FOLDER=$RESULTS_FOLDER"
echo "[INFO] Pretrained: $PRETRAINED"

bash /platforms/radiomics/NanoMask/finetuning_aug/00_assert_strict_plans.sh "$TASK_DIRNAME" || exit 1

nnUNet_train 3d_fullres "$TRAINER" "$TASK_DIRNAME" "$FOLD" \
  -pretrained_weights "$PRETRAINED" \
  -p nnUNetPlansv2.1
