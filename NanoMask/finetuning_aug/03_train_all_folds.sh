#!/usr/bin/env bash
WORKROOT="/platforms/radiomics/NanoMask/finetuning_aug"
STATE_FILE="$WORKROOT/.last_task"
TRAINER="${TRAINER:-nnUNetTrainerV2}"

if [ -n "$1" ]; then
  TASK_DIRNAME="$1"
else
  if [ -f "$STATE_FILE" ]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
  else
    echo "[ERR] Provide TASK_DIRNAME or create $STATE_FILE first." >&2
    exit 1
  fi
fi

# Optional NVMe routing (if mounted)
[ -d /mnt/nvme/pre_data ] && export nnUNet_preprocessed="/mnt/nvme/pre_data" || export nnUNet_preprocessed="$WORKROOT/nnUNet_data/pre_data"
[ -d /mnt/nvme/nnUNet ]   && export RESULTS_FOLDER="/mnt/nvme/nnUNet"        || export RESULTS_FOLDER="$WORKROOT/nnUNet_data/nnUNet"
export nnUNet_raw_data_base="$WORKROOT/nnUNet_data/nnUNet_raw_data"

# BLAS threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

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


logdir="$WORKROOT/logs/$TASK_DIRNAME"
mkdir -p "$logdir"
bash /platforms/radiomics/NanoMask/finetuning_aug/00_assert_strict_plans.sh "$TASK_DIRNAME" || exit 1

echo "[INFO] Launching folds 0..3 on GPUs 0..3 (TRAINER=$TRAINER)"
for gpu in 0 1 2 3; do
  fold="$gpu"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[LAUNCH] GPU $gpu -> fold $fold"
    nnUNet_train 3d_fullres "$TRAINER" "$TASK_DIRNAME" "$fold" \
      -pretrained_weights "$PRETRAINED" \
      -p nnUNetPlansv2.1 \
      2>&1 | tee "$logdir/train_fold${fold}.log"
  ) &
done
wait
echo "[DONE] All folds finished."
