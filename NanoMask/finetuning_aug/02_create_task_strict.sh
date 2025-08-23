#!/usr/bin/env bash
# Strict creator: stage -> preprocess -> copy+verify Task006 plans

# ----------------- Config -----------------
WORKROOT="/platforms/radiomics/NanoMask/finetuning_aug"
SINGLES_ROOT="/platforms/radiomics/NanoMask/FinetuneData"
TASK_NAME="${TASK_NAME:-CT2PET_FT}"      # override via env if desired
COPY_FILES="${COPY_FILES:-0}"            # 1=copy, 0=symlink
LOCK_PLANS="${LOCK_PLANS:-0}"            # 1=attempt chattr +i after copy

# Source (pretrained Task006)
PLANS_SRC="/platforms/radiomics/NanoMask/nnunet/nnUNet_data/nnUNet/3d_fullres/Task006_CT2PET/nnUNetTrainerV2__nnUNetPlansv2.1/plans.pkl"
PRETRAINED_SRC="/platforms/radiomics/NanoMask/nnunet/nnUNet_data/nnUNet/3d_fullres/Task006_CT2PET/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model"

# ----------------- Env for nnU-Net -----------------
export nnUNet_raw_data_base="$WORKROOT/nnUNet_data/nnUNet_raw_data"
export nnUNet_preprocessed="$WORKROOT/nnUNet_data/pre_data"
export RESULTS_FOLDER="$WORKROOT/nnUNet_data/nnUNet"
mkdir -p "$nnUNet_raw_data_base" "$nnUNet_preprocessed" "$RESULTS_FOLDER"

# ----------------- Helpers -----------------
pick_task_id() {
  local used last i
  used=$(ls -1 "$nnUNet_raw_data_base" 2>/dev/null | grep -E '^Task[0-9]{3}_' | awk -F'_' '{print substr($1,5,3)}' || true)
  for i in $(seq 200 299); do
    printf "%s\n" "$used" | grep -qx "$(printf '%03d' "$i")" || { echo "$i"; return; }
  done
  last=$(printf "%s\n" "$used" | sort -n | tail -n1)
  if [ -z "${last:-}" ]; then echo 200; else echo $((10#$last + 1)); fi
}

die() { echo "[ERR] $*" >&2; exit 1; }
need_file() { [ -f "$1" ] || die "Missing file: $1"; }

# ----------------- Preflight -----------------
need_file "$PLANS_SRC"
need_file "$PRETRAINED_SRC"
[ -f "$WORKROOT/01_stage_task_from_singles.py" ] || die "Missing: $WORKROOT/01_stage_task_from_singles.py"
command -v nnUNet_plan_and_preprocess >/dev/null 2>&1 || die "nnUNet_plan_and_preprocess not found in PATH"

# ----------------- Choose Task ID -----------------
TASK_ID=$(pick_task_id)
TASK_DIRNAME="Task$(printf '%03d' "$TASK_ID")_${TASK_NAME}"
echo "[INFO] Using Task ID: $TASK_ID  ->  $TASK_DIRNAME"

# ----------------- Stage dataset -----------------
echo "[STEP] Staging dataset from: $SINGLES_ROOT"
python "$WORKROOT/01_stage_task_from_singles.py" \
  --singles-root "$SINGLES_ROOT" \
  --task-id "$TASK_ID" \
  --task-name "$TASK_NAME" \
  $([ "$COPY_FILES" -eq 1 ] && echo --copy || true)

# ----------------- Plan & preprocess -----------------
echo "[STEP] nnUNet_plan_and_preprocess -t $TASK_ID (auto-generates plans we will overwrite)"
nnUNet_plan_and_preprocess -t "$TASK_ID" --verify_dataset_integrity -tl 8 -tf 8

# ----------------- Strict plans swap -----------------
DST_PRE="$nnUNet_preprocessed/$TASK_DIRNAME/nnUNetData_plans_v2.1"
DST_PLANS="$DST_PRE/nnUNetPlansv2.1_plans_3D.pkl"
mkdir -p "$DST_PRE"

[ -f "$DST_PLANS" ] && cp -v "$DST_PLANS" "$DST_PLANS.bak.$(date +%s)" || true
cp -v "$PLANS_SRC" "$DST_PLANS"
echo "PLANS_SOURCE=$PLANS_SRC" > "$DST_PRE/PLANS_LOCK.txt"

# Optional lock (best effort; ignore failures)
if [ "$LOCK_PLANS" -eq 1 ]; then
  if command -v chattr >/dev/null 2>&1; then
    sudo chattr +i "$DST_PLANS" 2>/dev/null || true
    echo "[INFO] Attempted to lock $DST_PLANS (chattr +i)."
  else
    echo "[WARN] chattr not found; skipping lock."
  fi
fi

# ----------------- Verify swap actually stuck -----------------
echo "[STEP] Verifying plans equality (Task006 -> $TASK_DIRNAME)"
python - "$PLANS_SRC" "$DST_PLANS" <<'PY'
import sys, pickle, hashlib
try:
    import numpy as np
except Exception:
    np = None

src, dst = sys.argv[1], sys.argv[2]

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

# Strongest check: raw file hashes
sha_src, sha_dst = sha(src), sha(dst)
print("SRC sha256:", sha_src)
print("DST sha256:", sha_dst)
if sha_src == sha_dst:
    print("STRUCT_EQUAL: True (byte-identical)")
    sys.exit(0)

# Fallback: recursive, NumPy-safe structural equality
s = pickle.load(open(src, 'rb'))
d = pickle.load(open(dst, 'rb'))

def eq(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(eq(x, y) for x, y in zip(a, b))
    if np is not None and isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b)
    return a == b

same = eq(s, d)
print("STRUCT_EQUAL:", same)
for k in ("num_stages", "plans_per_stage"):
    print(f"SRC[{k}] =", s.get(k))
    print(f"DST[{k}] =", d.get(k))
if not same:
    sys.stderr.write("Plans differ after copy; refusing to proceed. Check paths and rerun.\n")
    sys.exit(2)
PY

# ----------------- Copy pretrained locally -----------------
PRE_DIR="$WORKROOT/pretrained/Task006_CT2PET_fold0"
mkdir -p "$PRE_DIR"
cp -v "$PRETRAINED_SRC" "$PRE_DIR/"

# enforce top-level plans symlink for this task
bash /platforms/radiomics/NanoMask/finetuning_aug/00_assert_strict_plans.sh "$TASK_DIRNAME" || exit 1

# ----------------- Persist state for training scripts -----------------
STATE_FILE="$WORKROOT/.last_task"
{
  echo "TASK_ID=$TASK_ID"
  echo "TASK_DIRNAME=$TASK_DIRNAME"
  echo "PRETRAINED=$PRE_DIR/model_final_checkpoint.model"
} > "$STATE_FILE"

echo
echo "[READY] $TASK_DIRNAME prepared with STRICT Task006 plans."
echo "State: $STATE_FILE"
echo "Train with:"
echo "  bash $WORKROOT/03_train.sh"
