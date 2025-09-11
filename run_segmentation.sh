#!/usr/bin/env bash
###############################################################################
# run_segmentation.sh – End-to-end mouse PET/CT preprocessing + nnUNet inference
# (updated: Task213 CT-only tumor; isolates CT-only input to avoid modality mismatch)
###############################################################################
set -euo pipefail

STEP() { echo -e "\033[1;34m[STEP]\033[0m $*"; }
OK()   { echo -e "\033[1;32m[ OK ]\033[0m $*"; }
FAIL() { echo -e "\033[1;31m[FAIL]\033[0m $*"; }

# 0) activate venv
STEP "=== Activating virtual-env …"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1090
  . "$SCRIPT_DIR/.venv/bin/activate"
  OK "venv activated"
else
  FAIL "No virtual-env at $SCRIPT_DIR/.venv"
  exit 1
fi

# 1) CLI
usage() {
  cat <<EOF
Usage: $(basename "$0") -ct <CT DICOM dir> -pet <PET DICOM dir> -id <caseID>
Required
  -ct   Folder with CT DICOMs
  -pet  Folder with PET DICOMs
  -id   Case identifier (used for output folder & filenames)
EOF
}
CT_PATH_DCM=""; PET_PATH_DCM=""; INSTANCE_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -ct)  CT_PATH_DCM="$2";  shift 2 ;;
    -pet) PET_PATH_DCM="$2"; shift 2 ;;
    -id)  INSTANCE_ID="$2";  shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option $1"; usage; exit 1 ;;
  esac
done
[[ -n "$CT_PATH_DCM" && -n "$PET_PATH_DCM" && -n "$INSTANCE_ID" ]] || { FAIL "-ct -pet -id are mandatory"; usage; exit 1; }
STEP "=== CT  DICOM dir : $CT_PATH_DCM"
STEP "=== PET DICOM dir : $PET_PATH_DCM"
STEP "=== Instance ID   : $INSTANCE_ID"

# 2) workspace
OUTPUT_DIR="/platforms/radiomics/NanoMask/output/${INSTANCE_ID}"
nnUNET_DIR="$OUTPUT_DIR"                  # common staging (CT+PET) for multi-modal tasks
nnUNET_TMP_DIR="$OUTPUT_DIR/tmp"
nnUNET_TMP_CT_DIR="$nnUNET_TMP_DIR/CT"
nnUNET_TMP_PET_DIR="$nnUNET_TMP_DIR/PET"

STEP "=== Creating workspace $OUTPUT_DIR …"
mkdir -p "$nnUNET_DIR" "$nnUNET_TMP_CT_DIR" "$nnUNET_TMP_PET_DIR"
cp "$CT_PATH_DCM"/*.dcm  "$nnUNET_TMP_CT_DIR/"
cp "$PET_PATH_DCM"/*.dcm "$nnUNET_TMP_PET_DIR/"
OK  "DICOMs copied"

# 3) DICOM → NIfTI
nnUNET_TMP_CT="$nnUNET_TMP_DIR/CT_0000.nii.gz"
nnUNET_TMP_PET="$nnUNET_TMP_DIR/PET_0001.nii.gz"

STEP "=== Converting CT DICOMs → $nnUNET_TMP_CT …"
python "$SCRIPT_DIR/preprocessing/dcm2nii.py" -i "$nnUNET_TMP_DIR" -o "$nnUNET_TMP_DIR" -a CT
OK "CT conversion done"

# blank organ mask (for affine_registration.py)
if [[ ! -f "$nnUNET_TMP_DIR/organs-CT.nii.gz" ]]; then
python - "$nnUNET_TMP_CT" "$nnUNET_TMP_DIR/organs-CT.nii.gz" <<'PY'
import SimpleITK as sitk, sys
img  = sitk.ReadImage(sys.argv[1])
mask = sitk.Image(img.GetSize(), sitk.sitkUInt8); mask.CopyInformation(img)
sitk.WriteImage(mask, sys.argv[2], useCompression=True)
PY
fi

# 4) affine-registration CT → PET (Greedy)
STEP "=== Affine-registering CT → PET (Greedy) …"
python "$SCRIPT_DIR/preprocessing/affine_registration.py" -i "$nnUNET_TMP_DIR" -m CT
greedy -d 3 -rf "$nnUNET_TMP_PET" -ri LINEAR \
       -rm "$nnUNET_TMP_CT"  "$nnUNET_TMP_DIR/CT_on_PET.nii.gz" \
       -r  "$nnUNET_TMP_DIR/CT2PET.mat"
OK  "Affine registration complete"

# 5) stage (CT+PET) for multi-modal tasks
STEP "=== Staging final NIfTI pair for nnUNet …"
mv "$nnUNET_TMP_DIR/CT_on_PET.nii.gz" "$nnUNET_DIR/${INSTANCE_ID}_0000.nii.gz"
cp "$nnUNET_TMP_PET"                  "$nnUNET_DIR/${INSTANCE_ID}_0001.nii.gz"

echo "────────────  summary  ────────────"
python - "$nnUNET_DIR" "$INSTANCE_ID" <<'PY'
import SimpleITK as sitk, sys, os
p, case = sys.argv[1], sys.argv[2]
for ch in ("0000","0001"):
    f = os.path.join(p, f"{case}_{ch}.nii.gz")
    img = sitk.ReadImage(f); print(f"{os.path.basename(f):25s} size {img.GetSize()}  spacing {img.GetSpacing()}")
PY
echo "───────────────────────────────────"

# 6) cleanup tmp
STEP "=== Cleaning tmp/ …"; rm -rf "$nnUNET_TMP_DIR"; OK "tmp/ removed"

# ──────────────────────────────────────────────────────────────── NEW: CT-only view for Task213
# Create a CT-only imagesTs folder so Task213 never sees the PET channel.
TASK213_IN="$nnUNET_DIR/task213_imagesTs"
mkdir -p "$TASK213_IN"
# use symlink (or 'cp' if you prefer copying)
ln -sfn "$nnUNET_DIR/${INSTANCE_ID}_0000.nii.gz" "$TASK213_IN/${INSTANCE_ID}_0000.nii.gz"

# 7) Run Task006 (CT+PET, unchanged)
STEP "=== Running nnUNet_predict (Task006; CT+PET) …"
cd "$SCRIPT_DIR/nnunet"
TASK006_OUT="$nnUNET_DIR/task006"
mkdir -p "$TASK006_OUT"
nnUNet_predict -i "$nnUNET_DIR" -o "$TASK006_OUT" -m 3d_fullres -t 006 -f 0
OK "Task006 prediction finished – results in $TASK006_OUT"

# 8) Run Task213 (CT-only tumor; reads CT-only folder)
STEP "=== Running nnUNet_predict (Task213; CT-only) …"
TASK213_OUT="$nnUNET_DIR/task213"
mkdir -p "$TASK213_OUT"
nnUNet_predict -i "$TASK213_IN" -o "$TASK213_OUT" -m 3d_fullres -t 213 -f 0 1 2 3 -chk model_best --save_npz
OK "Task213 prediction finished – results in $TASK213_OUT"

# 9) Replace tumor label in Task006 with Task213 tumor
STEP "=== Replacing tumor label in Task006 with Task213 tumor …"
CASE_NII_BASENAME="${INSTANCE_ID}.nii.gz"
SEG006="$TASK006_OUT/$CASE_NII_BASENAME"
SEG213="$TASK213_OUT/$CASE_NII_BASENAME"
OUT_MERGED="$nnUNET_DIR/${INSTANCE_ID}_seg_final.nii.gz"

python "$SCRIPT_DIR/replace_tumor_from_task213.py" \
  --seg006 "$SEG006" \
  --seg213 "$SEG213" \
  --out    "$OUT_MERGED" \
  --tumor_label 1 \
  --fallback_keep_006_if_213_empty 0
OK "Final merged segmentation written to $OUT_MERGED"
