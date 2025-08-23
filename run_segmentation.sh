#!/usr/bin/env bash
###############################################################################
# run_segmentation.sh – End-to-end mouse PET/CT preprocessing + nnUNet inference
#
# Workflow
#   0.  Activate project-local .venv
#   1.  Parse CLI arguments:  -ct <CT-DICOM-dir>  -pet <PET-DICOM-dir>  -id <caseID>
#   2.  Copy raw DICOMs into a scratch workspace
#   3.  Convert BOTH series ➜ CT_0000.nii.gz / PET_0001.nii.gz
#   4.  Affine-register CT → PET  (Greedy; produces CT_on_PET.nii.gz)
#   5.  Stage CT_0000.nii.gz / PET_0001.nii.gz for nnUNet, print summary
#   6.  Clean tmp/  •  run nnUNet_predict
#
# Example
#   ./run_segmentation.sh -ct /data/CT  -pet /data/PET  -id Case42
###############################################################################
set -euo pipefail

# ──────────────────────────────────────────────────────────────── log helpers
STEP() { echo -e "\033[1;34m[STEP]\033[0m $*"; }
OK()   { echo -e "\033[1;32m[ OK ]\033[0m $*"; }
FAIL() { echo -e "\033[1;31m[FAIL]\033[0m $*"; }

# ──────────────────────────────────────────────────────────────── 0. activate venv
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

# ──────────────────────────────────────────────────────────────── 1. CLI parsing
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
    *)  echo "Unknown option $1"; usage; exit 1 ;;
  esac
done
if [[ -z "$CT_PATH_DCM" || -z "$PET_PATH_DCM" || -z "$INSTANCE_ID" ]]; then
  FAIL "-ct -pet -id are mandatory"; usage; exit 1
fi
STEP "=== CT  DICOM dir : $CT_PATH_DCM"
STEP "=== PET DICOM dir : $PET_PATH_DCM"
STEP "=== Instance ID   : $INSTANCE_ID"

# ──────────────────────────────────────────────────────────────── 2. workspace
OUTPUT_DIR="/platforms/radiomics/NanoMask/output/${INSTANCE_ID}"
nnUNET_DIR="$OUTPUT_DIR"
nnUNET_TMP_DIR="$OUTPUT_DIR/tmp"
nnUNET_TMP_CT_DIR="$nnUNET_TMP_DIR/CT"
nnUNET_TMP_PET_DIR="$nnUNET_TMP_DIR/PET"

STEP "=== Creating workspace $OUTPUT_DIR …"
mkdir -p "$nnUNET_DIR" "$nnUNET_TMP_CT_DIR" "$nnUNET_TMP_PET_DIR"
cp "$CT_PATH_DCM"/*.dcm  "$nnUNET_TMP_CT_DIR/"
cp "$PET_PATH_DCM"/*.dcm "$nnUNET_TMP_PET_DIR/"
OK  "DICOMs copied"

# ──────────────────────────────────────────────────────────────── 3. DICOM → NIfTI
nnUNET_TMP_CT="$nnUNET_TMP_DIR/CT_0000.nii.gz"
nnUNET_TMP_PET="$nnUNET_TMP_DIR/PET_0001.nii.gz"

STEP "=== Converting CT DICOMs → $nnUNET_TMP_CT …"
python "$SCRIPT_DIR/preprocessing/dcm2nii.py" \
       -i "$nnUNET_TMP_DIR" \
       -o "$nnUNET_TMP_DIR" \
       -a CT
OK "CT conversion done"

# create a blank organ mask (required by affine_registration.py)
if [[ ! -f "$nnUNET_TMP_DIR/organs-CT.nii.gz" ]]; then
  python - <<'PY' "$nnUNET_TMP_CT" "$nnUNET_TMP_DIR/organs-CT.nii.gz"
import SimpleITK as sitk, sys, os
src  = sys.argv[1]          # CT image
dst  = sys.argv[2]          # mask location
img  = sitk.ReadImage(src)
mask = sitk.Image(img.GetSize(), sitk.sitkUInt8)
mask.CopyInformation(img)
sitk.WriteImage(mask, dst, useCompression=True)
PY
fi

# ──────────────────────────────────────────────────────────────── 4. affine-registration
STEP "=== Affine-registering CT → PET (Greedy) …"
python "$SCRIPT_DIR/preprocessing/affine_registration.py" -i "$nnUNET_TMP_DIR" -m CT

# apply affine to CT volume
greedy -d 3 \
       -rf "$nnUNET_TMP_PET" \
       -ri LINEAR \
       -rm "$nnUNET_TMP_CT"  "$nnUNET_TMP_DIR/CT_on_PET.nii.gz" \
       -r  "$nnUNET_TMP_DIR/CT2PET.mat"

OK  "Affine registration complete"

# ──────────────────────────────────────────────────────────────── 5. stage for nnUNet
STEP "=== Staging final NIfTI pair for nnUNet …"
mv "$nnUNET_TMP_DIR/CT_on_PET.nii.gz" "$nnUNET_DIR/${INSTANCE_ID}_0000.nii.gz"
cp "$nnUNET_TMP_PET"                  "$nnUNET_DIR/${INSTANCE_ID}_0001.nii.gz"

echo "────────────  summary  ────────────"
python - <<'PY' "$nnUNET_DIR" "$INSTANCE_ID"
import SimpleITK as sitk, sys, os
p, case = sys.argv[1], sys.argv[2]
for ch in ("0000","0001"):
    f = os.path.join(p, f"{case}_{ch}.nii.gz")
    img = sitk.ReadImage(f)
    print(f"{os.path.basename(f):25s} size {img.GetSize()}  spacing {img.GetSpacing()}")
PY
echo "───────────────────────────────────"

# ──────────────────────────────────────────────────────────────── 6. cleanup tmp + nnUNet
STEP "=== Cleaning tmp/ …"; rm -rf "$nnUNET_TMP_DIR"; OK "tmp/ removed"

STEP "=== Running nnUNet_predict …"
cd "$SCRIPT_DIR/nnunet"
nnUNet_predict -i "$nnUNET_DIR" -o "$nnUNET_DIR" -m 3d_fullres -t 006 -f 0
OK "nnUNet prediction finished – results in $nnUNET_DIR"
