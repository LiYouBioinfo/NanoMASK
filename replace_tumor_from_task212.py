#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Replace the tumor label in a multi-class segmentation (Task006) with the tumor
predicted by Task213 (usually a binary mask).

Usage:
  python replace_tumor_from_task213.py \
      --seg006 /path/to/output/CaseID/task006/CaseID.nii.gz \
      --seg213 /path/to/output/CaseID/task213/CaseID.nii.gz \
      --out    /path/to/output/CaseID/CaseID_seg_final.nii.gz \
      --tumor_label 6 \
      [--fallback_keep_006_if_213_empty 0]

Default tumor_label is 1.
"""

import argparse, sys
import numpy as np
import nibabel as nib

def load_nii(p):
    img = nib.load(p)
    arr = np.asarray(img.dataobj)
    return img, arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg006", required=True, help="Task006 prediction .nii.gz (multi-class)")
    ap.add_argument("--seg213", required=True, help="Task213 tumor .nii.gz (binary or multi-class with tumor>0)")
    ap.add_argument("--out",    required=True, help="Output path for merged segmentation .nii.gz")
    ap.add_argument("--tumor_label", type=int, default=6, help="Numeric label id for tumor in Task006 space (default: 6)")
    ap.add_argument("--fallback_keep_006_if_213_empty", type=int, default=0,
                    help="If 6 and Task213 tumor is empty, keep Task006 tumor as-is. Default 0 = remove tumor when 213 is empty.")
    args = ap.parse_args()

    img006, seg006 = load_nii(args.seg006)
    img213, seg213 = load_nii(args.seg213)

    # Basic checks
    if seg006.shape != seg213.shape:
        print(f"[ERROR] shape mismatch: 006 {seg006.shape} vs 213 {seg213.shape}", file=sys.stderr)
        sys.exit(2)
    if not np.allclose(img006.affine, img213.affine, atol=1e-5):
        print("[WARN] affines differ slightly; proceeding but ensure both tasks were run on the same staged inputs.", file=sys.stderr)

    # Ensure integer types
    seg006 = seg006.astype(np.int16, copy=False)

    # Create replacement mask from Task213 (binary or multi)
    tumor213_mask = (seg213 > 0)

    # If Task213 is empty, decide fallback behavior
    if not tumor213_mask.any():
        if args.fallback_keep_006_if_213_empty:
            print("[INFO] Task213 tumor empty -> keeping Task006 tumor unchanged (fallback enabled).")
            out_arr = seg006
        else:
            print("[INFO] Task213 tumor empty -> removing tumor label from Task006 (fallback disabled).")
            out_arr = seg006.copy()
            out_arr[seg006 == args.tumor_label] = 0
    else:
        # Replace: remove tumor from Task006, then write Task213 tumor as the tumor label
        out_arr = seg006.copy()
        out_arr[out_arr == args.tumor_label] = 0
        out_arr[tumor213_mask] = args.tumor_label
        print(f"[OK] Replaced tumor label {args.tumor_label} using Task213 mask (vox={int(tumor213_mask.sum())}).")

    nib.save(nib.Nifti1Image(out_arr.astype(np.int16), img006.affine, img006.header), args.out)
    print(f"[DONE] Wrote merged segmentation -> {args.out}")

if __name__ == "__main__":
    main()
