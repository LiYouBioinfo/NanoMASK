#!/usr/bin/env python3
"""
Prepare nnU-Net training data from a curated batch directory whose subfolders encode
usable sides via suffix flags:

  batch20250807_Curated/
    s01234-L/      -> generate: s01234-L
    s04567-LM/     -> generate: s04567-L, s04567-M
    s08901/        -> skipped (no flags)
    s12345-R/      -> generate: s12345-R
    s24680-LMR/    -> generate: s24680-L, s24680-M, s24680-R

Inside each patient folder, we expect NIfTI files. By default, any file matching
--label-globs is treated as the label; all other NIfTI files are treated as image
modalities and assigned indices _0000, _0001, ...

Usage:
  python prep_from_curated_LMR.py \
    --curated /path/to/batch20250807_Curated \
    --dst /path/to/nnUNet_raw/Task202_CT2PET_FT_LMR \
    --task-name "Task202_CT2PET_FT_LMR" \
    --modality-names CT PET \
    --copy-mode symlink

If your labels have a custom name, set --label-globs (comma-separated globs, first match wins):
  --label-globs "*_label.nii.gz,label.nii.gz,*_mask.nii.gz,mask.nii.gz"

Notes:
- Plain folders without any of L/M/R flags are skipped.
- Folders with any combination (L, M, R) generate one case per present flag.
- Creates imagesTr/, labelsTr/, dataset.json. Test split is left empty.
- By default uses symlinks; use --copy-mode copy for physical copies or hardlink for hard links.
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

VALID_FLAGS = {"L", "M", "R"}

def parse_flags_from_dirname(name: str) -> Tuple[str, List[str]]:
    """
    Extract base ID and list of unique flags (subset of L,M,R) from a dirname like:
      s01234-LM  -> ("s01234", ["L","M"])
      s04567-R   -> ("s04567", ["R"])
      s08901     -> ("s08901", [])
    """
    m = re.match(r"^(?P<base>[^-]+)(?:-(?P<flags>[LMR]+))?$", name)
    if not m:
        return name, []
    base = m.group("base")
    flags = list(dict.fromkeys([c for c in (m.group("flags") or "") if c in VALID_FLAGS]))
    return base, flags

def find_label_file(case_dir: Path, label_globs: List[str]) -> Optional[Path]:
    for g in label_globs:
        for p in case_dir.glob(g):
            return p
    # Fallback: common names
    for g in ["label.nii.gz", "label.nii", "*_label.nii.gz", "*_label.nii", "mask.nii.gz", "mask.nii", "*_mask.nii.gz", "*_mask.nii"]:
        for p in case_dir.glob(g):
            return p
    return None

def list_nifti_files(case_dir: Path) -> List[Path]:
    return sorted(list(case_dir.glob("*.nii"))) + sorted(list(case_dir.glob("*.nii.gz")))

def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True, help="Path to curated batch folder (e.g., batch20250807_Curated)")
    ap.add_argument("--dst", required=True, help="Destination nnU-Net raw task folder to create")
    ap.add_argument("--task-name", required=True, help='Task name to embed in dataset.json (e.g., "Task202_CT2PET_FT_LMR")')
    ap.add_argument("--modality-names", nargs="*", default=None,
                    help="Names for modalities in dataset.json (order will match sorted image files). "
                         "Example: CT PET  (defaults to generic channels if omitted)")
    ap.add_argument("--label-globs", default="label.nii.gz,label.nii,*_label.nii.gz,*_label.nii,*_mask.nii.gz,*_mask.nii,mask.nii.gz,mask.nii",
                    help="Comma-separated glob patterns to detect label file within each case folder (first match wins).")
    ap.add_argument("--copy-mode", choices=["symlink", "hardlink", "copy"], default="symlink",
                    help="Materialization mode for files (default: symlink)")
    args = ap.parse_args()

    curated = Path(args.curated).resolve()
    dst = Path(args.dst).resolve()
    imagesTr = dst / "imagesTr"
    labelsTr = dst / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    label_globs = [g.strip() for g in args.label_globs.split(",") if g.strip()]
    # Scan curated subfolders
    subdirs = [p for p in sorted(curated.iterdir()) if p.is_dir()]

    training_entries = []
    num_cases_written = 0
    skipped_plain = 0

    for case_dir in subdirs:
        base, flags = parse_flags_from_dirname(case_dir.name)
        if not flags:
            skipped_plain += 1
            continue

        # Gather files
        label_file = find_label_file(case_dir, label_globs)
        all_nii = [p for p in list_nifti_files(case_dir)]
        image_files = [p for p in all_nii if p != label_file]
        if len(image_files) == 0:
            print(f"[WARN] No image NIfTI files in {case_dir}; skipping.")
            continue

        # Deterministic order for modalities
        image_files = sorted(image_files, key=lambda p: p.name)

        # For each requested flag, create a nnU-Net case
        for flag in flags:
            case_id = f"{base}-{flag}"
            # Emit image modalities as case_id_0000.nii.gz, _0001.nii.gz, ...
            for k, img in enumerate(image_files):
                ext = ".nii.gz" if img.name.endswith(".nii.gz") else ".nii"
                dst_img = imagesTr / f"{case_id}_{k:04d}{ext}"
                link_or_copy(img, dst_img, args.copy_mode)
                # Only add one representative image path to dataset.json per case (nnU-Net expects a list of modalities;
                # we’ll provide channels via dataset.json "modality" map; entries list a single image path)
                # nnU-Net's dataset.json training entries generally include *only one* image path template;
                # the framework infers other channels by suffix. We'll store channel 0 path here.
                if k == 0:
                    rep_image = f"./imagesTr/{dst_img.name}"

            # Label
            if label_file is None:
                print(f"[WARN] No label found in {case_dir}; creating unlabeled entry (still usable for semi/unsup).")
                lbl_path = None
            else:
                ext_l = ".nii.gz" if label_file.name.endswith(".nii.gz") else ".nii"
                dst_lbl = labelsTr / f"{case_id}{ext_l}"
                link_or_copy(label_file, dst_lbl, args.copy_mode)
                lbl_path = f"./labelsTr/{dst_lbl.name}"

            entry = {"image": rep_image}
            if lbl_path is not None:
                entry["label"] = lbl_path
            training_entries.append(entry)
            num_cases_written += 1

    # Build dataset.json
    # Modality map: if names provided, use them; else generate "channel_0", "channel_1", ...
    max_channels = None
    # Heuristic: infer channel count from one written case by counting files with same prefix
    sample_case = None
    for e in training_entries:
        # extract case prefix from image path
        m = re.search(r"imagesTr/(.+?)_0000\.nii(\.gz)?$", e["image"])
        if m:
            sample_case = m.group(1)
            break
    if sample_case:
        ch = 0
        while (imagesTr / f"{sample_case}_{ch:04d}.nii").exists() or (imagesTr / f"{sample_case}_{ch:04d}.nii.gz").exists():
            ch += 1
        max_channels = ch

    if args.modality_names:
        modality_names = args.modality_names
    else:
        modality_names = [f"channel_{i}" for i in range(max_channels or 1)]

    dataset = {
        "name": args.task-name if hasattr(args, "task-name") else args.task_name,  # safety for hyphenated arg
        "description": "Finetuning dataset expanded from curated side-flagged folders",
        "tensorImageSize": "4D",
        "reference": "",
        "licence": "unknown",
        "release": "1.0",
        "modality": {str(i): n for i, n in enumerate(modality_names)},
        "labels": {
            "0": "background",
            "1": "foreground"
        },
        "numTraining": len(training_entries),
        "numTest": 0,
        "training": training_entries,
        "test": []
    }

    dst_ds = dst / "dataset.json"
    with open(dst_ds, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote {num_cases_written} cases to {dst}")
    if skipped_plain:
        print(f"[INFO] Skipped {skipped_plain} plain folders without L/M/R flags.")
    print(f"[INFO] Example entry: {training_entries[0] if training_entries else 'NONE'}")

if __name__ == "__main__":
    main()
