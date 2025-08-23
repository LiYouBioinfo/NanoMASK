#!/usr/bin/env python3
import argparse, json, os, re, sys, shutil
from pathlib import Path

FINETUNE_ROOT = Path("/platforms/radiomics/NanoMask/finetuning_aug")
SINGLES_ROOT  = Path("/platforms/radiomics/NanoMask/FinetuneData")

LABELS = {"0":"background","1":"Heart","2":"Lungs","3":"Liver","4":"Spleen","5":"Kidneys","6":"Tumor"}
MODALITY = {"0":"CT","1":"PET"}  # 0:CT, 1:PET

def find_triplet(dirpath: Path):
    tag = dirpath.name.split("-")[0]
    ct = list(dirpath.glob(f"{tag}_*_0000.nii.gz"))
    pt = list(dirpath.glob(f"{tag}_*_0001.nii.gz"))
    sg = [p for p in dirpath.glob(f"{tag}_*.nii.gz")
          if not (p.name.endswith("_0000.nii.gz") or p.name.endswith("_0001.nii.gz"))]
    if len(ct)==1 and len(pt)==1 and len(sg)==1:
        return ct[0], pt[0], sg[0]
    return None

def link_or_copy(src: Path, dst: Path, do_copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): dst.unlink()
    if do_copy: shutil.copy2(src, dst)
    else: os.symlink(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--singles-root", default=str(SINGLES_ROOT))
    ap.add_argument("--task-id", type=int, required=True)        # e.g. 207
    ap.add_argument("--task-name", default="CT2PET_FT")          # suffix after TaskXXX_
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    task_dirname = f"Task{args.task_id:03d}_{args.task_name}"
    task_root = FINETUNE_ROOT / "nnUNet_data/nnUNet_raw_data" / task_dirname
    imagesTr = task_root/"imagesTr"
    labelsTr = task_root/"labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    singles_root = Path(args.singles_root)
    candidates = [d for d in singles_root.iterdir() if d.is_dir() and re.search(r"-[LRM]+$", d.name)]
    candidates.sort()

    mapping, idx = [], 1
    for d in candidates:
        trip = find_triplet(d)
        if not trip: 
            continue
        ct, pt, seg = trip
        case_id = f"case{idx}"
        link_or_copy(ct, imagesTr/f"{case_id}_0000.nii.gz", args.copy)
        link_or_copy(pt, imagesTr/f"{case_id}_0001.nii.gz", args.copy)
        link_or_copy(seg, labelsTr/f"{case_id}.nii.gz", args.copy)
        mapping.append((case_id, d.name))
        idx += 1

    if not mapping:
        print("No valid single-mouse triplets found in FinetuneData.", file=sys.stderr)
        sys.exit(2)

    dataset = {
        "name": task_dirname,
        "description": "CT->PET finetune on mouse singles",
        "tensorImageSize": "4D",
        "reference": "internal", "licence": "internal", "release": "0.0",
        "modality": MODALITY, "labels": LABELS,
        "numTraining": len(mapping), "numTest": 0, "training": [], "test": []
    }
    for case_id, _orig in mapping:
        dataset["training"].append({
            "image": f"./imagesTr/{case_id}.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz",
            "case_id": case_id
        })

    (task_root/"dataset.json").write_text(json.dumps(dataset, indent=2))
    with open(task_root/"caseID.txt", "w") as f:
        for cid, orig in mapping:
            f.write(f"{cid}\t{orig}\n")

    print(f"[OK] Staged {len(mapping)} cases at {task_root}")
    print("  imagesTr: only *_0000.nii.gz and *_0001.nii.gz (no plain caseX.nii.gz).")
    print("  labelsTr: caseX.nii.gz")
    print("  dataset.json uses './imagesTr/caseX.nii.gz' strings")

if __name__ == "__main__":
    main()
