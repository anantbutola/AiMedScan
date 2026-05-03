import argparse
import os
import shutil
from glob import glob
from typing import Dict


def parse_txt_record(path: str) -> Dict[str, str]:
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()
    return data


def find_image_for_record(clinical_dir: str, record_id: str):
    # look for a matching image in the same folder
    patterns = [f"scan_{record_id}.*", f"{record_id}.*"]
    for pat in patterns:
        for ext in ("png", "jpg", "jpeg", "bmp"):
            matches = glob(os.path.join(clinical_dir, pat.replace("*", f"*.{ext}")))
            if matches:
                return matches[0]
    # fallback: any image in folder
    for ext in ("png", "jpg", "jpeg", "bmp"):
        matches = glob(os.path.join(clinical_dir, f"*.{ext}"))
        if matches:
            return matches[0]
    return None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare knee dataset into class folders consumable by train_knee.py")
    parser.add_argument("--clinical-dir", default="clinical_dataset/Knee", help="Source clinical folder with .txt records and any sample images")
    parser.add_argument("--out-dir", default="knee_dataset", help="Output root where class folders (e.g. 0Normal) will be created")
    parser.add_argument(
        "--mapping",
        default=None,
        help=(
            "Optional JSON file mapping diagnosis strings to class tokens e.g. {\"Normal\":\"0Normal\",\"Pathological\":\"1Pathological\"}. "
        ),
    )
    parser.add_argument("--sample-image", default="datasets/xray_sample.png", help="Fallback sample image to copy when no image found")
    parser.add_argument("--use-sample-if-missing", action="store_true", help="Copy sample image when an image for a record is missing")
    args = parser.parse_args()

    clinical_dir = args.clinical_dir
    out_dir = args.out_dir
    ensure_dir(out_dir)

    # default mapping when none provided
    mapping = {"Normal": "0Normal", "Pathological": "1Pathological"}
    if args.mapping:
        import json

        with open(args.mapping, "r", encoding="utf-8") as f:
            mapping = json.load(f)

    txt_files = [p for p in glob(os.path.join(clinical_dir, "*.txt"))]
    if not txt_files:
        print(f"No .txt records found in {clinical_dir}")
        return

    counts: Dict[str, int] = {}
    copied = 0
    for txt in txt_files:
        rec = parse_txt_record(txt)
        record_id = rec.get("Record ID") or os.path.splitext(os.path.basename(txt))[0].split("_")[-1]
        diagnosis = rec.get("Diagnosis", "Unknown")
        class_token = mapping.get(diagnosis, None)
        if class_token is None:
            # create a fallback token numbered by unique diagnoses
            safe_name = diagnosis.replace(" ", "_") or "Unknown"
            class_token = f"9_{safe_name}"

        class_dir = os.path.join(out_dir, class_token)
        ensure_dir(class_dir)

        img_path = find_image_for_record(clinical_dir, record_id)
        if img_path:
            dst = os.path.join(class_dir, f"scan_{record_id}{os.path.splitext(img_path)[1]}")
            shutil.copy2(img_path, dst)
            copied += 1
        elif args.use_sample_if_missing and os.path.exists(args.sample_image):
            dst = os.path.join(class_dir, f"scan_{record_id}{os.path.splitext(args.sample_image)[1]}")
            shutil.copy2(args.sample_image, dst)
            copied += 1

        counts[class_token] = counts.get(class_token, 0) + 1

    print("Prepared knee dataset:")
    for k, v in counts.items():
        print(f" - {k}: {v} records")
    print(f"Total files copied: {copied}")
    print(f"Output root: {out_dir}")


if __name__ == "__main__":
    main()
