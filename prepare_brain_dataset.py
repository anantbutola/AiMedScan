import argparse
import os
import random
from typing import List, Tuple

import cv2
import h5py
import nibabel as nib
import numpy as np


def normalize_slice(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    min_v = np.min(img)
    max_v = np.max(img)
    if max_v - min_v < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - min_v) / (max_v - min_v)
    img = (img * 255.0).clip(0, 255)
    return img.astype(np.uint8)


def extract_slices(volume: np.ndarray, count: int) -> List[np.ndarray]:
    if volume.ndim != 3:
        raise ValueError("Expected 3D volume.")
    center = volume.shape[2] // 2
    half = count // 2
    indices = list(range(center - half, center + half + 1))
    indices = [i for i in indices if 0 <= i < volume.shape[2]]
    slices = [normalize_slice(volume[:, :, i]) for i in indices]
    return slices


def find_brats_file(subject_dir: str, modality: str) -> str:
    for fname in os.listdir(subject_dir):
        if fname.endswith(f"_{modality}.nii.gz"):
            return os.path.join(subject_dir, fname)
    raise FileNotFoundError(f"Missing modality {modality} in {subject_dir}")


def find_brats_nifti_files(brats_dir: str, modality: str) -> List[str]:
    files = []
    modality = modality.lower()
    for root, _, filenames in os.walk(brats_dir):
        for fname in filenames:
            lower = fname.lower()
            if (lower.endswith(f"_{modality}.nii.gz") or lower.endswith(f"_{modality}.nii")):
                files.append(os.path.join(root, fname))
    return files


def find_brats_h5_files(brats_dir: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(brats_dir):
        for fname in filenames:
            if fname.lower().endswith(".h5"):
                files.append(os.path.join(root, fname))
    return files


def find_ixi_files(ixi_dir: str, modality: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(ixi_dir):
        for fname in filenames:
            lower = fname.lower()
            if (lower.endswith(".nii.gz") or lower.endswith(".nii")) and modality.lower() in lower:
                files.append(os.path.join(root, fname))
    return files


def save_slices(slices: List[np.ndarray], out_dir: str, prefix: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for idx, slc in enumerate(slices):
        path = os.path.join(out_dir, f"{prefix}_{idx:03d}.png")
        cv2.imwrite(path, slc)


def split_paths(paths: List[str], val_split: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    val_count = int(len(paths) * val_split)
    return paths[val_count:], paths[:val_count]


def stem_without_nifti_suffix(path: str) -> str:
    base = os.path.basename(path)
    if base.lower().endswith(".nii.gz"):
        return base[:-7]
    return os.path.splitext(base)[0]


def load_brats_h5_slice(path: str, modality_index: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "image" not in f:
            raise KeyError(f"Missing 'image' dataset in {path}")
        image = f["image"][()]
    if image.ndim != 3:
        raise ValueError(f"Expected H5 'image' with shape (H, W, C), got {image.shape} in {path}")
    if not (0 <= modality_index < image.shape[2]):
        raise IndexError(
            f"Invalid BraTS H5 modality index {modality_index} for {path}; "
            f"available channels: 0..{image.shape[2] - 1}"
        )
    return normalize_slice(image[:, :, modality_index])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BraTS+IXI slices for brain classifier.")
    parser.add_argument("--brats-dir", required=True, help="Path to BraTS training dataset root.")
    parser.add_argument("--ixi-dir", required=True, help="Path to IXI dataset root.")
    parser.add_argument("--out-dir", required=True, help="Output directory for prepared slices.")
    parser.add_argument("--brats-modality", default="flair", help="BraTS modality (flair, t1, t1ce, t2).")
    parser.add_argument(
        "--brats-h5-modality-index",
        type=int,
        default=-1,
        help=(
            "Channel index to use when BraTS is in H5 slice format (image shape HxWxC). "
            "Default uses flair=0, t1=1, t1ce=2, t2=3 based on --brats-modality."
        ),
    )
    parser.add_argument("--ixi-modality", default="t1", help="IXI modality substring (t1 or t2).")
    parser.add_argument("--slices-per-volume", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-brats", type=int, default=0, help="Limit BraTS subjects (0 = full).")
    parser.add_argument("--max-ixi", type=int, default=0, help="Limit IXI volumes (0 = full).")
    args = parser.parse_args()

    if not os.path.isdir(args.brats_dir):
        raise FileNotFoundError(f"BraTS directory not found: {args.brats_dir}")
    if not os.path.isdir(args.ixi_dir):
        raise FileNotFoundError(f"IXI directory not found: {args.ixi_dir}")

    brats_nifti_files = find_brats_nifti_files(args.brats_dir, args.brats_modality)
    brats_h5_files = find_brats_h5_files(args.brats_dir)
    brats_source = "nifti" if brats_nifti_files else "h5" if brats_h5_files else "none"
    brats_items = brats_nifti_files if brats_source == "nifti" else brats_h5_files
    if not brats_items:
        raise FileNotFoundError(
            f"No BraTS files found under {args.brats_dir}. "
            "Expected either BraTS NIfTI files (e.g., *_flair.nii.gz) or H5 slice files."
        )
    if args.max_brats > 0:
        brats_items = brats_items[: args.max_brats]

    ixi_files = find_ixi_files(args.ixi_dir, args.ixi_modality)
    if not ixi_files:
        raise FileNotFoundError(
            f"No IXI NIfTI files found under {args.ixi_dir} for modality '{args.ixi_modality}'. "
            "This script does not download IXI automatically. Download and extract IXI first, then point "
            "--ixi-dir to the extracted folder containing .nii or .nii.gz files."
        )
    if args.max_ixi > 0:
        ixi_files = ixi_files[: args.max_ixi]

    brats_train, brats_val = split_paths(brats_items, args.val_split, args.seed)
    ixi_train, ixi_val = split_paths(ixi_files, args.val_split, args.seed)

    modality_to_index = {"flair": 0, "t1": 1, "t1ce": 2, "t2": 3}
    h5_modality_index = (
        args.brats_h5_modality_index
        if args.brats_h5_modality_index >= 0
        else modality_to_index.get(args.brats_modality.lower(), 0)
    )

    for split, items in [("train", brats_train), ("val", brats_val)]:
        out_dir = os.path.join(args.out_dir, split, "positive")
        for item_path in items:
            if brats_source == "h5":
                slc = load_brats_h5_slice(item_path, h5_modality_index)
                slices = [slc]
                prefix = stem_without_nifti_suffix(item_path)
            else:
                vol = nib.load(item_path).get_fdata()
                slices = extract_slices(vol, args.slices_per_volume)
                prefix = stem_without_nifti_suffix(item_path)
            save_slices(slices, out_dir, prefix)

    for split, files in [("train", ixi_train), ("val", ixi_val)]:
        out_dir = os.path.join(args.out_dir, split, "negative")
        for file_path in files:
            vol = nib.load(file_path).get_fdata()
            slices = extract_slices(vol, args.slices_per_volume)
            prefix = stem_without_nifti_suffix(file_path)
            save_slices(slices, out_dir, prefix)

    print(f"Prepared dataset at {args.out_dir}")


if __name__ == "__main__":
    main()
