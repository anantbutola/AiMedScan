# MedAI Scan (Multi-Modal Classifier)

This repo is a small Flask web app that:
- serves a dashboard UI (single-page app)
- accepts an uploaded image at `POST /gradcam`
- returns a label + confidence and a Grad-CAM heatmap overlay

It can run in **trained mode** per modality:
- **Chest X-Ray** (CheXpert 5-label)
- **Brain MRI** (BraTS + IXI, tumor vs normal)
- **Bone X-Ray** (MURA, abnormal vs normal)

If a modality model is missing, the API returns `model_unavailable` for that request.

> Note: This project is for education/demo. It is **not** a clinical diagnostic system.

## Setup (macOS)

Recommended: Python 3.11 (TensorFlow support is much better).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Seed the local sample dataset hub (optional):
```bash
python ingest_dataset.py
```

Run the server:
```bash
PORT=5050 python app.py
```

Open:
- http://127.0.0.1:5050/

## Train chest model (CheXpert 5-label)

1. Download the **CheXpert v1.0 (small)** dataset from Stanford (license required).
2. Unzip it so you have a folder containing `train.csv`, `valid.csv`, and `train/` images.

Then run:
```bash
python train_chexpert.py --data-dir /path/to/CheXpert-v1.0-small --out-dir models
```

Defaults:
- 5 labels: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion
- Uncertain labels treated as negative (U-Zeros)

Outputs:
- `models/chest_classifier.keras`
- `models/chest_classes.json`

Restart the server and it will automatically use the trained model.

## Train brain model (BraTS + IXI)

1. Download **BraTS** (tumor MRI) and **IXI** (healthy MRI) datasets.
   - IXI is **not** downloaded by this repository automatically; download/extract it yourself.
   - `--ixi-dir` must point to extracted files containing `.nii` or `.nii.gz` volumes.
   - BraTS can be either subject-level NIfTI files (`*_flair.nii.gz`, etc.) or pre-sliced `.h5` files with an `image` dataset.
2. Prepare a 2D slice dataset:

```bash
python prepare_brain_dataset.py \
  --brats-dir /path/to/BraTS \
  --ixi-dir /path/to/IXI \
  --out-dir brain_dataset
```

If your BraTS data is `.h5` with multi-channel images, set `--brats-h5-modality-index` when needed (default mapping: flair=0, t1=1, t1ce=2, t2=3).

3. Train the classifier:

```bash
python train_brain.py --data-dir brain_dataset --out-dir models
```

Outputs:
- `models/brain_classifier.keras`
- `models/brain_classes.json`

## Train bone model (MURA)

1. Download the **MURA v1.1** dataset.
2. Train:

```bash
python train_mura.py --data-dir /path/to/MURA-v1.1 --out-dir models
```

Outputs:
- `models/bone_classifier.keras`
- `models/bone_classes.json`

## API

- `GET /health` – readiness + model info
- `GET /metadata` – classes + input size
- `GET /list_dataset/<category>` – files for the dataset hub
- `POST /gradcam` – upload `multipart/form-data` with `image`
