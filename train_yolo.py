"""
train_yolo.py
Fine-tune a pretrained YOLO26 model on your labelled tennis-ball images.

DATA_YAML = the data.yaml from your exported dataset (YOLO format).
"""

from pathlib import Path

import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent
DATA_YAML = ROOT_DIR / "dataset" / "yolo" / "data.yaml"
PROJECT_DIR = ROOT_DIR / "runs"

MODEL = ROOT_DIR / "yolo26s.pt"  # start from the available pretrained weights
IMG_SIZE = 640           # keep high: the ball is small in a wide CCTV view
BATCH = 8                # 4 GB VRAM. Lower to 2 if you get "CUDA out of memory"
EPOCHS = 100


def main():
    if not torch.cuda.is_available():
        print("WARNING: CUDA GPU not available - training would be very slow on CPU.")
        print("Install the CUDA build of PyTorch first (see pytorch.org).")
        return

    model = YOLO(MODEL)
    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        workers=2,
        cache="ram",          # Windows: keep low
        device=0,
        patience=5,        # stop early if no improvement
        project=PROJECT_DIR,
        name="ball",
        # augmentations suited to a fixed CCTV view and a small ball
        fliplr=0.5,
        degrees=0.0,
        scale=0.3,
        mosaic=1.0,
    )

    print("\nDone. Best weights are inside:", PROJECT_DIR, "\\ball\\weights\\best.pt")


if __name__ == "__main__":      # required on Windows
    main()