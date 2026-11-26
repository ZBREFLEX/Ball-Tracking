"""
train_yolo.py
Fine-tune a pretrained YOLO26 model on your labelled tennis-ball images.

DATA_YAML = the data.yaml from your exported dataset (YOLO format).
"""

from ultralytics import YOLO
import torch

DATA_YAML = r"E:\ball project\TennisBallPath\dataset\yolo\data.yaml"
PROJECT_DIR = r"E:\ball project\TennisBallPath\runs"

MODEL = r"E:\ball project\TennisBallPath\runs\ball-4\weights\best.pt"   # continue from the first model  # use "yolo26n.pt" if you run out of GPU memory
IMG_SIZE = 960           # keep high: the ball is small in a wide CCTV view
BATCH = 4                # 4 GB VRAM. Lower to 2 if you get "CUDA out of memory"
EPOCHS = 30


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
        patience=10,        # stop early if no improvement
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