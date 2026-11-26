"""
prepare_dataset.py
Builds the YOLO training folder from your labelled images:

    dataset/yolo/images/train, images/val
    dataset/yolo/labels/train, labels/val
    dataset/yolo/data.yaml

Split: images are sorted by capture time and cut into blocks of BLOCK
consecutive images; every VAL_EVERY-th block goes to validation. This keeps
near-identical neighbouring frames from landing in both train and val
(a plain random split would make the validation scores look better than they are).
"""

import glob
import os
import shutil

RAW_DIR = os.path.join("dataset", "raw")
LABEL_DIR = os.path.join("dataset", "labels_all")
OUT_DIR = os.path.join("dataset", "yolo")
BLOCK = 10        # consecutive images per block
VAL_EVERY = 5     # every 5th block -> validation (about 20%)


def stamp(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem.split("_")[1])
    except (IndexError, ValueError):
        return 0


def label_file(img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(LABEL_DIR, stem + ".txt")


def main():
    images = sorted(glob.glob(os.path.join(RAW_DIR, "*.jpg")),
                    key=lambda p: (stamp(p), p))
    labelled = [p for p in images if os.path.exists(label_file(p))]

    if not labelled:
        print("No labelled images found. Run label_images.py first.")
        return

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT_DIR, "images", split))
        os.makedirs(os.path.join(OUT_DIR, "labels", split))

    counts = {"train": 0, "val": 0}
    with_ball = {"train": 0, "val": 0}

    for i, img in enumerate(labelled):
        split = "val" if (i // BLOCK) % VAL_EVERY == VAL_EVERY - 1 else "train"
        lbl = label_file(img)

        shutil.copy(img, os.path.join(OUT_DIR, "images", split, os.path.basename(img)))
        shutil.copy(lbl, os.path.join(OUT_DIR, "labels", split, os.path.basename(lbl)))

        counts[split] += 1
        if os.path.getsize(lbl) > 0:
            with_ball[split] += 1

    root = os.path.abspath(OUT_DIR).replace("\\", "/")
    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write(f'path: "{root}"\n')
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write("  0: tennis_ball\n")

    print("Train images:", counts["train"], f"(with ball: {with_ball['train']})")
    print("Val images:  ", counts["val"], f"(with ball: {with_ball['val']})")
    print("data.yaml written to", os.path.join(OUT_DIR, "data.yaml"))
    print("Next step: python train_yolo.py")


if __name__ == "__main__":
    main()