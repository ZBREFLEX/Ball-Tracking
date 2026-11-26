"""
Generate YOLO-format tennis-ball label proposals from dataset/raw.

Run `python auto_annotate.py --review --apply` to inspect predictions and
copy approved labels to dataset/labels_all. Drag a box to correct a prediction,
press N to confirm an empty frame, S to skip, or Q to stop.

Without --review, predictions are saved as proposals in dataset/auto_labels.
Use --apply to copy positive proposals to dataset/labels_all; empty predictions
are only applied with --include-empty. Existing labels are never overwritten.
If proposals already exist and need review, add --overwrite to regenerate them.
"""

import argparse
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "dataset" / "raw"
MODEL_PATH = ROOT / "runs" / "ball-4" / "weights" / "best.pt"
PROPOSAL_DIR = ROOT / "dataset" / "auto_labels"
TRAINING_LABEL_DIR = ROOT / "dataset" / "labels_all"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_DISPLAY_WIDTH = 1440
MAX_DISPLAY_HEIGHT = 900
WINDOW_NAME = "Review auto annotations"


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-annotate tennis-ball images.")
    parser.add_argument("--images", type=Path, default=IMAGE_DIR,
                        help="Folder containing input images.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH,
                        help="YOLO checkpoint with a tennis_ball or sports ball class.")
    parser.add_argument("--output", type=Path, default=PROPOSAL_DIR,
                        help="Folder for generated YOLO label proposals.")
    parser.add_argument("--training-labels", type=Path, default=TRAINING_LABEL_DIR,
                        help="Label folder used by prepare_dataset.py.")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Minimum prediction confidence (default: 0.25).")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="YOLO inference image size (default: 960).")
    parser.add_argument("--batch", type=int, default=1,
                        help="Inference batch size.")
    parser.add_argument("--device", choices=("auto", "cpu", "0"), default="auto",
                        help="Inference device (default: GPU when available, otherwise CPU).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N images; 0 means all.")
    parser.add_argument("--review", action="store_true",
                        help="Review or correct each prediction in an OpenCV window.")
    parser.add_argument("--apply", action="store_true",
                        help="Copy proposals to the training labels folder.")
    parser.add_argument("--include-empty", action="store_true",
                        help="With --apply, also apply empty predictions as no-ball labels.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing proposal files (never replaces training labels).")
    return parser.parse_args()


def class_ids_for_ball(model):
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    target_ids = [
        class_id for class_id, name in items
        if str(name).strip().lower().replace("_", " ") in ("tennis ball", "sports ball")
    ]
    if not target_ids:
        raise ValueError(
            "The model has no 'tennis_ball' or COCO 'sports ball' class. "
            f"Model classes: {names}"
        )
    return target_ids


def yolo_label(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None

    center_x = (x1 + x2) / (2 * width)
    center_y = (y1 + y2) / (2 * height)
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def predictions_from_result(result, target_class_ids):
    height, width = result.orig_shape
    boxes = []
    labels = []
    for prediction in result.boxes:
        class_id = int(prediction.cls[0].item())
        if class_id not in target_class_ids:
            continue
        coords = tuple(float(value) for value in prediction.xyxy[0].tolist())
        label = yolo_label(coords, width, height)
        if label is not None:
            boxes.append((*coords, float(prediction.conf[0].item())))
            labels.append(label)
    return boxes, labels


def mouse_callback(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["start"] = (x, y)
        state["end"] = (x, y)
        state["drawing"] = True
    elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
        state["end"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
        state["end"] = (x, y)
        state["drawing"] = False


def review_image(image, boxes, predicted_labels):
    height, width = image.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height, 1.0)
    display_size = (int(width * scale), int(height * scale))
    state = {"start": None, "end": None, "drawing": False}

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, state)
    try:
        while True:
            display = cv2.resize(image, display_size) if scale < 1.0 else image.copy()
            for x1, y1, x2, y2, confidence in boxes:
                cv2.rectangle(
                    display,
                    (int(x1 * scale), int(y1 * scale)),
                    (int(x2 * scale), int(y2 * scale)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display, f"ball {confidence:.2f}",
                    (int(x1 * scale), max(18, int(y1 * scale) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                )

            if state["start"] is not None and state["end"] is not None:
                cv2.rectangle(display, state["start"], state["end"], (255, 0, 0), 2)

            cv2.putText(
                display,
                "ENTER/Y=accept  drag=correct then ENTER  N=no ball  S=skip  R=clear  Q=quit",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10, ord("y")):
                if state["start"] is not None and state["end"] is not None:
                    x1, x2 = sorted((state["start"][0], state["end"][0]))
                    y1, y2 = sorted((state["start"][1], state["end"][1]))
                    manual = yolo_label(
                        (x1 / scale, y1 / scale, x2 / scale, y2 / scale),
                        width,
                        height,
                    )
                    if manual is not None:
                        return "accept", [manual]
                if predicted_labels:
                    return "accept", predicted_labels
            elif key == ord("n"):
                return "accept", []
            elif key == ord("s"):
                return "skip", []
            elif key == ord("r"):
                state.update(start=None, end=None, drawing=False)
            elif key == ord("q"):
                return "quit", []
    finally:
        cv2.destroyWindow(WINDOW_NAME)


def write_label(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def main():
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1.")
    if args.batch < 1:
        raise ValueError("--batch must be at least 1.")
    if args.imgsz < 1 or args.limit < 0:
        raise ValueError("--imgsz must be positive and --limit cannot be negative.")
    if not args.images.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {args.images}")

    all_images = sorted(
        path for path in args.images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    images = all_images
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise FileNotFoundError(f"No supported images found in: {args.images}")

    if not args.model.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {args.model}")

    import torch
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    target_class_ids = class_ids_for_ball(model)
    device = args.device
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}")
    print(f"Device: {device}; batch: {args.batch}; images: {len(images)}")
    print("Target model classes:", [model.names[class_id] for class_id in target_class_ids])

    results = model.predict(
        source=[str(path) for path in images],
        stream=True,
        imgsz=args.imgsz,
        conf=args.confidence,
        classes=target_class_ids,
        batch=args.batch,
        device=device,
        verbose=False,
    )

    image_by_name = {path.name: path for path in images}
    generated = positives = negatives = skipped = applied = 0
    try:
        for result in results:
            image_path = image_by_name.get(Path(result.path).name)
            if image_path is None:
                print(f"Model returned an unknown image path; skipping: {result.path}")
                skipped += 1
                continue

            proposal_path = args.output / f"{image_path.stem}.txt"
            if proposal_path.exists() and not args.overwrite:
                if args.apply and not args.review:
                    labels = [
                        line.strip()
                        for line in proposal_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    generated += 1
                    if labels:
                        positives += 1
                    else:
                        negatives += 1
                    if labels or args.include_empty:
                        training_path = args.training_labels / proposal_path.name
                        if training_path.exists():
                            print(f"Preserving existing training label: {training_path.name}")
                        else:
                            write_label(training_path, labels)
                            applied += 1
                    continue
                skipped += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not read image; skipping: {image_path}")
                skipped += 1
                continue

            boxes, labels = predictions_from_result(result, target_class_ids)
            reviewed = False
            if args.review:
                action, labels = review_image(image, boxes, labels)
                if action == "quit":
                    break
                if action == "skip":
                    skipped += 1
                    continue
                reviewed = True

            write_label(proposal_path, labels)
            generated += 1
            if labels:
                positives += 1
            else:
                negatives += 1

            should_apply = args.apply and (
                bool(labels) or reviewed or args.include_empty
            )
            if should_apply:
                training_path = args.training_labels / proposal_path.name
                if training_path.exists():
                    print(f"Preserving existing training label: {training_path.name}")
                else:
                    write_label(training_path, labels)
                    applied += 1

            if generated % 100 == 0:
                print(f"Processed {generated}/{len(images)} images...")
    finally:
        if args.review:
            cv2.destroyAllWindows()

    print(
        f"Finished: {generated} proposals ({positives} with boxes, "
        f"{negatives} empty), {applied} applied, {skipped} skipped."
    )
    print("Proposals:", args.output)
    if args.apply:
        print("Training labels:", args.training_labels)
    else:
        print("For reviewed training labels, use: --review --apply")


if __name__ == "__main__":
    main()
