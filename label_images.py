"""
label_images.py
Simple offline labelling tool: drag a box around the ball in each image.
Labels are saved in YOLO format to dataset/labels_all/<image-name>.txt
(your images never leave your PC).

Mouse:  LEFT-DRAG = draw a tight box around the ball
Keys:   ENTER / SPACE = save the box and go to next image
        N = NO ball visible in this image (saved as an empty label)
        R = clear the current box
        B = go back to the previous image
        X = reject this image (bad/blurred/ball unclear) -> moved to dataset/rejected
        Q = quit (progress is saved; run again to resume)
"""

import cv2
import glob
import os
import shutil

RAW_DIR = os.path.join("dataset", "raw")
LABEL_DIR = os.path.join("dataset", "labels_all")
REJECT_DIR = os.path.join("dataset", "rejected")
MAX_W, MAX_H = 1600, 850      # display size limit (image is scaled to fit)
WIN = "Label tennis ball"

state = {"start": None, "end": None, "drawing": False}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["start"] = (x, y)
        state["end"] = (x, y)
        state["drawing"] = True
    elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
        state["end"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
        state["end"] = (x, y)
        state["drawing"] = False


def label_path(img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(LABEL_DIR, stem + ".txt")


def box_valid():
    if state["start"] is None or state["end"] is None:
        return False
    return (abs(state["end"][0] - state["start"][0]) >= 3 and
            abs(state["end"][1] - state["start"][1]) >= 3)


def clear_box():
    state["start"] = None
    state["end"] = None
    state["drawing"] = False


def load_existing(path, dw, dh):
    """Show a previously saved box when stepping back to an image."""
    clear_box()
    if not os.path.exists(path):
        return
    with open(path) as f:
        line = f.readline().strip()
    if not line:
        return
    _, cx, cy, bw, bh = [float(v) for v in line.split()]
    state["start"] = (int((cx - bw / 2) * dw), int((cy - bh / 2) * dh))
    state["end"] = (int((cx + bw / 2) * dw), int((cy + bh / 2) * dh))


def save_label(path, dw, dh, has_ball):
    if not has_ball:
        open(path, "w").close()          # empty file = no ball
        return
    x1, x2 = sorted([state["start"][0], state["end"][0]])
    y1, y2 = sorted([state["start"][1], state["end"][1]])
    x1, x2 = max(0, x1), min(dw, x2)
    y1, y2 = max(0, y1), min(dh, y2)
    cx, cy = (x1 + x2) / 2 / dw, (y1 + y2) / 2 / dh
    bw, bh = (x2 - x1) / dw, (y2 - y1) / dh
    with open(path, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def main():
    os.makedirs(LABEL_DIR, exist_ok=True)
    os.makedirs(REJECT_DIR, exist_ok=True)

    images = sorted(glob.glob(os.path.join(RAW_DIR, "*.jpg")))
    if not images:
        print("No images found in", RAW_DIR)
        return

    # resume: start at first image without a label file
    idx = 0
    while idx < len(images) and os.path.exists(label_path(images[idx])):
        idx += 1
    if idx >= len(images):
        print("All images are already labelled.")
        return

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    while 0 <= idx < len(images):
        img_path = images[idx]
        if not os.path.exists(img_path):      # was rejected earlier
            idx += 1
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        disp = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img
        dh, dw = disp.shape[:2]

        load_existing(label_path(img_path), dw, dh)
        action = None

        while action is None:
            view = disp.copy()
            if state["start"] is not None and state["end"] is not None:
                cv2.rectangle(view, state["start"], state["end"], (0, 255, 0), 1)

            done = len(glob.glob(os.path.join(LABEL_DIR, "*.txt")))
            cv2.rectangle(view, (0, 0), (dw, 28), (0, 0, 0), -1)
            cv2.putText(view,
                        f"[{idx + 1}/{len(images)}] labelled: {done} | "
                        "drag box + ENTER | N=no ball | R=reset | B=back | X=reject | Q=quit",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow(WIN, view)

            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32):                       # Enter / Space
                if box_valid():
                    save_label(label_path(img_path), dw, dh, True)
                    action = "next"
                else:
                    print("Draw a box around the ball first (or press N for no ball).")
            elif key == ord("n"):
                save_label(label_path(img_path), dw, dh, False)
                action = "next"
            elif key == ord("r"):
                clear_box()
            elif key == ord("b"):
                action = "back"
            elif key == ord("x"):
                shutil.move(img_path, os.path.join(REJECT_DIR, os.path.basename(img_path)))
                lp = label_path(img_path)
                if os.path.exists(lp):
                    os.remove(lp)
                action = "next"
            elif key == ord("q"):
                action = "quit"

        if action == "quit":
            break
        idx = max(0, idx - 1) if action == "back" else idx + 1

    cv2.destroyAllWindows()
    print("Labels saved in", LABEL_DIR)
    print("Next step: python prepare_dataset.py")


if __name__ == "__main__":
    main()