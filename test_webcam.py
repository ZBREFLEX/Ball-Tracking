"""
test_webcam.py
Shows a live preview for each camera index so you can see which one is
your real webcam. Press Q in the preview window to move to the next index.
"""

import cv2
import time

INDEXES = [0, 1]

for i in INDEXES:
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"Index {i}: could not open")
        continue

    print(f"Index {i}: opened. Press Q in the window to continue.")
    shown = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            if time.time() - start > 5:
                print(f"Index {i}: opened but gives NO frames (wrong device or in use)")
                break
            continue

        shown += 1
        if shown == 1:
            print(f"Index {i}: frame size {frame.shape[1]}x{frame.shape[0]}")

        cv2.putText(frame, f"index {i}  (press Q)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"camera index {i}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()