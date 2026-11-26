"""
collect_frames.py
Capture full-resolution frames from BOTH cameras to build a YOLO training set.

Keys:  S = save one frame from each camera now
       A = turn auto-save on/off (one pair every AUTO_INTERVAL seconds)
       Q = quit

Tip: while auto-save is on, hold / throw / bounce / roll the ball in many
positions. Also move the ball out of view for a while to collect "no ball"
frames (they teach the model what NOT to detect).
"""

import cv2
import os
import time

from ball_tracker import CameraStream, CAMERAS   # reuses your threaded reader

SAVE_DIR = os.path.join("dataset", "raw")
AUTO_INTERVAL = 0.7        # seconds between auto-saved pairs
JPEG_QUALITY = 95
SHOW_WIDTH = 640


def save_frames(streams):
    saved = 0
    stamp = int(time.time() * 1000)
    for name, stream in streams.items():
        frame, _, _ = stream.read()
        if frame is None:
            continue
        prefix = name.lower().replace(" ", "")
        path = os.path.join(SAVE_DIR, f"{prefix}_{stamp}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        saved += 1
    return saved


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    streams = {}
    for name, url in CAMERAS.items():
        stream = CameraStream(name, url)
        if stream.opened:
            stream.start()
            streams[name] = stream
            print(name, "connected successfully")
        else:
            print(name, "connection failed")

    if not streams:
        return

    auto = False
    last_auto = 0.0
    total = 0

    try:
        while True:
            for name, stream in streams.items():
                frame, _, _ = stream.read()
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (SHOW_WIDTH, int(h * SHOW_WIDTH / w)))
                cv2.putText(small, f"saved: {total}  auto: {'ON' if auto else 'OFF'}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if auto else (255, 255, 255), 2)
                cv2.imshow(name, small)

            now = time.time()
            if auto and now - last_auto >= AUTO_INTERVAL:
                total += save_frames(streams)
                last_auto = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                total += save_frames(streams)
            elif key == ord("a"):
                auto = not auto
                last_auto = now
    except KeyboardInterrupt:
        pass

    for stream in streams.values():
        stream.stop()
    cv2.destroyAllWindows()
    print("Total images saved:", total, "->", SAVE_DIR)


if __name__ == "__main__":
    main()