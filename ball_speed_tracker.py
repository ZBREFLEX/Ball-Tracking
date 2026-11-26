"""
ball_speed_tracker.py
Detects the ball (colour or YOLO, chosen in ball_tracker.py), draws its path
from DETECTED positions only (no Kalman prediction) and shows its speed.

Speed method (2D, single camera):
  metres-per-pixel = real ball diameter / ball diameter in pixels
  speed = distance between two detections (pixels) * metres-per-pixel / time between them
This is the speed ACROSS the image at the ball's distance. Movement toward or
away from the camera is not measured (that needs the two-camera 3D step).

Keys:   Q = quit   M = show/hide mask   C = clear trail   R = reset peak speed
Mouse:  LEFT-CLICK on the ball = auto-set the colour range (colour detector)
"""

import cv2
import numpy as np
import time
from collections import deque

# detector, camera reader and settings window are reused from ball_tracker.py
from ball_tracker import (CAMERAS, PROC_WIDTH, USE_YOLO, CameraStream,
                          create_settings_window, detect_candidates,
                          detect_yolo, on_mouse)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
ACTIVE_CAMERAS = ["Camera 1"]     # ["Camera 1"] at the office

BALL_DIAMETER_M = 0.067         # tennis ball is about 6.5 - 6.9 cm
TRAIL_SECONDS = 2.0             # how long the path stays on screen
MAX_GAP_S = 0.25                # no detection for this long = path is cut
MIN_DT_S = 0.004                # ignore pairs closer in time than this
MAX_SPEED_MS = 70.0             # reject impossible values (about 250 km/h)
SMOOTH_PAIRS = 3                # median over the last N speed values
PEAK_HOLD_S = 5.0               # how long the peak speed stays displayed
SHOW_DOTS = True                # draw a dot at every detection


# ----------------------------------------------------------------------------
# TRACKER: detections only, no prediction
# ----------------------------------------------------------------------------
class SpeedTracker:
    def __init__(self):
        self.points = deque()                    # (x, y, t); x=None marks a cut
        self.diams = deque(maxlen=7)             # recent ball diameters (px)
        self.pair_speeds = deque(maxlen=SMOOTH_PAIRS)
        self.speed = None                        # m/s, smoothed
        self.peak = 0.0
        self.peak_t = 0.0
        self.last_t = None
        self.radius = 8
        self.diam_px = None
        self.last_hsv = None                     # for click-to-calibrate

    def clear_trail(self):
        self.points.clear()

    def reset_peak(self):
        self.peak = 0.0

    def _cut(self, t):
        self.pair_speeds.clear()
        self.speed = None
        if self.points and self.points[-1][0] is not None:
            self.points.append((None, None, t))

    def update(self, candidates, t):
        """Returns (status, (x, y) or None)."""
        if not candidates:
            if self.last_t is not None and t - self.last_t > MAX_GAP_S:
                self._cut(t)
            self._purge(t)
            return "SEARCHING", None

        c = max(candidates, key=lambda c: c["score"])
        x, y = c["x"], c["y"]
        self.diams.append(c.get("d", 2 * c["r"]))
        self.radius = c["r"]
        self.diam_px = float(np.median(self.diams))

        if self.points and self.points[-1][0] is not None:
            px, py, pt = self.points[-1]
            dt = t - pt
            if dt > MAX_GAP_S:
                self._cut(t)
            elif dt >= MIN_DT_S:
                m_per_px = BALL_DIAMETER_M / max(self.diam_px, 1.0)
                v = np.hypot(x - px, y - py) * m_per_px / dt
                if v <= MAX_SPEED_MS:
                    self.pair_speeds.append(v)
                    self.speed = float(np.median(self.pair_speeds))
                    if self.speed > self.peak or t - self.peak_t > PEAK_HOLD_S:
                        self.peak = self.speed
                        self.peak_t = t

        self.points.append((x, y, t))
        self.last_t = t
        self._purge(t)
        return "DETECTED", (x, y)

    def _purge(self, t):
        while self.points and t - self.points[0][2] > TRAIL_SECONDS:
            self.points.popleft()

    def peak_visible(self, t):
        return self.peak > 0 and t - self.peak_t <= PEAK_HOLD_S


# ----------------------------------------------------------------------------
# DRAWING
# ----------------------------------------------------------------------------
def put_text(img, text, org, color=(255, 255, 255), scale=0.7):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_trail(img, points):
    n = len(points)
    for i in range(1, n):
        a, b = points[i - 1], points[i]
        if a[0] is None or b[0] is None:
            continue
        thickness = max(1, int(1 + 4 * i / n))
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 (0, 0, 255), thickness, cv2.LINE_AA)
    if SHOW_DOTS:
        for p in points:
            if p[0] is not None:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    streams, trackers, stats = {}, {}, {}

    for name in ACTIVE_CAMERAS:
        stream = CameraStream(name, CAMERAS[name])
        if not stream.opened:
            print(name, "connection failed")
            continue
        print(name, "connected successfully")
        stream.start()
        streams[name] = stream
        trackers[name] = SpeedTracker()
        stats[name] = {"t0": time.time(), "proc": 0, "fid0": 0,
                       "proc_fps": 0.0, "cam_fps": 0.0}

    if not streams:
        print("No cameras connected.")
        return

    create_settings_window()
    for name in streams:
        cv2.namedWindow(name)
        cv2.setMouseCallback(name, on_mouse, trackers[name])

    last_ids = {name: 0 for name in streams}
    show_mask = False

    try:
        while True:
            got_new = False

            for name, stream in streams.items():
                frame, ts, fid = stream.read()
                if frame is None or fid == last_ids[name]:
                    continue
                last_ids[name] = fid
                got_new = True

                h, w = frame.shape[:2]
                scale = PROC_WIDTH / w
                small = cv2.resize(frame, (PROC_WIDTH, int(h * scale)))

                tracker = trackers[name]
                if USE_YOLO:
                    candidates = detect_yolo(frame, scale)
                    mask = np.zeros(small.shape[:2], dtype=np.uint8)
                    hsv = None
                else:
                    candidates, mask, hsv = detect_candidates(small)
                tracker.last_hsv = hsv

                status, pos = tracker.update(candidates, ts)

                # --- stats: camera fps vs processed fps ---
                st = stats[name]
                st["proc"] += 1
                now = time.time()
                if now - st["t0"] >= 1.0:
                    el = now - st["t0"]
                    st["proc_fps"] = st["proc"] / el
                    st["cam_fps"] = (fid - st["fid0"]) / el
                    st["t0"], st["proc"], st["fid0"] = now, 0, fid

                # --- drawing ---
                draw_trail(small, tracker.points)

                if pos is not None:
                    x, y = int(pos[0]), int(pos[1])
                    r = int(tracker.radius) + 4
                    cv2.circle(small, (x, y), r, (0, 255, 0), 2, cv2.LINE_AA)
                    if tracker.speed is not None:
                        put_text(small, f"{tracker.speed * 3.6:.0f} km/h",
                                 (x + r + 4, y - r), (0, 255, 0), 0.6)

                put_text(small, f"{name} | {status}", (10, 28))
                if tracker.speed is not None:
                    put_text(small,
                             f"Speed: {tracker.speed * 3.6:.1f} km/h  ({tracker.speed:.1f} m/s)",
                             (10, 60), (0, 255, 0))
                else:
                    put_text(small, "Speed: --", (10, 60), (180, 180, 180))
                if tracker.peak_visible(ts):
                    put_text(small, f"Peak: {tracker.peak * 3.6:.1f} km/h",
                             (10, 92), (0, 200, 255))
                if tracker.diam_px is not None and pos is not None:
                    put_text(small, f"ball: {tracker.diam_px:.0f} px",
                             (10, 124), (255, 255, 255), 0.55)
                put_text(small,
                         f"camera {st['cam_fps']:.0f} fps | processed {st['proc_fps']:.0f} fps",
                         (10, small.shape[0] - 12), (255, 255, 255), 0.55)

                cv2.imshow(name, small)
                if show_mask:
                    cv2.imshow(f"{name} mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                for t in trackers.values():
                    t.clear_trail()
            elif key == ord("r"):
                for t in trackers.values():
                    t.reset_peak()
            elif key == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    for name in streams:
                        try:
                            cv2.destroyWindow(f"{name} mask")
                        except cv2.error:
                            pass

            if not got_new:
                time.sleep(0.002)

    except KeyboardInterrupt:
        pass

    for stream in streams.values():
        stream.stop()
    cv2.destroyAllWindows()
    print("Speed tracker stopped")


if __name__ == "__main__":
    main()