"""
ball_tracker.py
Real-time tennis ball detection + trajectory line (2D) on live RTSP feeds.

Method: HSV colour mask -> shape filtering (circularity / fill) -> Kalman filter
for smoothing, prediction during short misses, and rejecting jumps to other
yellow objects.

Keys:   Q = quit     M = show/hide mask     C = clear trail
Mouse:  LEFT-CLICK on the ball in a camera window = auto-set the HSV range
"""

import cv2
import numpy as np
import threading
import time
from collections import deque

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
CAMERAS = {
    # "Camera 1": "rtsp://admin:12345@192.168.100.225:554/",
    "Camera 1": "rtsp://admin:admin12345@192.168.100.197:554/live",
}

ACTIVE_CAMERAS = ["Camera 1"]   # use ["Camera 1"] at the office, ["Camera 1", "Camera 2"] for both

PROC_WIDTH = 960          # frames are resized to this width for detection/display
TRAIL_SECONDS = 2.0       # how long the path line stays visible
TRACK_REFERENCE_FPS = 25.0
MAX_TRACK_GAP_S = 0.5     # stop predicting after this long without a detection
ACQUIRE_WINDOW_S = 0.12   # require two nearby observations before starting a track
MAX_JUMP_PX = 150         # max allowed jump per 1/25 s (in PROC_WIDTH pixels)
MIN_CIRCULARITY = 0.5     # lower = accepts blurry/elongated (fast) balls
MIN_FILL = 0.45

# --- YOLO detector (used after you have trained your own model) ---
USE_YOLO = False          # keep HSV mode for real-time processing
MODEL_PATH = r"E:\ball project\TennisBallPath\runs\ball-4\weights\best.pt"
YOLO_CONF = 0.25          # lower = more detections, more false positives
YOLO_IMGSZ = 960

SETTINGS_WIN = "Settings"


# ----------------------------------------------------------------------------
# CAMERA READER (same idea as your camera_test.py)
# ----------------------------------------------------------------------------
class CameraStream:
    def __init__(self, name, url):
        self.name = name
        if isinstance(url, int):
            self.cap = cv2.VideoCapture(url, cv2.CAP_DSHOW)     # local webcam (Windows)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 60)                  # camera uses what it supports
            # optional, less motion blur (driver dependent, try values -5 to -8):
            # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            # self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        else:
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)    # RTSP camera
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.opened = self.cap.isOpened()
        self.frame = None
        self.timestamp = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                ts = time.perf_counter()
                with self.lock:
                    self.frame = frame
                    self.timestamp = ts
                    self.frame_id += 1
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, None, 0
            return self.frame.copy(), self.timestamp, self.frame_id

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


# ----------------------------------------------------------------------------
# SETTINGS WINDOW (HSV trackbars)
# ----------------------------------------------------------------------------
def create_settings_window():
    cv2.namedWindow(SETTINGS_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(SETTINGS_WIN, 420, 320)
    defaults = [
        ("H low", 25, 179), ("H high", 45, 179),
        ("S low", 70, 255), ("S high", 255, 255),
        ("V low", 70, 255), ("V high", 255, 255),
        ("Min area", 20, 5000), ("Max area", 3000, 8000),
    ]
    for label, value, maximum in defaults:
        cv2.createTrackbar(label, SETTINGS_WIN, value, maximum, lambda x: None)


def get_settings():
    g = lambda n: cv2.getTrackbarPos(n, SETTINGS_WIN)
    lower = np.array([g("H low"), g("S low"), g("V low")], dtype=np.uint8)
    upper = np.array([g("H high"), g("S high"), g("V high")], dtype=np.uint8)
    return lower, upper, g("Min area"), max(g("Max area"), g("Min area") + 1)


# ----------------------------------------------------------------------------
# DETECTION
# ----------------------------------------------------------------------------
def detect_candidates(small_bgr):
    """Return (candidates, mask, hsv). Each candidate: dict(x, y, r, score)."""
    lower, upper, min_area, max_area = get_settings()

    blur = cv2.GaussianBlur(small_bgr, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            continue

        (x, y), r = cv2.minEnclosingCircle(c)
        if r < 2:
            continue

        fill = area / (np.pi * r * r)
        if fill < MIN_FILL:
            continue

        short_side = min(cv2.minAreaRect(c)[1])      # blur stretches the long side only
        candidates.append({"x": x, "y": y, "r": r, "d": max(short_side, 2.0),
                           "score": circularity * fill})

    return candidates, mask, hsv


_yolo_model = None


def detect_yolo(frame, scale):
    """Run YOLO on the FULL-resolution frame (better for a small ball).
    Returns candidates in the scaled (display) coordinate system."""
    global _yolo_model
    import torch

    use_gpu = torch.cuda.is_available()
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(MODEL_PATH)
        print("YOLO loaded. GPU:", use_gpu)

    results = _yolo_model.predict(
        frame,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        device=0 if use_gpu else "cpu",
        quantize=16 if use_gpu else 32,
        verbose=False,
    )

    candidates = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        candidates.append({
            "x": (x1 + x2) / 2 * scale,
            "y": (y1 + y2) / 2 * scale,
            "r": max(x2 - x1, y2 - y1) / 2 * scale,
            "d": min(x2 - x1, y2 - y1) * scale,
            "score": float(box.conf[0]),
        })
    return candidates


# ----------------------------------------------------------------------------
# TRACKER (Kalman filter + trail)
# ----------------------------------------------------------------------------
class BallTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0

        self.active = False
        self.radius = 8
        self.trail = deque()      # (x, y, timestamp) or (None, None, timestamp) = break
        self.last_hsv = None      # used by the click-to-calibrate handler
        self.last_update_timestamp = None
        self.last_detection_timestamp = None
        self.pending_candidate = None
        self.pending_timestamp = None

    def reset_track(self, timestamp):
        self.active = False
        self.last_detection_timestamp = None
        self.pending_candidate = None
        self.pending_timestamp = None
        if self.trail and self.trail[-1][0] is not None:
            self.trail.append((None, None, timestamp))   # break the line

    def clear_trail(self):
        self.trail.clear()

    def update(self, candidates, timestamp):
        """Returns (status, (x, y) or None)."""
        dt = (1.0 / TRACK_REFERENCE_FPS if self.last_update_timestamp is None
              else max(timestamp - self.last_update_timestamp, 1e-3))
        self.last_update_timestamp = timestamp

        if (self.active and self.last_detection_timestamp is not None
                and timestamp - self.last_detection_timestamp > MAX_TRACK_GAP_S):
            self.reset_track(timestamp)

        predicted = None
        if self.active:
            frame_dt = min(dt, MAX_TRACK_GAP_S) * TRACK_REFERENCE_FPS
            self.kf.transitionMatrix[0, 2] = frame_dt
            self.kf.transitionMatrix[1, 3] = frame_dt
            p = self.kf.predict()
            predicted = (float(p[0, 0]), float(p[1, 0]))

        chosen = None
        if candidates:
            if predicted is None:
                best = max(candidates, key=lambda c: c["score"])
                pending_age = (timestamp - self.pending_timestamp
                               if self.pending_timestamp is not None else None)
                if (self.pending_candidate is not None
                        and pending_age is not None
                        and 0 < pending_age <= ACQUIRE_WINDOW_S):
                    limit = MAX_JUMP_PX * max(pending_age * TRACK_REFERENCE_FPS, 1.0)
                    pending_d = max(
                        self.pending_candidate.get("d", 2 * self.pending_candidate["r"]),
                        1.0,
                    )
                    near = []
                    for candidate in candidates:
                        distance = np.hypot(
                            candidate["x"] - self.pending_candidate["x"],
                            candidate["y"] - self.pending_candidate["y"],
                        )
                        diameter = candidate.get("d", 2 * candidate["r"])
                        if distance <= limit and 0.5 <= diameter / pending_d <= 2.0:
                            near.append(candidate)
                    if near:
                        chosen = min(
                            near,
                            key=lambda c: np.hypot(
                                c["x"] - self.pending_candidate["x"],
                                c["y"] - self.pending_candidate["y"]),
                        )
                if chosen is None:
                    self.pending_candidate = best
                    self.pending_timestamp = timestamp
                else:
                    self.pending_candidate = None
                    self.pending_timestamp = None
            else:
                elapsed = timestamp - self.last_detection_timestamp
                limit = MAX_JUMP_PX * max(elapsed * TRACK_REFERENCE_FPS, 1.0)
                near = [c for c in candidates
                        if np.hypot(c["x"] - predicted[0], c["y"] - predicted[1]) <= limit]
                if near:
                    chosen = min(near, key=lambda c: np.hypot(
                        c["x"] - predicted[0], c["y"] - predicted[1]))

        status, pos = "SEARCHING", None

        if chosen is not None:
            if not self.active:
                self.kf.statePost = np.array(
                    [[chosen["x"]], [chosen["y"]], [0], [0]], dtype=np.float32)
                self.kf.errorCovPost = np.eye(4, dtype=np.float32)
                self.active = True
            else:
                self.kf.correct(np.array([[chosen["x"]], [chosen["y"]]], dtype=np.float32))

            self.last_detection_timestamp = timestamp
            self.radius = chosen["r"]
            pos = (chosen["x"], chosen["y"])
            status = "DETECTED"
            self.trail.append((pos[0], pos[1], timestamp))

        elif self.active:
            pos = predicted
            status = "PREDICTED"

        # drop old trail points
        while self.trail and timestamp - self.trail[0][2] > TRAIL_SECONDS:
            self.trail.popleft()

        return status, pos


# ----------------------------------------------------------------------------
# DRAWING
# ----------------------------------------------------------------------------
def draw_trail(img, trail):
    n = len(trail)
    for i in range(1, n):
        a, b = trail[i - 1], trail[i]
        if a[0] is None or b[0] is None:
            continue
        thickness = max(1, int(1 + 4 * i / n))
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 (0, 0, 255), thickness, cv2.LINE_AA)


def draw_ball(img, status, pos, radius, scale):
    if pos is None:
        return
    x, y = int(pos[0]), int(pos[1])
    r = int(radius) + 4
    color = (0, 255, 0) if status == "DETECTED" else (0, 165, 255)
    cv2.circle(img, (x, y), r, color, 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), 2, color, -1)
    # coordinates reported in ORIGINAL full-resolution pixels (needed later for 3D)
    label = f"{status} ({x / scale:.0f}, {y / scale:.0f})"
    cv2.putText(img, label, (x + r + 4, y - r),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# CLICK ON BALL -> AUTO HSV RANGE
# ----------------------------------------------------------------------------
def on_mouse(event, x, y, flags, tracker):
    if event != cv2.EVENT_LBUTTONDOWN or tracker.last_hsv is None:
        return

    patch = tracker.last_hsv[max(0, y - 3):y + 4, max(0, x - 3):x + 4].reshape(-1, 3)
    if len(patch) == 0:
        return

    h, s, v = [int(val) for val in np.median(patch, axis=0)]
    print(f"Sampled HSV at click: H={h} S={s} V={v}")

    cv2.setTrackbarPos("H low", SETTINGS_WIN, max(0, h - 12))
    cv2.setTrackbarPos("H high", SETTINGS_WIN, min(179, h + 12))
    cv2.setTrackbarPos("S low", SETTINGS_WIN, max(0, s - 80))
    cv2.setTrackbarPos("S high", SETTINGS_WIN, 255)
    cv2.setTrackbarPos("V low", SETTINGS_WIN, max(0, v - 80))
    cv2.setTrackbarPos("V high", SETTINGS_WIN, 255)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    streams = {}
    trackers = {}

    for name in ACTIVE_CAMERAS:
        stream = CameraStream(name, CAMERAS[name])
        if not stream.opened:
            print(name, "connection failed")
            continue
        print(name, "connected successfully")
        stream.start()
        streams[name] = stream
        trackers[name] = BallTracker()

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

                draw_trail(small, tracker.trail)
                draw_ball(small, status, pos, tracker.radius, scale)
                cv2.putText(small, f"{name} | {status}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow(name, small)
                if show_mask:
                    cv2.imshow(f"{name} mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                for t in trackers.values():
                    t.clear_trail()
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
    print("Ball tracker stopped")


if __name__ == "__main__":
    main()