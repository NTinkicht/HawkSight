"""HawkSight core: video capture, object detection, and frame processing."""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


# ─── Defaults shared by the desktop app and the CLI ─────────────────────────

ROOT       = Path(__file__).resolve().parent.parent
YOLO_PT    = ROOT / "yolov8n.pt"
CUSTOM_PT  = ROOT / "hawksight_custom.pt"
# Prefer the purpose-trained model when it is present.
DEFAULT_MODEL = CUSTOM_PT if CUSTOM_PT.exists() else YOLO_PT
DEFAULT_CONF  = 0.65
MAX_CAMERAS   = 6      # camera indices 0..5 are checked when scanning


def _camera_backend() -> int:
    # On Windows, OpenCV's default camera driver (Media Foundation) takes
    # 16-21 s to open on the dev webcam; DirectShow takes 3-4 s.
    return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


def list_cameras(max_index: int = MAX_CAMERAS,
                 assume_present: tuple = ()) -> list[int]:
    """Indices of the cameras that can be opened. OpenCV has no way to list
    devices, so each index is tried in turn. A camera that is already in use
    may refuse a second open, so indices in `assume_present` (e.g. the live
    feed) are listed without being opened. Slow: call off the UI thread."""
    found = []
    # Every empty index makes OpenCV print a warning; hide them while probing.
    log_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    try:
        for i in range(max_index):
            if i in assume_present:
                found.append(i)
                continue
            cap = cv2.VideoCapture(i, _camera_backend())
            try:
                if cap.isOpened():
                    found.append(i)
            finally:
                cap.release()
    finally:
        cv2.utils.logging.setLogLevel(log_level)
    return found


# ─── Data ────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    boxes:       list  = field(default_factory=list)
    labels:      list  = field(default_factory=list)
    confidences: list  = field(default_factory=list)
    frame_time:  float = field(default_factory=time.time)

    @property
    def count(self) -> int:
        return len(self.boxes)

    @property
    def best_confidence(self) -> float:
        return max(self.confidences, default=0.0)


# ─── VideoSource ─────────────────────────────────────────────────────────────

class VideoSource:
    # A live camera is read nonstop on a background thread that keeps only the
    # newest frame. Otherwise, when detection is slower than the camera, the
    # driver's buffer fills and read() hands back frames that are seconds old.
    # Video files are read in order on the caller's thread (no frame skipping).
    READ_TIMEOUT = 2.0   # seconds without a new camera frame = stream ended

    def __init__(self, source: Union[int, str] = 0):
        self._source = source
        self._cap: Optional[cv2.VideoCapture] = None
        self._reader: Optional[threading.Thread] = None
        self._reading = False
        self._lock    = threading.Lock()
        self._fresh   = threading.Event()   # set when _latest holds a new frame
        self._latest: Optional[np.ndarray] = None

    def open(self) -> bool:
        is_camera = isinstance(self._source, int)
        backend   = _camera_backend() if is_camera else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self._source, backend)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if is_camera:
            self._latest  = None
            self._fresh.clear()
            self._reading = True
            self._reader  = threading.Thread(target=self._read_forever,
                                             daemon=True)
            self._reader.start()
        return True

    def _read_forever(self):
        while self._reading:
            ok, frame = self._cap.read()
            if not ok:
                break
            with self._lock:
                self._latest = frame
                self._fresh.set()
        self._reading = False
        self._fresh.set()   # wake read() so it sees the stream has ended

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            return None
        if self._reader is None:
            ok, frame = self._cap.read()
            return frame if ok else None
        if not self._fresh.wait(self.READ_TIMEOUT):
            return None
        with self._lock:
            frame, self._latest = self._latest, None
            self._fresh.clear()
        return frame

    def release(self):
        self._reading = False
        if self._reader:
            self._reader.join(timeout=2)
            self._reader = None
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def source(self) -> Union[int, str]:
        return self._source

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()


# ─── DetectionModel ──────────────────────────────────────────────────────────

class DetectionModel:
    # Stock COCO models have no "gas cylinder" class, so with yolov8n.pt the
    # closest proxy ("bottle") is used.  The custom checkpoint
    # (hawksight_custom.pt) has a single gas_cylinder class, so every
    # detection from it is accepted.
    PROXY_CLASSES = {"bottle"}

    def __init__(self, model_path: Union[str, Path] = "yolov8n.pt",
                 conf: float = 0.4):
        self._model_path = str(model_path)
        self.conf        = conf    # setter clamps to 0.05–0.95
        self._model      = None
        self._custom     = False   # True when a purpose-trained model is loaded

    @property
    def conf(self) -> float:
        return self._conf

    @conf.setter
    def conf(self, value: float):
        self._conf = max(0.05, min(0.95, float(value)))

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_custom(self) -> bool:
        return self._custom

    def load(self):
        from ultralytics import YOLO
        self._model  = YOLO(self._model_path)
        names        = set(self._model.names.values())
        self._custom = "bottle" not in names   # COCO model → proxy mode

    def switch(self, model_path: Union[str, Path]):
        """Point at a different weights file. Caller must call load() again
        (is_loaded becomes False) before the next predict()."""
        self._model_path = str(model_path)
        self._model      = None
        self._custom     = False

    def predict(self, frame: np.ndarray) -> DetectionResult:
        if self._model is None:
            return DetectionResult()
        results = self._model.predict(frame, conf=self._conf, verbose=False)[0]
        boxes, labels, confs = [], [], []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label  = self._model.names[cls_id]
            if not self._custom and label not in self.PROXY_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2))
            labels.append("Gas Cylinder")
            confs.append(conf)
        return DetectionResult(boxes=boxes, labels=labels, confidences=confs)


# ─── FrameProcessor ──────────────────────────────────────────────────────────

class FrameProcessor:
    BOX_COLOR = (0, 220, 0)     # BGR green, bright enough to stand out
    TEXT_BG   = (0, 130, 0)     # darker green so the white label text reads
    FONT      = cv2.FONT_HERSHEY_SIMPLEX

    def annotate(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        out = frame.copy()
        for (x1, y1, x2, y2), label, conf in zip(
                result.boxes, result.labels, result.confidences):
            cv2.rectangle(out, (x1, y1), (x2, y2), self.BOX_COLOR, 2)
            tag = f"{label}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(tag, self.FONT, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1),
                          self.TEXT_BG, -1)
            cv2.putText(out, tag, (x1 + 2, y1 - 4), self.FONT, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def resize_for_display(self, frame: np.ndarray,
                           target_w: int, target_h: int) -> np.ndarray:
        h, w    = frame.shape[:2]
        scale   = min(target_w / w, target_h / h)
        nw, nh  = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas  = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_off   = (target_h - nh) // 2
        x_off   = (target_w - nw) // 2
        canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
        return canvas


# ─── SystemController ────────────────────────────────────────────────────────

class SystemController:
    # A cylinder must be seen in this many consecutive frames before it is
    # reported.  Filters out one-frame false alarms (chair legs, bottles …).
    MIN_CONSECUTIVE = 5

    def __init__(self, video: VideoSource, model: DetectionModel,
                 processor: FrameProcessor, stop_on_end: bool = False):
        # stop_on_end: stop when the source runs out of frames (end of a video
        # file, or a camera that stops sending). Otherwise keep waiting.
        self._stop_on_end = stop_on_end
        self._streak     = 0
        self._video      = video
        self._model      = model
        self._processor  = processor
        self._running    = False
        self._thread:    Optional[threading.Thread] = None
        self._frame_q:   queue.Queue = queue.Queue(maxsize=2)
        self._fps_times: deque       = deque(maxlen=30)
        self._total:     int         = 0
        self._start_ts:  float       = 0.0

    def start(self) -> bool:
        if self._running:
            return False
        if not self._video.open():
            return False
        self._running   = True
        self._total     = 0
        self._streak    = 0
        self._start_ts  = time.monotonic()
        self._fps_times.clear()
        self._thread    = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._video.release()

    def _loop(self):
        while self._running:
            frame = self._video.read()
            if frame is None:
                if self._stop_on_end:
                    self._running = False
                    break
                time.sleep(0.05)
                continue
            self._handle_frame(frame)

    def _handle_frame(self, frame: np.ndarray):
        self._fps_times.append(time.monotonic())
        result       = self._model.predict(frame)
        # Temporal filter: require MIN_CONSECUTIVE frames in a row.
        self._streak = self._streak + 1 if result.count else 0
        if self._streak < self.MIN_CONSECUTIVE:
            result = DetectionResult()
        # Count each sighting once, when it is first confirmed, not every frame.
        if self._streak == self.MIN_CONSECUTIVE:
            self._total += 1
        annotated    = self._processor.annotate(frame, result)
        if self._frame_q.full():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                pass
        self._frame_q.put((annotated, result))

    def poll_frame(self):
        try:
            return self._frame_q.get_nowait()
        except queue.Empty:
            return None

    @property
    def fps(self) -> float:
        t = list(self._fps_times)
        if len(t) < 2:
            return 0.0
        elapsed = t[-1] - t[0]
        return (len(t) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def runtime(self) -> float:
        return time.monotonic() - self._start_ts if self._running else 0.0

    @property
    def total_detections(self) -> int:
        return self._total

    @property
    def is_running(self) -> bool:
        return self._running
