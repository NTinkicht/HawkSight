"""HawkSight core: video capture, object detection, and frame processing."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


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
    def __init__(self, source: Union[int, str] = 0):
        self._source = source
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._source)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return True
        return False

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()


# ─── DetectionModel ──────────────────────────────────────────────────────────

class DetectionModel:
    # Stock COCO models have no "gas cylinder" class, so with yolov8n.pt the
    # closest proxy ("bottle") is used.  A custom-trained checkpoint
    # (hawksight_custom.pt, produced by train_hawksight.py) contains only
    # cylinder classes, so every detection from it is accepted.
    PROXY_CLASSES = {"bottle"}

    def __init__(self, model_path: Union[str, Path] = "yolov8n.pt",
                 conf: float = 0.4):
        self._model_path = str(model_path)
        self._conf       = conf
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
    BOX_COLOR = (0, 140, 255)   # BGR orange
    TEXT_BG   = (0, 80,  180)
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
                 processor: FrameProcessor):
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
                time.sleep(0.05)
                continue
            self._fps_times.append(time.monotonic())
            result       = self._model.predict(frame)
            # Temporal filter: require MIN_CONSECUTIVE frames in a row.
            self._streak = self._streak + 1 if result.count else 0
            if self._streak < self.MIN_CONSECUTIVE:
                result = DetectionResult()
            self._total += result.count
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
