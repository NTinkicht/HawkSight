"""HawkSight – headless CLI mode.

Usage
-----
  python hawksight.py                    # default camera (index 0)
  python hawksight.py --source 1         # second camera
  python hawksight.py --source video.mp4 # video file
  python hawksight.py --conf 0.6         # confidence threshold
  python hawksight.py --model custom.pt  # custom model weights

Press  Q  in the OpenCV window to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from src.core import VideoSource, DetectionModel, FrameProcessor


# ─── Status overlay ──────────────────────────────────────────────────────────

def _draw_status(frame, n_objects: int, fps: float) -> None:
    h, w  = frame.shape[:2]
    alert = "  ⚑ CYLINDER DETECTED" if n_objects else ""
    text  = (f"  HawkSight  |  Objects: {n_objects}{alert}"
             f"  |  FPS: {fps:.1f}  |  Q = quit")
    cv2.rectangle(frame, (0, h - 28), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, text, (8, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (200, 200, 200), 1, cv2.LINE_AA)


# ─── Main run loop ───────────────────────────────────────────────────────────

def run(source: int | str, model_path: str, conf: float) -> None:
    video     = VideoSource(source)
    model     = DetectionModel(model_path, conf=conf)
    processor = FrameProcessor()

    print(f"Loading model: {model_path}")
    model.load()
    print("Model loaded. Opening camera/source…")

    if not video.open():
        sys.exit(f"ERROR: Cannot open video source: {source!r}")

    WINDOW = "HawkSight — Gas Cylinder Detection  (Q = quit)"
    fps_times: list[float] = []
    print("Running. Press Q in the window to quit.\n")

    try:
        while True:
            frame = video.read()
            if frame is None:
                print("Stream ended.")
                break

            t = time.monotonic()
            fps_times.append(t)
            # keep a rolling 30-frame window
            fps_times = fps_times[-30:]
            fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0]) \
                  if len(fps_times) >= 2 else 0.0

            result    = model.predict(frame)
            annotated = processor.annotate(frame, result)
            _draw_status(annotated, result.count, fps)

            cv2.imshow(WINDOW, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video.release()
        cv2.destroyAllWindows()
        print("HawkSight stopped.")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HawkSight CLI — gas cylinder detection"
    )
    parser.add_argument(
        "--source", default="0",
        help="Camera index (0, 1, …) or path to a video file (default: 0)",
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="Path to YOLO model weights (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Confidence threshold 0.05–0.95 (default: 0.5)",
    )
    args = parser.parse_args()

    try:
        source: int | str = int(args.source)
    except ValueError:
        source = args.source

    run(source=source, model_path=args.model, conf=args.conf)
