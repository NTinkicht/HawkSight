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
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from src.core import (
    DEFAULT_CONF, DEFAULT_MODEL,
    DetectionModel, FrameProcessor, SystemController, VideoSource,
)


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
    model = DetectionModel(model_path, conf=conf)
    # Same pipeline as the desktop app, including the consecutive-frame
    # filter; stop_on_end ends the run when a video file finishes.
    controller = SystemController(VideoSource(source), model,
                                  FrameProcessor(), stop_on_end=True)

    if not Path(model_path).exists():
        print(f"{Path(model_path).name} not found, downloading it once…")
    print(f"Loading model: {model_path}")
    model.load()
    print("Model loaded. Opening camera/source…")

    if not controller.start():
        sys.exit(f"ERROR: Cannot open video source: {source!r}")

    WINDOW = "HawkSight — Gas Cylinder Detection  (Q = quit)"
    print("Running. Press Q in the window to quit.\n")

    try:
        while True:
            # Check before polling: once the stream has ended and the queue is
            # empty, no more frames can arrive. Frames still queued when the
            # stream ends are shown first instead of being dropped.
            running = controller.is_running
            data = controller.poll_frame()
            if data is not None:
                annotated, result = data
                _draw_status(annotated, result.count, controller.fps)
                cv2.imshow(WINDOW, annotated)
            elif not running:
                print("Stream ended.")
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        controller.stop()
        cv2.destroyAllWindows()
        print("HawkSight stopped.")


# ─── Entry point ─────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HawkSight CLI — gas cylinder detection"
    )
    parser.add_argument(
        "--source", default="0",
        help="Camera index (0, 1, …) or path to a video file (default: 0)",
    )
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL),
        help=f"Path to YOLO model weights (default: {DEFAULT_MODEL.name}). "
             "Only load weight files you trust: loading a .pt file can run code.",
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"Confidence threshold 0.05–0.95 (default: {DEFAULT_CONF})",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    try:
        source: int | str = int(args.source)
    except ValueError:
        source = args.source

    run(source=source, model_path=args.model, conf=args.conf)
