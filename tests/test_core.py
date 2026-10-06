"""Tests for src.core, using fakes instead of a real camera or model."""

import time
import unittest
from unittest import mock

import numpy as np

from src import core
from src.core import (
    DetectionModel, DetectionResult, FrameProcessor, SystemController,
    VideoSource,
)

FRAME = np.zeros((20, 20, 3), dtype=np.uint8)
HIT   = DetectionResult(boxes=[(1, 1, 5, 5)], labels=["Gas Cylinder"],
                        confidences=[0.9])


class FakeModel:
    """Returns the queued results in order, one per predict() call."""
    def __init__(self, results):
        self._results = iter(results)

    def predict(self, frame):
        return next(self._results)


class TotalCountTest(unittest.TestCase):
    def run_frames(self, results):
        ctrl = SystemController(None, FakeModel(results), FrameProcessor())
        for _ in results:
            ctrl._handle_frame(FRAME)
        return ctrl

    def test_one_long_sighting_counts_as_one_alert(self):
        n = SystemController.MIN_CONSECUTIVE
        ctrl = self.run_frames([HIT] * (n + 100))
        self.assertEqual(ctrl.total_detections, 1)

    def test_short_flicker_is_not_counted(self):
        n = SystemController.MIN_CONSECUTIVE
        ctrl = self.run_frames([HIT] * (n - 1))
        self.assertEqual(ctrl.total_detections, 0)

    def test_two_separate_sightings_count_as_two(self):
        n = SystemController.MIN_CONSECUTIVE
        ctrl = self.run_frames([HIT] * n + [DetectionResult()] + [HIT] * n)
        self.assertEqual(ctrl.total_detections, 2)


class ConfidenceLimitTest(unittest.TestCase):
    def test_constructor_clamps_confidence(self):
        self.assertEqual(DetectionModel(conf=2).conf, 0.95)
        self.assertEqual(DetectionModel(conf=-1).conf, 0.05)
        self.assertEqual(DetectionModel(conf=0.5).conf, 0.5)


class FakeCapture:
    """Delivers frames 1..FRAMES (a frame whose pixels all equal its number),
    one every 5 ms like a camera, then reports end of stream."""
    FRAMES = 40

    def __init__(self, source, backend=0):   # 0 == cv2.CAP_ANY
        self.n = 0
        self.backend = backend

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def read(self):
        if self.n >= self.FRAMES:
            return False, None
        time.sleep(0.005)
        self.n += 1
        return True, np.full((2, 2, 3), self.n, dtype=np.uint8)

    def release(self):
        pass


class VideoSourceTest(unittest.TestCase):
    def open_video(self, source):
        patcher = mock.patch.object(core.cv2, "VideoCapture", FakeCapture)
        patcher.start()
        self.addCleanup(patcher.stop)
        video = VideoSource(source)
        self.addCleanup(video.release)
        self.assertTrue(video.open())
        return video

    def test_camera_skips_stale_frames(self):
        video = self.open_video(0)
        time.sleep(0.1)   # slow detection: ~20 frames arrive meanwhile
        frame = video.read()
        self.assertGreater(int(frame[0, 0, 0]), 10,
                           "read() returned an old buffered frame")

    def test_camera_never_returns_same_frame_twice(self):
        video = self.open_video(0)
        seen = []
        while (frame := video.read()) is not None:
            seen.append(int(frame[0, 0, 0]))
        self.assertEqual(seen, sorted(set(seen)))
        self.assertEqual(seen[-1], FakeCapture.FRAMES)

    def test_video_file_reads_every_frame_in_order(self):
        video = self.open_video("clip.mp4")
        time.sleep(0.1)
        seen = []
        while (frame := video.read()) is not None:
            seen.append(int(frame[0, 0, 0]))
        self.assertEqual(seen, list(range(1, FakeCapture.FRAMES + 1)))

    def test_windows_camera_uses_directshow(self):
        with mock.patch.object(core.sys, "platform", "win32"):
            video = self.open_video(0)
        self.assertEqual(video._cap.backend, core.cv2.CAP_DSHOW)

    def test_video_file_uses_default_backend(self):
        with mock.patch.object(core.sys, "platform", "win32"):
            video = self.open_video("clip.mp4")
        self.assertEqual(video._cap.backend, core.cv2.CAP_ANY)

    def test_other_platforms_use_default_backend(self):
        with mock.patch.object(core.sys, "platform", "linux"):
            video = self.open_video(0)
        self.assertEqual(video._cap.backend, core.cv2.CAP_ANY)


if __name__ == "__main__":
    unittest.main()
