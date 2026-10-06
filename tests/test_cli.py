"""Tests for the command-line tool, using a fake video source and model."""

import time
import unittest
from unittest import mock

import numpy as np

import hawksight
from src import core
from src.core import DetectionResult, SystemController

FRAME = np.zeros((20, 20, 3), dtype=np.uint8)
HIT   = DetectionResult(boxes=[(1, 1, 5, 5)], labels=["Gas Cylinder"],
                        confidences=[0.9])


class FakeVideo:
    """Yields `frames` frames (10 ms apart), then reports end of stream."""
    def __init__(self, frames):
        self._left = frames

    def open(self):
        return True

    def read(self):
        if self._left == 0:
            return None
        self._left -= 1
        time.sleep(0.01)
        return FRAME

    def release(self):
        pass


class FakeModel:
    def __init__(self, *args, **kwargs):
        pass

    def load(self):
        pass

    def predict(self, frame):
        return HIT


class CliTest(unittest.TestCase):
    def run_cli(self, frames):
        shown = []
        with mock.patch.object(hawksight, "VideoSource",
                               lambda src: FakeVideo(frames)), \
             mock.patch.object(hawksight, "DetectionModel", FakeModel), \
             mock.patch.object(hawksight, "_draw_status",
                               lambda f, n, fps: shown.append(n)), \
             mock.patch.object(hawksight.cv2, "imshow"), \
             mock.patch.object(hawksight.cv2, "waitKey", return_value=-1), \
             mock.patch.object(hawksight.cv2, "destroyAllWindows"):
            hawksight.run(source=0, model_path="fake.pt", conf=0.5)
        return shown

    def test_cli_applies_the_consecutive_frame_filter(self):
        # Fewer hits in a row than the filter needs: nothing may be reported.
        shown = self.run_cli(SystemController.MIN_CONSECUTIVE - 1)
        self.assertTrue(shown, "no frames were displayed")
        self.assertEqual(set(shown), {0})

    def test_cli_reports_confirmed_detections(self):
        shown = self.run_cli(SystemController.MIN_CONSECUTIVE + 20)
        self.assertIn(1, shown)

    def test_frames_still_queued_when_the_video_ends_are_shown(self):
        # Display slower than the video: the stream ends while processed
        # frames are still waiting. They must be shown, not dropped.
        shown = []
        with mock.patch.object(hawksight, "VideoSource",
                               lambda src: FakeVideo(3)), \
             mock.patch.object(hawksight, "DetectionModel", FakeModel), \
             mock.patch.object(hawksight, "_draw_status",
                               lambda f, n, fps: shown.append(n)), \
             mock.patch.object(hawksight.cv2, "imshow"), \
             mock.patch.object(hawksight.cv2, "waitKey",
                               side_effect=lambda _: time.sleep(0.2) or -1), \
             mock.patch.object(hawksight.cv2, "destroyAllWindows"):
            hawksight.run(source=0, model_path="fake.pt", conf=0.5)
        self.assertGreaterEqual(len(shown), 2, "queued frames were dropped")

    def test_cli_defaults_match_the_app(self):
        args = hawksight.parse_args([])
        self.assertEqual(args.conf, core.DEFAULT_CONF)
        self.assertEqual(args.model, str(core.DEFAULT_MODEL))


if __name__ == "__main__":
    unittest.main()
