"""Tests for src.core, using fakes instead of a real camera or model."""

import unittest

import numpy as np

from src.core import (
    DetectionModel, DetectionResult, FrameProcessor, SystemController,
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


if __name__ == "__main__":
    unittest.main()
