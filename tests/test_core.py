"""Tests for src.core, using fakes instead of a real camera or model."""

import time
import unittest
from unittest import mock

import numpy as np

from src import core
from src.core import (
    DetectionModel, DetectionResult, FrameProcessor, ReplayBuffer,
    SystemController, VideoSource,
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


class AnnotateTest(unittest.TestCase):
    def test_box_and_label_are_green(self):
        frame  = np.zeros((100, 100, 3), dtype=np.uint8)
        result = DetectionResult(boxes=[(20, 40, 80, 90)],
                                 labels=["Gas Cylinder"], confidences=[0.9])
        out = FrameProcessor().annotate(frame, result)
        b, g, r = (int(c) for c in out[90, 50])        # bottom edge of box
        self.assertTrue(g > 150 and b == 0 and r == 0, (b, g, r))
        b, g, r = (int(c) for c in out[38, 21])        # label background
        self.assertTrue(g > 100 and b == 0 and r == 0, (b, g, r))


class ReplayBufferTest(unittest.TestCase):
    def test_keeps_only_the_last_n_seconds(self):
        buf = ReplayBuffer(seconds=15)
        for t in range(31):                      # one frame a second for 30 s
            buf.add(np.full((10, 10, 3), t, np.uint8), t=float(t))
        self.assertEqual(len(buf), 16)           # t = 15 .. 30
        self.assertEqual(buf.duration, 15.0)
        times = [t for t, _ in buf.frames()]
        self.assertEqual(times[0], 0.0)
        self.assertEqual(times[-1], 15.0)

    def test_frames_come_back_in_order_and_close_to_the_original(self):
        buf = ReplayBuffer()
        for t, value in enumerate((40, 120, 200)):
            buf.add(np.full((20, 20, 3), value, np.uint8), t=t * 0.1)
        values = [int(f.mean()) for _, f in buf.frames()]
        for got, want in zip(values, (40, 120, 200)):
            self.assertAlmostEqual(got, want, delta=3)   # JPEG is lossy

    def test_large_frames_are_shrunk(self):
        buf = ReplayBuffer()
        buf.add(np.zeros((720, 1280, 3), np.uint8), t=0.0)
        (_, frame), = buf.frames()
        self.assertEqual(frame.shape, (540, 960, 3))

    def test_clear_and_empty(self):
        buf = ReplayBuffer()
        self.assertEqual(buf.frames(), [])
        self.assertEqual(buf.duration, 0.0)
        buf.add(np.zeros((10, 10, 3), np.uint8))
        buf.clear()
        self.assertEqual(len(buf), 0)


class FakeBox:
    """Shaped like one entry of an ultralytics result's .boxes."""
    def __init__(self, xyxy, conf, cls=0):
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.conf = [conf]
        self.cls  = [cls]


class WholeFrameFalseAlarmTest(unittest.TestCase):
    def predict(self, *xyxys):
        model = DetectionModel()
        result = mock.Mock(boxes=[FakeBox(b, 0.75) for b in xyxys])
        model._model  = mock.Mock(names={0: "gas_cylinder"},
                                  predict=mock.Mock(return_value=[result]))
        model._custom = True
        return model.predict(np.zeros((720, 1280, 3), np.uint8))

    def test_box_covering_the_whole_frame_is_ignored(self):
        self.assertEqual(self.predict((0, 0, 1280, 720)).count, 0)
        self.assertEqual(self.predict((20, 10, 1260, 710)).count, 0)   # ~94 %

    def test_normal_and_large_boxes_are_kept(self):
        self.assertEqual(self.predict((400, 150, 800, 650)).count, 1)
        self.assertEqual(self.predict((100, 50, 1180, 670)).count, 1)  # ~73 %

    def test_only_the_whole_frame_box_is_dropped(self):
        result = self.predict((0, 0, 1280, 720), (400, 150, 800, 650))
        self.assertEqual(result.boxes, [(400, 150, 800, 650)])


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

    def test_windows_falls_back_to_media_foundation(self):
        class DshowFails(FakeCapture):
            def isOpened(self):
                return self.backend != core.cv2.CAP_DSHOW
        patcher = mock.patch.object(core.cv2, "VideoCapture", DshowFails)
        patcher.start()
        self.addCleanup(patcher.stop)
        with mock.patch.object(core.sys, "platform", "win32"):
            video = VideoSource(0)
            self.addCleanup(video.release)
            self.assertTrue(video.open())
        self.assertEqual(video._cap.backend, core.cv2.CAP_MSMF)

    def test_camera_that_never_opens_reports_failure(self):
        class NeverOpens(FakeCapture):
            def isOpened(self):
                return False
        with mock.patch.object(core.cv2, "VideoCapture", NeverOpens), \
             mock.patch.object(core.sys, "platform", "win32"):
            video = VideoSource(0)
            self.assertFalse(video.open())
            self.assertFalse(video.is_open())

    def test_video_file_uses_default_backend(self):
        with mock.patch.object(core.sys, "platform", "win32"):
            video = self.open_video("clip.mp4")
        self.assertEqual(video._cap.backend, core.cv2.CAP_ANY)

    def test_other_platforms_use_default_backend(self):
        with mock.patch.object(core.sys, "platform", "linux"):
            video = self.open_video(0)
        self.assertEqual(video._cap.backend, core.cv2.CAP_ANY)


class ListCamerasTest(unittest.TestCase):
    def fake_capture(self, connected, opened):
        class Capture:
            def __init__(self, index, backend=0):
                self.index = index
                opened.append(index)

            def isOpened(self):
                return self.index in connected

            def release(self):
                pass
        return Capture

    def test_lists_only_cameras_that_open(self):
        opened = []
        with mock.patch.object(core.cv2, "VideoCapture",
                               self.fake_capture({0, 2}, opened)):
            self.assertEqual(core.list_cameras(max_index=4), [0, 2])
        self.assertEqual(opened, [0, 1, 2, 3])

    def test_camera_in_use_is_listed_without_opening_it(self):
        opened = []
        with mock.patch.object(core.cv2, "VideoCapture",
                               self.fake_capture({0}, opened)):
            found = core.list_cameras(max_index=3, assume_present=(1,))
        self.assertEqual(found, [0, 1])
        self.assertNotIn(1, opened)


if __name__ == "__main__":
    unittest.main()
