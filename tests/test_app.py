"""GUI tests for HawkSightApp. Skipped when Tk can't open a window."""

import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import hawksight_app
from hawksight_app import HawkSightApp
from src.core import DetectionResult


def _pump_until(app, condition, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.update()
        if condition():
            return True
        time.sleep(0.02)
    return False


class FakeVideo:
    """Stands in for VideoSource: a camera that always delivers a frame."""
    opened = []

    def __init__(self, source):
        self.source   = source
        self.released = False

    def open(self):
        FakeVideo.opened.append(self)
        return True

    def read(self):
        time.sleep(0.01)
        return None if self.released else np.zeros((20, 20, 3), np.uint8)

    def release(self):
        self.released = True


class AppTestCase(unittest.TestCase):
    CAMERAS = [0]   # what the camera scan finds; no real webcam is touched

    def setUp(self):
        patcher = mock.patch.object(hawksight_app, "list_cameras",
                                    lambda assume_present=(): list(self.CAMERAS))
        patcher.start()
        self.addCleanup(patcher.stop)
        try:
            self.app = HawkSightApp()
        except tk.TclError as exc:
            self.skipTest(f"Tk unavailable: {exc}")
        self.assertTrue(_pump_until(self.app, lambda: not self.app._scanning),
                        "camera scan never finished")

    def tearDown(self):
        self.app.on_close()

    def log_text(self):
        return self.app._log.get("1.0", tk.END)


class StartupTest(AppTestCase):
    def test_window_starts_idle(self):
        self.assertEqual(self.app._sv_status.get(), "Ready")
        self.assertEqual(str(self.app._btn_start["state"]), tk.NORMAL)


class ModelLoadFailureTest(AppTestCase):
    def test_failed_load_reenables_start_and_logs_reason(self):
        def broken_load():
            raise RuntimeError("weights file is corrupt")
        self.app._model.load = broken_load

        self.app._on_start()
        recovered = _pump_until(
            self.app, lambda: str(self.app._btn_start["state"]) == tk.NORMAL)

        self.assertTrue(recovered, "START button stayed disabled")
        self.assertEqual(str(self.app._cmb_model["state"]), "readonly")
        self.assertEqual(self.app._sv_status.get(), "Detector problem")
        self.assertIn("weights file is corrupt", self.log_text())
        self.assertFalse(self.app._controller.is_running)


class StockModelTest(AppTestCase):
    """YOLOv8n stays selectable when yolov8n.pt hasn't been downloaded yet."""

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.missing = Path(tmp.name) / "yolov8n.pt"
        paths = dict(hawksight_app.MODEL_PATHS, yolo=self.missing)
        patcher = mock.patch.object(hawksight_app, "MODEL_PATHS", paths)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app._model_key = "custom"

    def test_yolo_can_be_chosen_before_it_is_downloaded(self):
        self.app._on_model_switch("yolo")
        self.assertEqual(self.app._model_key, "yolo")
        self.assertEqual(self.app._model._model_path, str(self.missing))

    def test_failed_download_says_so(self):
        self.app._on_model_switch("yolo")

        def offline_load():
            raise ConnectionError("no internet")
        self.app._model.load = offline_load

        self.app._on_start()
        self.assertIn("downloading", self.app._sv_status.get().lower())
        recovered = _pump_until(
            self.app, lambda: str(self.app._btn_start["state"]) == tk.NORMAL)

        self.assertTrue(recovered, "START button stayed disabled")
        self.assertIn("Could not download yolov8n.pt", self.log_text())
        self.assertIn("internet connection", self.log_text())


class ModelDropdownTest(AppTestCase):
    def choose(self, label):
        self.app._sv_model.set(label)
        self.app._cmb_model.event_generate("<<ComboboxSelected>>")
        self.app.update()

    def test_dropdown_lists_every_available_model(self):
        self.assertEqual(
            list(self.app._cmb_model["values"]),
            [hawksight_app.MODEL_LABELS[k]
             for k in hawksight_app.available_model_keys()])
        self.assertIn(hawksight_app.MODEL_LABELS["yolo"],
                      self.app._cmb_model["values"])

    def test_choosing_a_model_switches_to_it(self):
        self.app._on_model_switch("custom")
        self.choose(hawksight_app.MODEL_LABELS["yolo"])
        self.assertEqual(self.app._model_key, "yolo")
        self.assertEqual(self.app._model._model_path,
                         str(hawksight_app.MODEL_PATHS["yolo"]))
        self.assertIn("model switched", self.log_text())

    def test_custom_model_is_hidden_when_its_file_is_missing(self):
        paths = dict(hawksight_app.MODEL_PATHS,
                     custom=Path("does_not_exist.pt"))
        with mock.patch.object(hawksight_app, "MODEL_PATHS", paths):
            self.assertEqual(hawksight_app.available_model_keys(), ["yolo"])

    def test_dropdown_is_locked_while_starting(self):
        def broken_load():
            raise RuntimeError("no model in tests")
        self.app._model.load = broken_load
        self.app._on_start()
        self.assertEqual(str(self.app._cmb_model["state"]), tk.DISABLED)
        _pump_until(self.app,
                    lambda: str(self.app._btn_start["state"]) == tk.NORMAL)
        self.assertEqual(str(self.app._cmb_model["state"]), "readonly")


class CameraDropdownTest(AppTestCase):
    CAMERAS = [0, 2]

    def setUp(self):
        super().setUp()
        FakeVideo.opened = []
        patcher = mock.patch.object(hawksight_app, "VideoSource", FakeVideo)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app._model.load = lambda: None

    def choose(self, label):
        self.app._sv_camera.set(label)
        self.app._cmb_camera.event_generate("<<ComboboxSelected>>")
        self.app.update()

    def start_and_wait(self):
        self.app._on_start()
        self.assertTrue(_pump_until(
            self.app, lambda: self.app._sv_status.get() == "Watching"))

    def test_dropdown_lists_connected_cameras(self):
        self.assertEqual(list(self.app._cmb_camera["values"]),
                         ["Camera 0 (laptop)", "Camera 2"])
        self.assertEqual(self.app._sv_camera.get(), "Camera 0 (laptop)")
        self.assertEqual(str(self.app._cmb_camera["state"]), "readonly")

    def test_start_uses_the_chosen_camera(self):
        self.choose("Camera 2")
        self.start_and_wait()
        self.assertEqual([v.source for v in FakeVideo.opened], [2])

    def test_switching_while_live_stops_old_feed_and_starts_new(self):
        self.start_and_wait()
        old = FakeVideo.opened[-1]
        self.choose("Camera 2")
        self.assertTrue(old.released, "old camera was not released")
        self.start_and_wait()
        new = FakeVideo.opened[-1]
        self.assertEqual(new.source, 2)
        self.assertFalse(new.released)
        self.assertTrue(self.app._controller.is_running)
        self.assertIn("camera switched", self.log_text())

    def test_rescan_picks_up_new_camera_and_keeps_selection(self):
        self.choose("Camera 2")
        self.CAMERAS = [0, 1, 2]
        self.app._btn_rescan.invoke()
        _pump_until(self.app, lambda: not self.app._scanning)
        self.assertEqual(list(self.app._cmb_camera["values"]),
                         ["Camera 0 (laptop)", "Camera 1", "Camera 2"])
        self.assertEqual(self.app._sv_camera.get(), "Camera 2")

    def test_no_camera_found_falls_back_to_laptop_camera(self):
        self.choose("Camera 2")
        self.CAMERAS = []
        self.app._btn_rescan.invoke()
        _pump_until(self.app, lambda: not self.app._scanning)
        self.assertEqual(self.app._sv_camera.get(), "Camera 0 (laptop)")
        self.assertIn("using the laptop camera", self.log_text())
        self.start_and_wait()
        self.assertEqual([v.source for v in FakeVideo.opened], [0])

    def test_camera_that_will_not_open_explains_what_to_check(self):
        class Unopenable(FakeVideo):
            def open(self):
                return False
        with mock.patch.object(hawksight_app, "VideoSource", Unopenable):
            self.app._on_start()
            _pump_until(self.app,
                        lambda: self.app._sv_status.get() == "Camera problem")
        self.assertIn("Could not open Camera 0 (laptop)", self.log_text())
        self.assertIn("Privacy", self.log_text())
        self.assertEqual(str(self.app._btn_start["state"]), tk.NORMAL)


class EasyUiTest(AppTestCase):
    def test_there_is_no_snapshot_button(self):
        self.assertFalse(hasattr(self.app, "_btn_snap"))

    def test_log_is_hidden_until_show_log_is_clicked(self):
        self.assertFalse(self.app.log_visible)
        self.app._btn_log.invoke()
        self.app.update()
        self.assertTrue(self.app.log_visible)
        self.assertIn("Hide log", self.app._btn_log["text"])
        self.app._btn_log.invoke()
        self.app.update()
        self.assertFalse(self.app.log_visible)

    def test_light_is_off_until_a_cylinder_is_spotted(self):
        self.assertEqual(self.app.light_color, self.app.LIGHT_OFF)
        self.assertEqual(self.app._sv_best.get(), "—")

        hit = DetectionResult(boxes=[(1, 1, 5, 5)], labels=["Gas Cylinder"],
                              confidences=[0.87])
        self.app._update_stats(hit)
        self.app.update()
        self.assertIn(self.app.light_color,
                      (self.app.LIGHT_ON, self.app.LIGHT_DIM))
        self.assertEqual(self.app._sv_best.get(), "87%")

        self.app._update_stats(DetectionResult())
        self.assertEqual(self.app.light_color, self.app.LIGHT_OFF)
        self.assertEqual(self.app._sv_best.get(), "—")

    def test_counters_are_gone(self):
        for name in ("_sv_objects", "_sv_runtime", "_sv_total", "_alert_lbl"):
            self.assertFalse(hasattr(self.app, name), name)

    def test_camera_problem_is_shown_on_the_video_area(self):
        class Unopenable(FakeVideo):
            def open(self):
                return False
        self.app._model.load = lambda: None
        with mock.patch.object(hawksight_app, "VideoSource", Unopenable):
            self.app._on_start()
            _pump_until(self.app,
                        lambda: self.app._sv_status.get() == "Camera problem")
        canvas_text = " ".join(
            self.app._canvas.itemcget(i, "text")
            for i in self.app._canvas.find_all()
            if self.app._canvas.type(i) == "text")
        self.assertIn("Camera problem", canvas_text)
        self.assertIn("press START again", canvas_text)


class PhotoKeyTest(AppTestCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.snap_dir = Path(tmp.name)
        patcher = mock.patch.object(hawksight_app, "SNAP_DIR", self.snap_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(hawksight_app, "VideoSource", FakeVideo)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app._model.load = lambda: None

    def test_p_takes_a_photo_while_the_camera_is_on(self):
        self.app._on_start()
        self.assertTrue(_pump_until(
            self.app, lambda: self.app._last_frame is not None))
        self.app.focus_force()
        self.app.update()
        self.app.event_generate("<KeyPress-p>")
        self.app.update()
        self.assertEqual(len(list(self.snap_dir.glob("*.jpg"))), 1)
        self.assertIn("Photo saved", self.app._sv_status.get())


class SnapshotTest(AppTestCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.snap_dir = Path(tmp.name) / "snapshots"
        patcher = mock.patch.object(hawksight_app, "SNAP_DIR", self.snap_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app._last_frame = np.zeros((10, 10, 3), dtype=np.uint8)

    def test_quick_snapshots_do_not_overwrite_each_other(self):
        self.app._on_snapshot()
        self.app._on_snapshot()
        self.assertEqual(len(list(self.snap_dir.glob("*.jpg"))), 2)

    def test_failed_write_is_reported(self):
        with mock.patch.object(hawksight_app.cv2, "imwrite", return_value=False):
            self.app._on_snapshot()
        self.assertIn("Could not save", self.log_text())
        self.assertNotIn("saved", self.log_text())


class KeyboardShortcutTest(AppTestCase):
    def press(self, key):
        self.app.focus_force()
        self.app.update()
        self.app.event_generate(f"<KeyPress-{key}>")
        self.app.update()

    def test_stop_key_does_nothing_while_stop_is_disabled(self):
        self.press("x")
        self.assertEqual(self.app._sv_status.get(), "Ready")

    def test_photo_key_does_nothing_while_camera_is_off(self):
        self.app._last_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(hawksight_app, "SNAP_DIR", Path(tmp)), \
                mock.patch.object(self.app, "_log_write") as log:
            self.press("p")
            self.assertEqual(list(Path(tmp).iterdir()), [])
        log.assert_not_called()

    def test_start_key_works_with_caps_lock(self):
        def broken_load():
            raise RuntimeError("no model in tests")
        self.app._model.load = broken_load
        self.press("S")
        self.assertNotEqual(self.app._sv_status.get(), "Ready")
        _pump_until(self.app,
                    lambda: str(self.app._btn_start["state"]) == tk.NORMAL)


if __name__ == "__main__":
    unittest.main()
