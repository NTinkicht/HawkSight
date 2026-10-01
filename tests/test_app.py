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


def _pump_until(app, condition, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.update()
        if condition():
            return True
        time.sleep(0.02)
    return False


class AppTestCase(unittest.TestCase):
    def setUp(self):
        try:
            self.app = HawkSightApp()
        except tk.TclError as exc:
            self.skipTest(f"Tk unavailable: {exc}")
        self.app.update()

    def tearDown(self):
        self.app.on_close()

    def log_text(self):
        return self.app._log.get("1.0", tk.END)


class StartupTest(AppTestCase):
    def test_window_starts_idle(self):
        self.assertEqual(self.app._sv_status.get(), "Idle")
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
        self.assertEqual(str(self.app._btn_model_yolo["state"]), tk.NORMAL)
        self.assertEqual(self.app._sv_status.get(), "Model error")
        self.assertIn("weights file is corrupt", self.log_text())
        self.assertFalse(self.app._controller.is_running)


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
        self.assertEqual(self.app._sv_status.get(), "Idle")

    def test_snapshot_key_does_nothing_while_snapshot_is_disabled(self):
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
        self.assertNotEqual(self.app._sv_status.get(), "Idle")
        _pump_until(self.app,
                    lambda: str(self.app._btn_start["state"]) == tk.NORMAL)


if __name__ == "__main__":
    unittest.main()
