"""GUI tests for HawkSightApp. Skipped when Tk can't open a window."""

import time
import tkinter as tk
import unittest

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


if __name__ == "__main__":
    unittest.main()
