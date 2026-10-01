# Implementation Plan: HawkSight review fixes

## Overview
Fix six issues from the code review, one at a time, in the order the user chose:
#1 model-load freeze, #6 snapshot safety, #3 misleading "Total" counter,
#9 confidence clamp, #4 camera lag, #2 dependency pinning.
Each fix gets its own test or quick check, its own commit, and a push to `Majed-Branch`.

## Architecture Decisions
- **Tests use stdlib `unittest`** (pytest isn't installed, and adding a dependency
  just for tests isn't worth it). Run with `python -m unittest discover -s tests -v`.
- **Fakes, not real hardware:** the tests swap in fake cameras and models, so they run
  with no webcam and no internet.
- **GUI smoke check:** each task also builds `HawkSightApp`, pumps its events once,
  and destroys it, which proves the window still starts.
- **The background thread no longer touches tkinter:** it stores its result and the
  main thread polls for it (tkinter isn't thread-safe).

## Task List

### Task 1: The app recovers when the model fails to load (#1)
- [x] An exception in `model.load()` or `controller.start()` re-enables START and the
      model buttons, shows an error status, and writes the reason to the log.
- [x] The worker thread no longer calls `self.after(...)`.
- Verify: unit test with a model whose `load()` raises; GUI smoke check.
- Files: `hawksight_app.py`, `tests/test_app.py` (new). Size: S

### Task 2: Safer snapshots (#6)
- [x] `snapshots/` is in `.gitignore`.
- [x] Two snapshots taken in the same second don't overwrite each other (milliseconds in the name).
- [x] A failed `cv2.imwrite` logs a warning instead of "saved".
- Verify: unit tests using a temp folder; GUI smoke check.
- Files: `.gitignore`, `hawksight_app.py`, `tests/test_app.py`. Size: S

### Task 3: "Total" counts alerts, not frames (#3)
- [x] Total goes up once each time a confirmed detection starts, not once per frame.
- [x] The streak resets on `start()`; the card label says "Alerts".
- Verify: unit test feeding frames through the controller; GUI smoke check.
- Files: `src/core.py`, `hawksight_app.py`, `tests/test_core.py` (new). Size: S

### Checkpoint A (after tasks 1–3)
- [x] All tests pass, the GUI starts, and the CLI `--help` works.

### Task 4: The confidence limit applies from the start (#9)
- [x] `DetectionModel(conf=2)` gives 0.95, and `conf=-1` gives 0.05.
- Verify: unit test.
- Files: `src/core.py`, `tests/test_core.py`. Size: XS

### Task 5: Reduce camera lag (#4)
- [x] Live cameras are read on a background thread that keeps only the newest frame,
      so the app processes recent frames. (The planned 1-frame buffer setting was
      dropped: the DirectShow driver ignores it.) Video files still read every frame.
- Verify: unit tests with a fake camera; real webcam measured (worst-case frame
  age ~300 ms → ~33 ms with a simulated 300 ms detection step).
- Files: `src/core.py`, `tests/test_core.py`. Size: XS

### Task 6: Pin dependencies (#2)
- [x] `requirements.txt` pins the exact versions that are installed and working.
- [x] The CLI `--model` help warns to load only trusted weight files.
- Verify: `pip install --dry-run -r requirements.txt` resolves; tests pass.
- Files: `requirements.txt`, `hawksight.py`. Size: XS

### Checkpoint B (complete)
- [x] All tests pass, the GUI starts, and everything is pushed to `Majed-Branch`.

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Buffer-size setting is ignored by some camera backends | Med | Harmless if ignored; flag for a real-camera measurement |
| GUI tests need a display | Low | Runs on the Windows desktop; skip the GUI tests if Tk can't start |
| Push needs GitHub auth | Low | Stop and report if the push fails |

## Open Questions
- Should Total count **alerts** (chosen) or **cylinders per alert**? Alerts is simplest and honest.
