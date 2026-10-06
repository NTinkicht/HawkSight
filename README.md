# HawkSight

HawkSight watches a live camera feed and alerts you when it sees a gas cylinder.
It uses a YOLO object-detection model and comes in two forms:

- **Desktop app** (`hawksight_app.py`): video panel, a red detection light with "How sure",
  snapshots, and a detection log.
- **Command-line tool** (`hawksight.py`): a plain OpenCV preview window.

## Project layout

| Path | What it is |
|------|------------|
| `hawksight_app.py` | Desktop app (tkinter) |
| `hawksight.py` | Command-line tool |
| `src/core.py` | Camera capture, detection, frame annotation, background processing |
| `hawksight_custom.pt` | Fine-tuned gas-cylinder model, used by default (see [Model](#model)) |
| `assets/` | Logo and window icon |
| `tests/` | Automated tests (fake camera and model; no webcam or internet needed) |
| `requirements.txt` | Exact package versions |

## Model

`hawksight_custom.pt` is **YOLOv8n fine-tuned on a Roboflow gas-cylinder dataset**.

| | |
|---|---|
| Base model | YOLOv8n |
| Classes | 1: `gas_cylinder` |
| Accuracy | **71.6% mAP50-95** on a held-out test set |

The app shows each detection as "Gas Cylinder". You can also run the stock
**YOLOv8n** model (`yolov8n.pt`), which uses the COCO "bottle" class as a rough
stand-in for cylinders. It is also the fallback when `hawksight_custom.pt` is missing.

`yolov8n.pt` is not stored in the repo. The first time you start with YOLOv8n,
HawkSight downloads it (about 6 MB) from the official Ultralytics GitHub release
and shows "Downloading yolov8n.pt…". This needs an internet connection once; if the
download fails, the log says so and START works again.

## Setup (once)

HawkSight runs in its own virtual environment so its packages can't clash with
other Python projects:

```powershell
cd C:\Users\Majed\HawkSight
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Use `opencv-python`, never `opencv-python-headless`. The headless build has no
window support, so the command-line preview fails with it installed.

## Run

```powershell
# Desktop app (no console window)
.venv\Scripts\pythonw.exe hawksight_app.py

# Command-line tool (press Q in the preview window to quit)
.venv\Scripts\python.exe hawksight.py
.venv\Scripts\python.exe hawksight.py --source 1 --conf 0.5
.venv\Scripts\python.exe hawksight.py --source video.mp4
```

The command-line tool uses the same detection pipeline and defaults as the desktop app.

| Option | Meaning | Default |
|--------|---------|---------|
| `--source` | Camera number (0, 1, …) or a video file path | `0` |
| `--model` | Model weights file | `hawksight_custom.pt` (`yolov8n.pt` if missing) |
| `--conf` | Confidence threshold, limited to 0.05–0.95 | `0.65` |

Only load model files you trust: loading a `.pt` file can run code.

**Desktop shortcut:** `HawkSight.lnk` on the desktop runs `HawkSight.vbs`, which starts
the desktop app with the `.venv` Python.

## Using the desktop app

1. Under **Camera**, pick a camera. HawkSight looks for cameras when it opens and
   lists each one as "Camera 0", "Camera 1", … (the number Windows gives it).
   Camera 0 is shown as "(laptop)" because on a laptop it is the built-in camera.
   If no camera is found, HawkSight still offers the laptop camera, and on Windows
   it retries with Media Foundation when DirectShow can't open a camera. Plugged
   in a camera later? Press **⟳ Find cameras**. You can switch cameras while the
   camera is on: the old one is released and the new one starts (a camera takes a
   few seconds to open).
2. Under **What to look for**, pick a detector:
   **HawkSight (best for cylinders)** (`hawksight_custom.pt`, only listed when
   the file exists) or **Basic YOLOv8n (spots bottles)**, which uses the stock
   COCO "bottle" class as a stand-in for cylinders. This choice is locked while
   the camera is on; press STOP to change it.
3. **How sure before it alerts** sets the confidence threshold (default 65 %).
4. Press the green **START** button (or `S`). The camera takes a few seconds to open.

| Key | Action |
|-----|--------|
| `S` | Start |
| `X` | Stop |
| `P` | Take a photo while the camera is on, saved to `snapshots/` (git-ignored) |
| `F9` | Toggle full screen |
| `Esc` | Exit full screen |

There is no photo button; use `P`. The status bar at the bottom confirms each photo.

The sidebar has these parts, from the top:
- **Light + How sure:** a small light that blinks red while a gas cylinder is
  spotted, next to how sure the detector is (for example 87 %). Grey means
  nothing is spotted or the camera is off.
- **Camera on / off:** START and STOP.
- **Settings:** **What to look for**, **Camera** with **⟳ Find cameras**, and
  **How sure before it alerts**.
- **Show log:** hidden until clicked. Lists sightings, photos, detector and
  camera changes, and problems.

If something goes wrong (camera won't open, detector won't load or download),
the video area says what happened and what to do next in plain words.

## How detection works

- A background thread reads the camera and keeps only the newest frame, so
  detection never works on old, buffered video.
- Each frame goes through the model. A cylinder counts only after it has been
  seen in **5 frames in a row**, which filters out one-frame false alarms.
- Confirmed cylinders get a green box and confidence label, and the alert
  light and the video border blink.
- If the model fails to load or the camera can't open, the video area explains
  the problem, the log has the details, and START works again.

## Tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

## Known issues

- In the command-line preview, the window title shows `\u2014` instead of a dash.
- The Camera dropdown shows camera numbers, not device names: OpenCV can't read
  names, and listing them would need an extra package. Up to 6 cameras (0–5)
  are checked.
