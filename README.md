# HawkSight

HawkSight watches a live camera feed and alerts you when it sees a gas cylinder.
It uses a YOLO object-detection model and comes in two forms:

- **Desktop app** (`hawksight_app.py`): video panel, live stats, alert banner,
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

1. Pick the camera number and confidence threshold in the sidebar (default 65 %).
2. Pick a model from the **Detection model** dropdown: **HawkSight custom**
   (`hawksight_custom.pt`, only listed when the file exists) or **YOLOv8n (bottle
   proxy)**, which uses the stock COCO "bottle" class as a stand-in for cylinders.
   The dropdown is locked while the feed is running; press STOP to change model.
3. Press **START** or `S`. The camera takes a few seconds to open.

| Key | Action |
|-----|--------|
| `S` | Start |
| `X` | Stop |
| `P` | Save a snapshot to `snapshots/` (git-ignored) |
| `F9` | Toggle fullscreen |
| `Esc` | Exit fullscreen |

The sidebar shows:
- **Objects:** cylinders in the current frame
- **Confidence:** best confidence in the current frame
- **Runtime:** time since START
- **Alerts:** how many separate sightings this session

## How detection works

- A background thread reads the camera and keeps only the newest frame, so
  detection never works on old, buffered video.
- Each frame goes through the model. A cylinder counts only after it has been
  seen in **5 frames in a row**, which filters out one-frame false alarms.
- Confirmed cylinders get an orange box and confidence label, and the alert
  banner and video border pulse.
- If the model fails to load or the camera can't open, the app shows the error
  and the reason in the log, and START works again.

## Tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

## Known issues

- In the command-line preview, the window title shows `\u2014` instead of a dash.
