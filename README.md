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
| `hawksight_custom.pt` | Custom-trained cylinder model, used by default |
| `assets/` | Logo and window icon |
| `tests/` | Automated tests (fake camera and model; no webcam or internet needed) |
| `requirements.txt` | Exact package versions |

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
.venv\Scripts\python.exe hawksight.py --model hawksight_custom.pt
.venv\Scripts\python.exe hawksight.py --model hawksight_custom.pt --source 1 --conf 0.6
.venv\Scripts\python.exe hawksight.py --model hawksight_custom.pt --source video.mp4
```

| Option | Meaning | Default |
|--------|---------|---------|
| `--source` | Camera number (0, 1, …) or a video file path | `0` |
| `--model` | Model weights file | `yolov8n.pt` |
| `--conf` | Confidence threshold, limited to 0.05–0.95 | `0.5` |

Only load model files you trust: loading a `.pt` file can run code.

**Desktop shortcut:** `HawkSight.lnk` on the desktop runs `HawkSight.vbs`, which starts
the desktop app with the `.venv` Python.

## Using the desktop app

1. Pick the camera number and confidence threshold in the sidebar (default 65 %).
2. Choose the model: **Custom** (`hawksight_custom.pt`) or **YOLOv8n**, which
   uses the stock COCO "bottle" class as a stand-in for cylinders.
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

- The command-line tool doesn't use the 5-frame filter yet, so it can show
  one-frame false alarms. Its defaults (`yolov8n.pt`, 0.5) also differ from the
  desktop app's (custom model, 0.65).
- In the command-line preview, the window title shows `\u2014` instead of a dash.
