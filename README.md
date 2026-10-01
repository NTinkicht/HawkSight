# HawkSight

Gas cylinder detection from a live camera, using YOLO. It comes as a desktop app
(`hawksight_app.py`) and a command-line tool (`hawksight.py`).

## Setup (once)

HawkSight runs in its own virtual environment, `.venv`, so its packages can't clash
with other Python projects on the machine. In particular, `opencv-python-headless`
(pulled in by tools like `roboflow`) has no window support and breaks the CLI preview.

```powershell
cd C:\Users\Majed\HawkSight
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

`requirements.txt` pins exact versions. Upgrade one package at a time and rerun the tests.

## Run

Always use the `.venv` Python:

```powershell
# Desktop app (no console window)
.venv\Scripts\pythonw.exe hawksight_app.py

# Command-line tool, with an OpenCV preview window (press Q to quit)
.venv\Scripts\python.exe hawksight.py --model hawksight_custom.pt
.venv\Scripts\python.exe hawksight.py --source 1 --conf 0.6 --model hawksight_custom.pt
.venv\Scripts\python.exe hawksight.py --source video.mp4 --model hawksight_custom.pt
```

Only load model weights (`.pt` files) you trust: loading one can run code.

**Desktop shortcut:** `HawkSight.lnk` on the desktop runs `HawkSight.vbs`, which starts
`.venv\Scripts\pythonw.exe hawksight_app.py` from this folder with no console window.

## Test

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

The tests use a fake camera and model, so no webcam or internet is needed. The GUI tests
open and close a Tk window.

## Keyboard shortcuts (desktop app)

`S` start · `X` stop · `P` snapshot (saved to `snapshots/`, which is git-ignored) ·
`F9` fullscreen · `Esc` exit fullscreen

---

## Project status and next steps  (last updated 1 Oct 2026)

> Read this section first when picking the project back up.

> **Missing files:** `train_hawksight.py` (the training script), `training_results/`
> (the training figures) and `generate_augmented_dataset.py` are **not in this repo**
> and are no longer in the old `OneDrive\Desktop\HawkSight_App` folder either. The
> training steps below need `train_hawksight.py` restored first.

### Where we are

- ✅ Desktop app works end-to-end (GUI, camera, detection, snapshots, log).
- ✅ README filled in (team / supervisor names still to add).
- ✅ Code prepared for a custom model:
  - `hawksight_app.py` loads `hawksight_custom.pt` automatically if the file
    exists, otherwise falls back to `yolov8n.pt`. The status bar shows which
    one is active ("HawkSight custom" vs "YOLOv8n (bottle proxy)").
  - `src/core.py` accepts all detections from a custom model and only
    filters to "bottle" when the stock model is loaded.
  - `train_hawksight.py` downloads a dataset, trains, and saves
    `hawksight_custom.pt` in one command.
- ✅ **Training complete (13 Sep 2026, re-run with early stopping).** A first
  local CPU attempt (`start_training.bat`) stalled overnight at epoch 6/15
  (too slow — hours per epoch). A first Colab GPU run finished 15 fixed
  epochs (10 Sep 2026, YOLOv8n). Re-run on Colab's free T4 GPU with
  `train_hawksight.py --epochs 100 --patience 3` (now an exposed CLI flag),
  which picked YOLO26n as the base model and early-stopped at epoch 17
  (best checkpoint from epoch 14) in about 1.5 hours. See "Results" below.
  Figures (confusion matrix, PR curve, results graph, sample batches) are
  saved in `training_results/`.
- ⚠️ **Validation mAP is very high (~0.995) but real-camera accuracy still
  needs checking.** CylinDeRS is studio/industrial photography; it may not
  generalize to webcam lighting/angles. If live accuracy is still poor,
  the fix is adding your own webcam photos of real cylinders to the
  training set (domain mismatch), not more epochs on the same source images.
- ✅ **Code-review fixes (1 Oct 2026, branch `Majed-Branch`):**
  - The app recovers when the model fails to load: it shows "Model error" and
    the reason, and re-enables START (before, it froze on "Loading model…").
  - Snapshots are git-ignored, no longer overwrite each other within the same
    second, and a failed save is reported instead of saying "saved".
  - The sidebar's **Total** card is now **Alerts**: it counts each confirmed
    sighting once, not every frame (one cylinder for 10 s used to read ~150).
  - The confidence threshold is limited to 5–95 % from the start
    (`--conf 2` used to silently detect nothing).
  - Live cameras are read on a background thread that keeps only the newest
    frame. On the dev webcam, worst-case frame age dropped from ~300 ms to ~33 ms.
  - Exact dependency versions are pinned in `requirements.txt`, and the app now
    runs from its own `.venv` (see Setup above).
  - First automated tests added in `tests/` (fake camera and model).

### Datasets (researched, all free, CC BY 4.0)

| Dataset | Images | Notes | Link |
|---|---|---|---|
| **CylinDeRS** (recommended, default in script) | 7,060 photos, 25,269 labelled cylinders | Real warehouses / industrial sites / outdoors; published research dataset; YOLO reaches ~0.91 mAP on it | <https://universe.roboflow.com/klearchos-stavrothanasopoulos-konstantinos-gkountakos-6jwgj/cylinders-iaq6n> |
| "gas cylinder" by lalith (`--dataset small`) | ~240 | 1 class "cylinder"; good for a quick test run | <https://universe.roboflow.com/lalith-jgg2e/gas-cylinder-n8tqc> |
| gas-cylinder-many-colors | 134 | 11 cylinder types by colour/size | <https://universe.roboflow.com/roboflow-i3c-thr5n/gas-cylinder-many-colors> |
| Gas cylinder detection (obj-dect) | 108 | Indian LPG cylinders | <https://universe.roboflow.com/obj-dect/gas-cylinder-detection> |

CylinDeRS paper: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11859601/>

Not useful (found while searching): the "Multimodal Gas Detection" sets on
Kaggle / Mendeley are gas-sensor readings, not cylinder images.

### How to train — step by step

> **Keep training out of HawkSight's `.venv`.** Training needs `roboflow`, which
> installs `opencv-python-headless` and breaks the CLI's preview window. Train on
> Colab (recommended) or in a separate environment, then copy `hawksight_custom.pt`
> into this folder.

**Step 1 — get a Roboflow API key (5 min, once)**
1. Go to <https://app.roboflow.com> and create a free account.
2. Click your workspace → **Settings** → **API Keys** → copy the Private key.

**Step 2 — quick test run on the laptop (20–60 min)**
In a separate training environment (not `.venv`), open a terminal in this folder and run:

```
python train_hawksight.py --key YOUR_KEY --dataset small
```

When it finishes it prints `mAP@50` and creates `hawksight_custom.pt`.
Launch `.venv\Scripts\python.exe hawksight_app.py`. The status bar should say
**HawkSight custom**. This proves the whole pipeline works.

**Step 3 — real model on the full dataset (do this for the final submission)**
Pick one:

- *Laptop, overnight (CPU, many hours):*
  ```
  python train_hawksight.py --key YOUR_KEY
  ```
- *Google Colab with free GPU (about 1 hour) — recommended:*
  1. Open <https://colab.research.google.com>, new notebook, Runtime →
     Change runtime type → **GPU (T4)**.
  2. Upload `train_hawksight.py` (folder icon on the left → upload).
  3. In a cell run: `!pip install -q ultralytics roboflow` then
     `!python train_hawksight.py --key YOUR_KEY`
  4. Download `hawksight_custom.pt` from the Colab file panel and put it in
     this folder.

**Step 4 — check and record results**
- Good result: mAP@50 above ~0.85. Write the numbers into the "Results"
  line below and into the capstone report.
- Training plots (confusion matrix, PR curve) are saved in
  `runs/hawksight/` — useful figures for the report.
- Test the app on a real cylinder (or a photo of one on a phone screen)
  and take snapshots for the report.

**Results:** mAP@50 = 0.995, mAP@50-95 = 0.992 (validation split of CylinDeRS,
YOLO26n, T4 GPU on Google Colab, 13 Sep 2026). Trained with `--epochs 100
--patience 3`; early-stopped at epoch 17 (best checkpoint from epoch 14).
Very high — test on real camera footage before trusting this number for the report.

### After training — remaining ideas (in priority order)

1. Test with drone footage / top-down angles; add drone frames to the
   dataset with `--data` if accuracy drops.
2. Add an audible alarm or notification when a cylinder is detected.
3. Export the detection log to CSV.
4. Optional: YOLOE zero-shot model (`model.set_classes(["gas cylinder"])`)
   as a comparison in the report.

### Still open from the 1 Oct 2026 code review

1. Make the CLI reuse `SystemController` so it gets the same 5-frame filter
   and defaults as the desktop app. Today the CLI shows one-frame false alarms
   and uses different defaults (conf 0.5 and `yolov8n.pt`, vs 0.65 and the custom model).
2. Keyboard shortcuts work even when their button is disabled: `P` after
   stopping saves an old frame, and `X` during model loading desyncs the UI.
   They also don't work with Caps Lock on.
3. Move per-frame resizing and colour conversion off the UI thread
   (measure FPS first).
4. Small clean-ups: remove the `sys.path.insert` lines, the unused
   `_divider(vert_pad=...)`, and duplicated button colours and strings.

### Useful commands

```
.venv\Scripts\python.exe hawksight_app.py                       # run the app
.venv\Scripts\python.exe hawksight.py --source video.mp4        # headless test on a video
.venv\Scripts\python.exe -m unittest discover -s tests -t .     # run the tests
python train_hawksight.py --key KEY --dataset small             # quick training test (training env)
python train_hawksight.py --key KEY                             # full training (training env)
python train_hawksight.py --key KEY --epochs 100                # longer training (training env)
python train_hawksight.py --data path\to\data.yaml              # train on own dataset (training env)
```
