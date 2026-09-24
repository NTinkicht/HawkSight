"""HawkSight – Gas Cylinder Detection  (GUI entry point)"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).parent))
from src.core import (
    DetectionResult, VideoSource, DetectionModel,
    FrameProcessor, SystemController,
)

# ─── Paths ───────────────────────────────────────────────────────────────────

_ROOT      = Path(__file__).parent
ASSETS_DIR = _ROOT / "assets"
LOGO_PNG   = ASSETS_DIR / "HawkSight_Logo.png"
LOGO_ICO   = ASSETS_DIR / "HawkSight_Logo.ico"
# Two selectable model checkpoints — user can toggle between them at runtime.
YOLO_PT   = _ROOT / "yolov8n.pt"
CUSTOM_PT = _ROOT / "hawksight_custom.pt"
MODEL_PATHS  = {"yolo": YOLO_PT, "custom": CUSTOM_PT}
MODEL_LABELS = {"yolo": "YOLOv8n (bottle proxy)", "custom": "HawkSight custom"}
# Prefer the purpose-trained model if train_hawksight.py has produced one.
DEFAULT_MODEL_KEY = "custom" if CUSTOM_PT.exists() else "yolo"
SNAP_DIR = _ROOT / "snapshots"

# ─── Fonts ───────────────────────────────────────────────────────────────────

FONT = "Segoe UI"
MONO = "Consolas"


# ─── DisplayManager ──────────────────────────────────────────────────────────

class DisplayManager:
    def __init__(self, canvas: tk.Canvas):
        self._canvas = canvas
        self._photo:  Optional[ImageTk.PhotoImage] = None
        self._img_id: Optional[int] = None

    def _dims(self):
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        return (w or 800), (h or 500)

    def render(self, bgr_frame: np.ndarray):
        rgb   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)
        w, h  = self._dims()
        if self._img_id is not None:
            self._canvas.itemconfig(self._img_id, image=photo)
            self._canvas.coords(self._img_id, w // 2, h // 2)
        else:
            self._img_id = self._canvas.create_image(
                w // 2, h // 2, anchor=tk.CENTER, image=photo
            )
        self._photo = photo  # prevent GC

    def draw_placeholder(self):
        self._canvas.delete("all")
        self._img_id = None
        w, h = self._dims()
        cx, cy = w // 2, h // 2

        # Camera body — visible against the softer dark background
        bw, bh = 144, 92
        self._canvas.create_rectangle(
            cx - bw // 2, cy - bh // 2,
            cx + bw // 2, cy + bh // 2,
            fill="#2e2e30", outline="#4a4a4c", width=2,
        )
        # Viewfinder bump
        self._canvas.create_rectangle(
            cx - 26, cy - bh // 2 - 16,
            cx + 26, cy - bh // 2,
            fill="#2e2e30", outline="#4a4a4c", width=2,
        )
        # Shutter button
        self._canvas.create_oval(
            cx + bw // 2 - 30, cy - bh // 2 + 8,
            cx + bw // 2 - 12, cy - bh // 2 + 24,
            fill="#3a3a3c", outline="#545456",
        )
        # Lens ring outer
        r = 32
        self._canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill="#262628", outline="#464648", width=2,
        )
        # Lens ring middle
        r2 = 20
        self._canvas.create_oval(
            cx - r2, cy - r2, cx + r2, cy + r2,
            fill="#2c2c2e", outline="#5a5a5c", width=1,
        )
        # Lens centre — brand orange dot
        r3 = 7
        self._canvas.create_oval(
            cx - r3, cy - r3, cx + r3, cy + r3,
            fill="#e07818", outline="",
        )

        # Prompt text — visible on the softer background
        self._canvas.create_text(
            cx, cy + bh // 2 + 30,
            text="Ready to detect",
            fill="#6a6a6c", font=(FONT, 13, "bold"),
        )
        self._canvas.create_text(
            cx, cy + bh // 2 + 52,
            text="Press  START  or  S  to begin",
            fill="#545456", font=(FONT, 9),
        )

    def show_message(self, text: str):
        self._canvas.delete("all")
        self._img_id = None
        w, h = self._dims()
        self._canvas.create_text(
            w // 2, h // 2,
            text=text, fill="#6a6a6c", font=(FONT, 13, "bold"),
        )


# ─── HawkSightApp ─────────────────────────────────────────────────────────────

class HawkSightApp(tk.Tk):
    # ── Palette — "Soft Dark" (easy on the eyes) ─────────────────────────────
    BG        = "#1e1e20"   # lifted from pitch-black to comfortable dark gray
    HEADER    = "#252527"
    PANEL     = "#2a2a2c"
    CARD_BG   = "#323234"
    SEP       = "#3e3e40"
    ORANGE    = "#e07818"   # softer, less saturated than #ff6b00
    ORANGE_LT = "#e89830"
    ORANGE_DK = "#a85e10"
    GREEN     = "#27ae67"   # muted, not neon
    GREEN_DK  = "#186a3b"
    RED       = "#c0392b"   # calmer red
    RED_DK    = "#6e1c16"
    BLUE      = "#2e86c1"   # softer blue
    PURPLE    = "#8e44ad"
    FG        = "#d8d8d8"
    FG_MID    = "#9a9a9a"   # brighter — easier to read
    FG_DIM    = "#6a6a6a"   # brighter — no more squinting

    SIDEBAR_W = 278

    def __init__(self):
        super().__init__()
        self.title("HawkSight — Gas Cylinder Detection")
        self.configure(bg=self.BG)
        self.resizable(True, True)
        self.minsize(940, 620)
        self.geometry("1220x840")

        # State
        self._last_frame:     Optional[np.ndarray] = None
        self._last_log_count: int   = -1
        self._last_log_time:  float = 0.0
        self._poll_id:  Optional[str] = None
        self._timer_id: Optional[str] = None
        self._alert_id: Optional[str] = None
        self._badge_id: Optional[str] = None
        self._alert_state = False
        self._badge_state = True
        self._model_key = DEFAULT_MODEL_KEY
        self._is_fullscreen = False

        self._load_icon()
        self._build_ui()
        self._init_backend()
        self._bind_keys()
        self.after(160, self._display.draw_placeholder)

    # ── Window icon ───────────────────────────────────────────────────────────

    def _load_icon(self):
        try:
            self.iconbitmap(str(LOGO_ICO))
        except Exception:
            try:
                img = Image.open(LOGO_PNG)
                self._icon_ref = ImageTk.PhotoImage(img)
                self.iconphoto(True, self._icon_ref)
            except Exception:
                pass

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def _bind_keys(self):
        self.bind("<s>", lambda _: self._on_start())
        self.bind("<x>", lambda _: self._on_stop())
        self.bind("<p>", lambda _: self._on_snapshot())
        self.bind("<F9>", lambda _: self._toggle_fullscreen())
        self.bind("<Escape>", lambda _: self._exit_fullscreen())

    def _toggle_fullscreen(self):
        self._is_fullscreen = not self._is_fullscreen
        self.attributes("-fullscreen", self._is_fullscreen)

    def _exit_fullscreen(self):
        if self._is_fullscreen:
            self._is_fullscreen = False
            self.attributes("-fullscreen", False)

    # ── UI construction ───────────────────────────────────────────────────────
    # Status bar must be packed BOTTOM before the expandable main area.

    def _build_ui(self):
        self._build_header()
        self._build_status_bar()   # pack bottom first
        self._build_main_area()    # then middle (expands)

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self, bg=self.HEADER, height=68)
        hdr.pack(fill=tk.X, side=tk.TOP)
        hdr.pack_propagate(False)

        # Orange accent line at bottom of header
        tk.Frame(hdr, bg=self.ORANGE, height=2).pack(side=tk.BOTTOM, fill=tk.X)

        inner = tk.Frame(hdr, bg=self.HEADER)
        inner.pack(fill=tk.BOTH, expand=True)

        # Logo image
        self._hdr_logo = None
        try:
            raw = Image.open(LOGO_PNG).resize((42, 42), Image.LANCZOS)
            self._hdr_logo = ImageTk.PhotoImage(raw)
            tk.Label(inner, image=self._hdr_logo,
                     bg=self.HEADER).pack(side=tk.LEFT, padx=(18, 10), pady=12)
        except Exception:
            pass

        # Brand name + subtitle
        brand = tk.Frame(inner, bg=self.HEADER)
        brand.pack(side=tk.LEFT, pady=10)

        name_row = tk.Frame(brand, bg=self.HEADER)
        name_row.pack(anchor=tk.W)
        tk.Label(name_row, text="HAWK", font=(FONT, 21, "bold"),
                 bg=self.HEADER, fg=self.ORANGE).pack(side=tk.LEFT)
        tk.Label(name_row, text="SIGHT", font=(FONT, 21, "bold"),
                 bg=self.HEADER, fg=self.FG).pack(side=tk.LEFT)

        tk.Label(brand, text="Gas Cylinder Detection System",
                 font=(FONT, 9), bg=self.HEADER,
                 fg=self.FG_DIM).pack(anchor=tk.W, pady=(1, 0))

        # Right side: FPS counter + status badge
        right = tk.Frame(inner, bg=self.HEADER)
        right.pack(side=tk.RIGHT, padx=22)

        # FPS block
        fps_blk = tk.Frame(right, bg=self.HEADER)
        fps_blk.pack(side=tk.LEFT, padx=(0, 28))
        tk.Label(fps_blk, text="FPS",
                 font=(FONT, 7, "bold"),
                 bg=self.HEADER, fg=self.FG_DIM).pack()
        self._sv_fps = tk.StringVar(value="—")
        tk.Label(fps_blk, textvariable=self._sv_fps,
                 font=(MONO, 16, "bold"),
                 bg=self.HEADER, fg=self.FG_MID).pack()

        # Status badge block
        badge_blk = tk.Frame(right, bg=self.HEADER)
        badge_blk.pack(side=tk.LEFT)
        tk.Label(badge_blk, text="STATUS",
                 font=(FONT, 7, "bold"),
                 bg=self.HEADER, fg=self.FG_DIM).pack()
        self._badge = tk.Label(badge_blk, text="● OFFLINE",
                               font=(FONT, 11, "bold"),
                               bg=self.HEADER, fg=self.RED)
        self._badge.pack()

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self):
        bar = tk.Frame(self, bg=self.HEADER, height=30)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)

        tk.Frame(bar, bg=self.SEP, height=1).pack(side=tk.TOP, fill=tk.X)

        self._sv_status = tk.StringVar(value="Idle")
        tk.Label(bar, textvariable=self._sv_status,
                 font=(FONT, 8, "bold"),
                 bg=self.HEADER, fg=self.FG_MID).pack(
            side=tk.LEFT, padx=16, pady=6
        )

        tk.Label(bar,
                 text="S = Start    X = Stop    P = Snapshot    F9 = Fullscreen",
                 font=(FONT, 8), bg=self.HEADER,
                 fg=self.FG_DIM).pack(side=tk.LEFT, padx=16)

        self._sv_footer = tk.StringVar(
            value=f"{MODEL_LABELS[self._model_key]} · HawkSight v2.0 · CIS 4913"
        )
        tk.Label(bar, textvariable=self._sv_footer,
                 font=(FONT, 8), bg=self.HEADER,
                 fg=self.SEP).pack(side=tk.RIGHT, padx=16)

    # ── Main area (video + sidebar) ───────────────────────────────────────────

    def _build_main_area(self):
        main = tk.Frame(self, bg=self.BG)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 8))

        # Sidebar first (RIGHT), then video fills remaining LEFT space
        self._build_sidebar(main)
        self._build_video_panel(main)

    # ── Video panel ───────────────────────────────────────────────────────────

    def _build_video_panel(self, parent):
        outer = tk.Frame(parent, bg=self.BG)
        outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self._border = tk.Frame(outer, bg=self.ORANGE)
        self._border.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(
            self._border, bg="#1a1a1c", highlightthickness=0
        )
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=self.PANEL, width=self.SIDEBAR_W)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        sb.pack_propagate(False)

        self._build_stat_cards(sb)
        self._divider(sb)
        self._build_alert_banner(sb)
        self._divider(sb)
        self._build_controls(sb)
        self._divider(sb)
        self._build_settings(sb)
        self._divider(sb)
        self._build_log(sb)          # expands to fill remaining space

    def _divider(self, parent, vert_pad=0):
        tk.Frame(parent, bg=self.SEP, height=1).pack(
            fill=tk.X, padx=10, pady=vert_pad
        )

    # ── Stat cards (2 × 2 grid) ───────────────────────────────────────────────

    def _make_card(self, parent, title: str, var: tk.StringVar,
                   accent: str, val_color: str) -> tk.Frame:
        f = tk.Frame(parent, bg=self.CARD_BG)
        tk.Frame(f, bg=accent, height=3).pack(fill=tk.X)
        tk.Label(f, text=title.upper(),
                 font=(FONT, 7, "bold"),
                 bg=self.CARD_BG, fg=self.FG_DIM).pack(pady=(5, 0))
        tk.Label(f, textvariable=var,
                 font=(MONO, 19, "bold"),
                 bg=self.CARD_BG, fg=val_color).pack(pady=(0, 7))
        return f

    def _build_stat_cards(self, parent):
        self._sv_objects = tk.StringVar(value="0")
        self._sv_best    = tk.StringVar(value="—")
        self._sv_runtime = tk.StringVar(value="00:00")
        self._sv_total   = tk.StringVar(value="0")

        grid = tk.Frame(parent, bg=self.PANEL)
        grid.pack(fill=tk.X, padx=8, pady=8)

        c1 = self._make_card(grid, "Objects",    self._sv_objects, self.ORANGE, self.ORANGE)
        c2 = self._make_card(grid, "Confidence", self._sv_best,    self.GREEN,  self.GREEN)
        c3 = self._make_card(grid, "Runtime",    self._sv_runtime, self.BLUE,   self.FG)
        c4 = self._make_card(grid, "Total",      self._sv_total,   self.PURPLE, self.FG)

        g = 5
        c1.grid(row=0, column=0, sticky="nsew", padx=(0, g), pady=(0, g))
        c2.grid(row=0, column=1, sticky="nsew", padx=(g, 0), pady=(0, g))
        c3.grid(row=1, column=0, sticky="nsew", padx=(0, g), pady=(g, 0))
        c4.grid(row=1, column=1, sticky="nsew", padx=(g, 0), pady=(g, 0))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

    # ── Alert banner ──────────────────────────────────────────────────────────

    def _build_alert_banner(self, parent):
        self._alert_frame = tk.Frame(parent, bg=self.CARD_BG, height=52)
        self._alert_frame.pack(fill=tk.X, padx=8, pady=8)
        self._alert_frame.pack_propagate(False)

        self._alert_dot = tk.Label(
            self._alert_frame, text="●",
            font=(FONT, 11), bg=self.CARD_BG, fg=self.FG_DIM,
        )
        self._alert_dot.place(relx=0.13, rely=0.5, anchor=tk.CENTER)

        self._alert_lbl = tk.Label(
            self._alert_frame, text="MONITORING",
            font=(FONT, 10, "bold"), bg=self.CARD_BG, fg=self.FG_DIM,
        )
        self._alert_lbl.place(relx=0.58, rely=0.5, anchor=tk.CENTER)

    # ── Controls ──────────────────────────────────────────────────────────────

    def _build_controls(self, parent):
        ctrl = tk.Frame(parent, bg=self.PANEL)
        ctrl.pack(fill=tk.X, padx=8, pady=8)

        cfg = dict(font=(FONT, 11, "bold"), relief=tk.FLAT,
                   cursor="hand2", bd=0)

        self._btn_start = tk.Button(
            ctrl, text="▶   START",
            bg=self.GREEN, fg="white",
            activebackground=self.GREEN_DK, activeforeground="white",
            command=self._on_start, **cfg,
        )
        self._btn_start.pack(fill=tk.X, pady=(0, 5), ipady=10)

        self._btn_stop = tk.Button(
            ctrl, text="■   STOP",
            bg="#3a2424", fg="#a06060",
            activebackground=self.RED_DK, activeforeground=self.RED,
            command=self._on_stop, state=tk.DISABLED, **cfg,
        )
        self._btn_stop.pack(fill=tk.X, pady=(0, 5), ipady=10)

        self._btn_snap = tk.Button(
            ctrl, text="◎   SNAPSHOT",
            bg="#242c38", fg="#6080a0",
            activebackground="#2a3a50", activeforeground=self.BLUE,
            command=self._on_snapshot, state=tk.DISABLED, **cfg,
        )
        self._btn_snap.pack(fill=tk.X, ipady=10)

        tk.Label(ctrl, text="S  ·  X  ·  P   keyboard shortcuts",
                 font=(FONT, 7), bg=self.PANEL,
                 fg=self.FG_DIM).pack(pady=(5, 0))

    # ── Settings ──────────────────────────────────────────────────────────────

    def _build_settings(self, parent):
        cfg = tk.Frame(parent, bg=self.PANEL)
        cfg.pack(fill=tk.X, padx=8, pady=8)

        # Detection model toggle
        row0 = tk.Frame(cfg, bg=self.PANEL)
        row0.pack(fill=tk.X, pady=(0, 9))
        tk.Label(row0, text="Detection model",
                 font=(FONT, 9), bg=self.PANEL, fg=self.FG_MID).pack(anchor=tk.W)

        toggle = tk.Frame(row0, bg=self.PANEL)
        toggle.pack(fill=tk.X, pady=(4, 0))
        toggle.columnconfigure(0, weight=1)
        toggle.columnconfigure(1, weight=1)

        seg_cfg = dict(font=(FONT, 9, "bold"), relief=tk.FLAT,
                       cursor="hand2", bd=0)
        self._btn_model_yolo = tk.Button(
            toggle, text="YOLOv8n",
            command=lambda: self._on_model_switch("yolo"), **seg_cfg,
        )
        self._btn_model_yolo.grid(row=0, column=0, sticky="ew", padx=(0, 3), ipady=6)

        self._btn_model_custom = tk.Button(
            toggle, text="Custom",
            command=lambda: self._on_model_switch("custom"), **seg_cfg,
        )
        self._btn_model_custom.grid(row=0, column=1, sticky="ew", padx=(3, 0), ipady=6)

        if not CUSTOM_PT.exists():
            self._btn_model_custom.config(
                state=tk.DISABLED,
                bg=self.PANEL, fg=self.FG_DIM,
            )

        self._refresh_model_buttons()

        # Camera source row
        row1 = tk.Frame(cfg, bg=self.PANEL)
        row1.pack(fill=tk.X, pady=(0, 7))
        tk.Label(row1, text="Camera source",
                 font=(FONT, 9), bg=self.PANEL, fg=self.FG_MID).pack(side=tk.LEFT)
        self._sv_source = tk.StringVar(value="0")
        tk.Spinbox(
            row1, from_=0, to=5,
            textvariable=self._sv_source,
            width=3, font=(FONT, 10), justify=tk.CENTER,
            bg=self.CARD_BG, fg=self.FG,
            buttonbackground=self.SEP, relief=tk.FLAT, bd=0,
            highlightthickness=1,
            highlightcolor=self.ORANGE, highlightbackground=self.SEP,
        ).pack(side=tk.RIGHT, ipady=3)

        # Confidence slider row
        row2 = tk.Frame(cfg, bg=self.PANEL)
        row2.pack(fill=tk.X)
        tk.Label(row2, text="Confidence threshold",
                 font=(FONT, 9), bg=self.PANEL, fg=self.FG_MID).pack(side=tk.LEFT)
        self._lbl_conf = tk.Label(row2, text="65%",
                                   font=(FONT, 9, "bold"),
                                   bg=self.PANEL, fg=self.ORANGE)
        self._lbl_conf.pack(side=tk.RIGHT)

        self._sv_conf = tk.DoubleVar(value=0.65)
        ttk.Scale(
            cfg, from_=0.05, to=0.95, orient=tk.HORIZONTAL,
            variable=self._sv_conf, command=self._on_conf_change,
        ).pack(fill=tk.X, pady=(5, 0))

        # Loading indicator
        self._lbl_loading = tk.Label(cfg, text="",
                                      font=(FONT, 8, "italic"),
                                      bg=self.PANEL, fg=self.ORANGE)
        self._lbl_loading.pack(pady=(4, 0))

    # ── Detection log ─────────────────────────────────────────────────────────

    def _build_log(self, parent):
        frame = tk.Frame(parent, bg=self.PANEL)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 10))

        hdr = tk.Frame(frame, bg=self.PANEL)
        hdr.pack(fill=tk.X, pady=(0, 4))
        tk.Label(hdr, text="DETECTION LOG",
                 font=(FONT, 7, "bold"),
                 bg=self.PANEL, fg=self.FG_DIM).pack(side=tk.LEFT)
        tk.Button(hdr, text="Clear", font=(FONT, 7),
                  bg=self.PANEL, fg=self.FG_DIM,
                  relief=tk.FLAT, cursor="hand2", bd=0,
                  command=self._clear_log).pack(side=tk.RIGHT)

        self._log = tk.Text(
            frame, bg=self.CARD_BG, fg=self.FG,
            font=(MONO, 8), relief=tk.FLAT,
            wrap=tk.WORD, state=tk.DISABLED,
            insertbackground=self.ORANGE,
            selectbackground="#4a3020",
        )
        self._log.tag_configure("detect", foreground=self.ORANGE_LT)
        self._log.tag_configure("clear",  foreground=self.FG_DIM)
        self._log.tag_configure("snap",   foreground=self.BLUE)
        self._log.tag_configure("warn",   foreground=self.RED)
        self._log.tag_configure("div",    foreground=self.SEP)

        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self._log.yview)
        self._log.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log.pack(fill=tk.BOTH, expand=True)

    # ── Backend ───────────────────────────────────────────────────────────────

    def _init_backend(self):
        self._model      = DetectionModel(MODEL_PATHS[self._model_key], conf=0.65)
        self._video      = VideoSource(0)
        self._processor  = FrameProcessor()
        self._controller = SystemController(
            self._video, self._model, self._processor
        )
        self._display = DisplayManager(self._canvas)

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _on_start(self):
        if self._controller.is_running:
            return
        self._btn_start.config(state=tk.DISABLED)
        self._btn_model_yolo.config(state=tk.DISABLED)
        if CUSTOM_PT.exists():
            self._btn_model_custom.config(state=tk.DISABLED)
        self._sv_status.set("Loading model…")
        self._lbl_loading.config(
            text=f"Loading {MODEL_LABELS[self._model_key]} weights…"
        )

        try:
            src = int(self._sv_source.get())
        except ValueError:
            src = 0

        self._model.conf = self._sv_conf.get()
        self._video      = VideoSource(src)
        self._controller = SystemController(
            self._video, self._model, self._processor
        )
        threading.Thread(target=self._load_and_start, daemon=True).start()

    def _load_and_start(self):
        if not self._model.is_loaded:
            self._model.load()
        ok = self._controller.start()
        self.after(0, self._post_start, ok)

    def _post_start(self, ok: bool):
        self._lbl_loading.config(text="")
        if ok:
            self._btn_stop.config(
                state=tk.NORMAL,
                bg=self.RED, fg="white",
                activebackground=self.RED_DK, activeforeground="white",
            )
            self._btn_snap.config(
                state=tk.NORMAL,
                bg=self.BLUE, fg="white",
                activebackground="#1f6da0", activeforeground="white",
            )
            self._sv_status.set("Running")
            self._badge.config(text="● LIVE", fg=self.GREEN)
            self._last_log_count = -1
            self._last_log_time  = 0.0
            self._log_write("── session started ──", "div")
            self._poll_id  = self.after(30,   self._poll_frames)
            self._timer_id = self.after(1000, self._tick_timer)
            self._badge_id = self.after(900,  self._pulse_live_badge)
        else:
            self._btn_start.config(state=tk.NORMAL)
            self._sv_status.set("Camera error")
            self._badge.config(text="● ERROR", fg=self.RED)
            self._log_write("⚠  Could not open camera source.", "warn")

    def _on_stop(self):
        for attr in ("_poll_id", "_timer_id", "_alert_id", "_badge_id"):
            after_id = getattr(self, attr)
            if after_id:
                self.after_cancel(after_id)
                setattr(self, attr, None)

        self._controller.stop()
        self._display.show_message(
            "Feed stopped  —  press  S  or  START  to resume"
        )
        self._btn_stop.config(
            state=tk.DISABLED,
            bg="#3a2424", fg="#a06060",
        )
        self._btn_snap.config(
            state=tk.DISABLED,
            bg="#242c38", fg="#6080a0",
        )
        self._btn_start.config(state=tk.NORMAL)
        self._btn_model_yolo.config(state=tk.NORMAL)
        if CUSTOM_PT.exists():
            self._btn_model_custom.config(state=tk.NORMAL)
        self._sv_status.set("Stopped")
        self._sv_fps.set("—")
        self._sv_runtime.set("00:00")
        self._badge.config(text="● OFFLINE", fg=self.RED)
        self._reset_alert()
        self._border.config(bg=self.ORANGE)
        self._log_write("── session ended ──", "div")
        self.title("HawkSight — Gas Cylinder Detection")

    def _on_snapshot(self):
        if self._last_frame is None:
            return
        SNAP_DIR.mkdir(exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SNAP_DIR / f"snap_{ts}.jpg"
        cv2.imwrite(str(path), self._last_frame)
        self._log_write(f"◎ snap_{ts}.jpg  saved", "snap")

    def _on_model_switch(self, key: str):
        if key == self._model_key or self._controller.is_running:
            return
        path = MODEL_PATHS[key]
        if not path.exists():
            return
        self._model_key = key
        self._model.switch(path)
        self._refresh_model_buttons()
        self._sv_footer.set(f"{MODEL_LABELS[key]} · HawkSight v2.0 · CIS 4913")
        self._log_write(f"◆ model switched → {MODEL_LABELS[key]}", "div")

    def _refresh_model_buttons(self):
        active, inactive = self.ORANGE, self.CARD_BG
        active_fg, inactive_fg = "white", self.FG_MID

        is_yolo = self._model_key == "yolo"
        self._btn_model_yolo.config(
            bg=active if is_yolo else inactive,
            fg=active_fg if is_yolo else inactive_fg,
            activebackground=active if is_yolo else self.SEP,
            activeforeground="white",
        )
        if self._btn_model_custom.cget("state") != tk.DISABLED:
            is_custom = self._model_key == "custom"
            self._btn_model_custom.config(
                bg=active if is_custom else inactive,
                fg=active_fg if is_custom else inactive_fg,
                activebackground=active if is_custom else self.SEP,
                activeforeground="white",
            )

    def _on_conf_change(self, _=None):
        v = self._sv_conf.get()
        self._lbl_conf.config(text=f"{v:.0%}")
        if self._model.is_loaded:
            self._model.conf = v

    def _on_canvas_resize(self, event=None):
        if self._controller.is_running:
            return
        status = self._sv_status.get()
        if status == "Idle":
            self._display.draw_placeholder()
        else:
            self._display.show_message(
                "Feed stopped  —  press  S  or  START  to resume"
            )

    # ── Frame polling ─────────────────────────────────────────────────────────

    def _poll_frames(self):
        data = self._controller.poll_frame()
        if data is not None:
            frame, result = data
            self._last_frame = frame
            w = self._canvas.winfo_width()
            h = self._canvas.winfo_height()
            resized = self._processor.resize_for_display(
                frame, w or 800, h or 500
            )
            self._display.render(resized)
            self._update_stats(result)
        if self._controller.is_running:
            self._poll_id = self.after(30, self._poll_frames)

    def _update_stats(self, result: DetectionResult):
        self._sv_objects.set(str(result.count))
        conf = result.best_confidence
        self._sv_best.set(f"{conf:.0%}" if conf > 0 else "—")
        self._sv_total.set(str(self._controller.total_detections))
        fps = self._controller.fps
        self._sv_fps.set(f"{fps:.1f}" if fps > 0 else "—")

        # Window title
        n = result.count
        if n > 0:
            word = "object" if n == 1 else "objects"
            self.title(f"HawkSight  —  {n} {word} detected")
        else:
            self.title("HawkSight — Gas Cylinder Detection")

        # Alert animation: start on first detection, stop when clear
        if result.count > 0:
            if self._alert_id is None:
                self._start_alert()
        else:
            if self._alert_id is not None:
                self._reset_alert()

        # Detection log — debounced (log on count change or every 2 s)
        now = time.monotonic()
        count_changed = result.count != self._last_log_count
        time_elapsed  = now - self._last_log_time >= 2.0
        if count_changed or (result.count > 0 and time_elapsed):
            if result.count > 0:
                ts   = datetime.now().strftime("%H:%M:%S")
                word = "cylinder" if result.count == 1 else "cylinders"
                self._log_write(
                    f"[{ts}]  {result.count} {word}  ({conf:.0%})", "detect"
                )
            elif self._last_log_count > 0:
                ts = datetime.now().strftime("%H:%M:%S")
                self._log_write(f"[{ts}]  — cleared", "clear")
            self._last_log_count = result.count
            self._last_log_time  = now

    def _tick_timer(self):
        if self._controller.is_running:
            secs = int(self._controller.runtime)
            m, s = divmod(secs, 60)
            h, m = divmod(m, 60)
            self._sv_runtime.set(
                f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
            )
            self._timer_id = self.after(1000, self._tick_timer)

    # ── Animations ────────────────────────────────────────────────────────────

    def _start_alert(self):
        self._alert_state = False
        self._alert_id = self.after(0, self._pulse_alert)

    def _reset_alert(self):
        if self._alert_id:
            self.after_cancel(self._alert_id)
            self._alert_id = None
        bg = self.CARD_BG
        self._alert_frame.config(bg=bg)
        self._alert_dot.config(bg=bg, fg=self.FG_DIM, text="●")
        self._alert_lbl.config(bg=bg, fg=self.FG_DIM, text="MONITORING")
        self._border.config(bg=self.ORANGE)

    def _pulse_alert(self):
        self._alert_state = not self._alert_state
        if self._alert_state:
            bg         = self.ORANGE
            dot_fg     = "white"
            lbl_fg     = "white"
            border_col = self.ORANGE_LT
        else:
            bg         = "#3c2010"   # softer dark amber — much easier than #2a0e00
            dot_fg     = self.ORANGE_LT
            lbl_fg     = self.ORANGE_LT
            border_col = self.ORANGE_DK
        self._alert_frame.config(bg=bg)
        self._alert_dot.config(bg=bg, fg=dot_fg, text="⚑")
        self._alert_lbl.config(bg=bg, fg=lbl_fg, text="CYLINDER DETECTED")
        self._border.config(bg=border_col)
        self._alert_id = self.after(750, self._pulse_alert)  # slower = calmer

    def _pulse_live_badge(self):
        if not self._controller.is_running:
            return
        self._badge_state = not self._badge_state
        self._badge.config(fg=self.GREEN if self._badge_state else self.GREEN_DK)
        self._badge_id = self.after(900, self._pulse_live_badge)

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _log_write(self, text: str, tag: str = ""):
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, text + "\n", tag or ())
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    def _clear_log(self):
        self._log.config(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)
        self._log.config(state=tk.DISABLED)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_close(self):
        self._controller.stop()
        self.destroy()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = HawkSightApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
