"""HawkSight – Gas Cylinder Detection  (GUI entry point)"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).parent))
from src.core import (
    CUSTOM_PT, DEFAULT_CONF, DEFAULT_MODEL, YOLO_PT,
    DetectionResult, VideoSource, DetectionModel,
    FrameProcessor, ReplayBuffer, SystemController, list_cameras,
)

# ─── Paths ───────────────────────────────────────────────────────────────────

_ROOT      = Path(__file__).parent
ASSETS_DIR = _ROOT / "assets"
LOGO_PNG   = ASSETS_DIR / "HawkSight_Logo.png"
LOGO_ICO   = ASSETS_DIR / "HawkSight_Logo.ico"
# Selectable model checkpoints, in dropdown order.
MODEL_PATHS  = {"custom": CUSTOM_PT, "yolo": YOLO_PT}
MODEL_LABELS = {"custom": "HawkSight (best for cylinders)",
                "yolo":   "Basic YOLOv8n (spots bottles)"}
DEFAULT_MODEL_KEY = "custom" if DEFAULT_MODEL == CUSTOM_PT else "yolo"
SNAP_DIR = _ROOT / "snapshots"
REPLAY_SECONDS = 15
# On a laptop, camera 0 is the built-in one. Used when the scan finds nothing.
LAPTOP_CAMERA = 0


def camera_label(index: int) -> str:
    return f"Camera {index} (laptop)" if index == LAPTOP_CAMERA else f"Camera {index}"


def available_model_keys() -> list[str]:
    """Models the dropdown offers: stock YOLOv8n is always available (it is
    downloaded on first use); the custom model only when its file exists."""
    return [k for k, p in MODEL_PATHS.items() if k == "yolo" or p.exists()]

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
            text="Ready!",
            fill="#a0a0a2", font=(FONT, 13, "bold"),
        )
        self._canvas.create_text(
            cx, cy + bh // 2 + 52,
            text="Press the green  START  button to begin",
            fill="#8a8a8c", font=(FONT, 9),
        )

    def show_message(self, title: str, hint: str = "", color: str = "#a0a0a2"):
        self._canvas.delete("all")
        self._img_id = None
        w, h = self._dims()
        self._canvas.create_text(
            w // 2, h // 2 - (14 if hint else 0),
            text=title, fill=color, font=(FONT, 16, "bold"),
        )
        if hint:
            self._canvas.create_text(
                w // 2, h // 2 + 18, width=max(w - 80, 200),
                text=hint, fill="#a0a0a2", font=(FONT, 10), justify=tk.CENTER,
            )


# ─── PillButton ──────────────────────────────────────────────────────────────

class PillButton(tk.Canvas):
    """A small rounded button (Tk buttons can't have round corners), with a
    hover glow and a dimmed disabled look. invoke() does nothing while it is
    disabled, like tk.Button, so keyboard shortcuts can call it safely."""

    def __init__(self, parent, text: str, command, *, parent_bg: str,
                 bg: str, fg: str, hover_bg: str, hover_fg: str,
                 off_bg: str, off_fg: str, font=None, padx: int = 14,
                 pady: int = 6):
        font = font or (FONT, 9, "bold")
        f = tkfont.Font(font=font)
        w = f.measure(text) + 2 * padx
        h = f.metrics("linespace") + 2 * pady
        super().__init__(parent, width=w, height=h, bg=parent_bg,
                         highlightthickness=0, bd=0)
        self._command = command
        self._colors = dict(bg=bg, fg=fg, hover_bg=hover_bg,
                            hover_fg=hover_fg, off_bg=off_bg, off_fg=off_fg)
        self._enabled = True
        self._hover   = False
        r = h // 2
        x1, y1, x2, y2 = 1, 1, w - 1, h - 1
        self._shape = self.create_polygon(
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
            x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
            smooth=True, outline="")
        self._text = self.create_text(w // 2, h // 2, text=text, font=font)
        self.bind("<Enter>", lambda _: self._set_hover(True))
        self.bind("<Leave>", lambda _: self._set_hover(False))
        self.bind("<ButtonRelease-1>", lambda _: self.invoke())
        self._paint()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        self._paint()

    def invoke(self):
        if self._enabled:
            self._command()

    def _set_hover(self, hover: bool):
        self._hover = hover
        self._paint()

    def _paint(self):
        c = self._colors
        if not self._enabled:
            bg, fg, cursor = c["off_bg"], c["off_fg"], ""
        elif self._hover:
            bg, fg, cursor = c["hover_bg"], c["hover_fg"], "hand2"
        else:
            bg, fg, cursor = c["bg"], c["fg"], "hand2"
        self.itemconfig(self._shape, fill=bg)
        self.itemconfig(self._text, fill=fg)
        self.config(cursor=cursor)


# ─── HawkSightApp ─────────────────────────────────────────────────────────────

class HawkSightApp(tk.Tk):
    # ── Palette — "Soft Dark" (easy on the eyes) ─────────────────────────────
    BG        = "#0c0c0e"   # near-black, not pure black
    HEADER    = "#111113"
    PANEL     = "#151517"
    CARD_BG   = "#1e1e20"
    SEP       = "#2a2a2c"
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
    FG_MID    = "#b4b4b4"   # secondary text, ~7:1 on PANEL
    FG_DIM    = "#9a9a9a"   # labels and hints, >= 4.5:1 on PANEL and CARD_BG

    # START / STOP: the button you can press next is lit; the other is dark.
    LIT  = {"start": "#2ecc71", "stop": "#e74c3c"}
    DARK = {"start": ("#0e2016", "#4d7d62"), "stop": ("#241010", "#85514c")}

    # Spacing scale (px): every gap in the layout is one of these.
    S1, S2, S3, S4 = 4, 8, 12, 16

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
        self._alert_id: Optional[str] = None
        self._badge_id: Optional[str] = None
        self._alert_state = False
        self._badge_state = True
        self._model_key = DEFAULT_MODEL_KEY
        self._cameras: list[int] = []        # indices found by the last scan
        self._camera:  Optional[int] = None  # selected camera index
        self._none_found = False             # last scan found no camera
        self._scanning = False
        self._starting = False
        self._is_fullscreen = False
        self._idle_msg: Optional[tuple] = None   # None = "Ready!" placeholder
        self._status_id: Optional[str] = None    # pending status-bar restore
        self._replay = ReplayBuffer(seconds=REPLAY_SECONDS)
        # Replay plays in the main video area until "Continue live".
        self._replay_frames: Optional[list] = None
        self._replay_i  = 0
        self._replay_id: Optional[str] = None

        self._load_icon()
        self._init_styles()
        self._build_ui()
        self._init_backend()
        self._bind_keys()
        self.after(160, self._display.draw_placeholder)
        self._scan_cameras()

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

    # ── ttk styles ────────────────────────────────────────────────────────────

    def _init_styles(self):
        # "clam" is the built-in theme that honours custom colours on Windows.
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Dark.TCombobox",
            fieldbackground=self.CARD_BG, background=self.CARD_BG,
            foreground=self.FG, arrowcolor=self.FG_MID,
            bordercolor=self.SEP, lightcolor=self.CARD_BG,
            darkcolor=self.CARD_BG, padding=(8, 5),
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", self.CARD_BG),
                             ("disabled", self.PANEL)],
            foreground=[("disabled", self.FG_DIM), ("readonly", self.FG)],
            selectbackground=[("readonly", self.CARD_BG)],
            selectforeground=[("readonly", self.FG)],
            bordercolor=[("focus", self.ORANGE)],
            arrowcolor=[("disabled", self.FG_DIM)],
        )
        style.configure("TScale", background=self.ORANGE,
                        troughcolor=self.CARD_BG, bordercolor=self.SEP,
                        lightcolor=self.ORANGE, darkcolor=self.ORANGE_DK)
        style.configure("Vertical.TScrollbar", background=self.SEP,
                        troughcolor=self.CARD_BG, bordercolor=self.CARD_BG,
                        arrowcolor=self.FG_MID, lightcolor=self.SEP,
                        darkcolor=self.SEP, gripcount=0)
        style.map("Vertical.TScrollbar",
                  background=[("pressed", self.ORANGE_DK),
                              ("active", self.FG_DIM), ("!active", self.SEP)])
        # The dropdown list is a plain Tk listbox, styled through the option db.
        self.option_add("*TCombobox*Listbox.background", self.CARD_BG)
        self.option_add("*TCombobox*Listbox.foreground", self.FG)
        self.option_add("*TCombobox*Listbox.selectBackground", self.ORANGE)
        self.option_add("*TCombobox*Listbox.selectForeground", "white")
        self.option_add("*TCombobox*Listbox.font", (FONT, 9))

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def _bind_keys(self):
        # Each shortcut presses its button; invoke() does nothing while the
        # button is disabled. Bind both cases so Caps Lock doesn't matter.
        for key, button in (("s", self._btn_start), ("x", self._btn_stop)):
            for k in (key, key.upper()):
                self.bind(f"<{k}>", lambda _, b=button: b.invoke())
        for key, button in (("p", self._btn_shot), ("r", self._btn_replay),
                            ("c", self._btn_continue)):
            for k in (key, key.upper()):
                self.bind(f"<{k}>", lambda _, b=button: b.invoke())
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
    # No fixed heights: rows size to their text, so nothing is clipped when
    # Windows display scaling is above 100 %.

    def _build_header(self):
        S2, S3, S4 = self.S2, self.S3, self.S4
        hdr = tk.Frame(self, bg=self.HEADER)
        hdr.pack(fill=tk.X, side=tk.TOP)

        # Orange accent line at bottom of header
        tk.Frame(hdr, bg=self.ORANGE, height=2).pack(side=tk.BOTTOM, fill=tk.X)

        inner = tk.Frame(hdr, bg=self.HEADER)
        inner.pack(fill=tk.BOTH, expand=True, padx=S4, pady=S2)

        # Second row: the settings, side by side, above the orange line.
        tk.Frame(hdr, bg=self.SEP, height=1).pack(fill=tk.X, padx=S4)
        strip = tk.Frame(hdr, bg=self.HEADER)
        strip.pack(fill=tk.X, padx=S4, pady=S2)
        self._build_settings(strip)

        # Logo image
        self._hdr_logo = None
        try:
            raw = Image.open(LOGO_PNG).resize((42, 42), Image.LANCZOS)
            self._hdr_logo = ImageTk.PhotoImage(raw)
            tk.Label(inner, image=self._hdr_logo,
                     bg=self.HEADER).pack(side=tk.LEFT, padx=(0, S3))
        except Exception:
            pass

        # Brand name + subtitle
        brand = tk.Frame(inner, bg=self.HEADER)
        brand.pack(side=tk.LEFT)

        name_row = tk.Frame(brand, bg=self.HEADER)
        name_row.pack(anchor=tk.W)
        tk.Label(name_row, text="HAWK", font=(FONT, 20, "bold"),
                 bg=self.HEADER, fg=self.ORANGE).pack(side=tk.LEFT)
        tk.Label(name_row, text="SIGHT", font=(FONT, 20, "bold"),
                 bg=self.HEADER, fg=self.FG).pack(side=tk.LEFT)

        tk.Label(brand, text="Spots gas cylinders on camera",
                 font=(FONT, 9), bg=self.HEADER,
                 fg=self.FG_MID).pack(anchor=tk.W)

        # Right side: FPS counter + status badge
        right = tk.Frame(inner, bg=self.HEADER)
        right.pack(side=tk.RIGHT)

        # Middle: Screenshot and Replay, centred in the space that is left.
        tools = tk.Frame(inner, bg=self.HEADER)
        tools.pack(side=tk.LEFT, expand=True)
        pill = dict(parent_bg=self.HEADER, bg=self.CARD_BG, fg=self.FG,
                    hover_bg=self.ORANGE, hover_fg="white",
                    off_bg=self.PANEL, off_fg="#77777a",
                    font=(FONT, 10, "bold"), padx=18, pady=8)
        self._btn_shot = PillButton(tools, "◉  Screenshot",
                                    self._on_snapshot, **pill)
        self._btn_shot.pack(side=tk.LEFT, padx=(0, S3))
        self._btn_replay = PillButton(
            tools, f"⏪  Replay last {REPLAY_SECONDS}s", self._on_replay, **pill)
        self._btn_replay.pack(side=tk.LEFT)
        self._btn_shot.set_enabled(False)
        self._btn_replay.set_enabled(False)

        fps_blk = tk.Frame(right, bg=self.HEADER)
        fps_blk.pack(side=tk.LEFT, padx=(0, S4 * 2))
        tk.Label(fps_blk, text="FPS", font=(FONT, 8, "bold"),
                 bg=self.HEADER, fg=self.FG_DIM).pack()
        self._sv_fps = tk.StringVar(value="—")
        tk.Label(fps_blk, textvariable=self._sv_fps,
                 font=(MONO, 15, "bold"),
                 bg=self.HEADER, fg=self.FG_MID).pack()

        badge_blk = tk.Frame(right, bg=self.HEADER)
        badge_blk.pack(side=tk.LEFT)
        tk.Label(badge_blk, text="CAMERA", font=(FONT, 8, "bold"),
                 bg=self.HEADER, fg=self.FG_DIM).pack()
        self._badge = tk.Label(badge_blk, text="● OFF",
                               font=(FONT, 11, "bold"),
                               bg=self.HEADER, fg=self.RED)
        self._badge.pack()

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self):
        S1, S4 = self.S1, self.S4
        bar = tk.Frame(self, bg=self.HEADER)
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Frame(bar, bg=self.SEP, height=1).pack(side=tk.TOP, fill=tk.X)

        self._sv_status = tk.StringVar(value="Ready")
        tk.Label(bar, textvariable=self._sv_status,
                 font=(FONT, 9, "bold"),
                 bg=self.HEADER, fg=self.FG).pack(
            side=tk.LEFT, padx=(S4, 0), pady=S1
        )

        tk.Label(bar,
                 text="Keys:   S = Start    X = Stop    P = Screenshot    R = Replay    C = Continue live    F9 = Full screen",
                 font=(FONT, 9), bg=self.HEADER,
                 fg=self.FG_DIM).pack(side=tk.LEFT, padx=S4 * 2)

        self._sv_footer = tk.StringVar(
            value=f"{MODEL_LABELS[self._model_key]} · HawkSight v2.0 · CIS 4913"
        )
        tk.Label(bar, textvariable=self._sv_footer,
                 font=(FONT, 9), bg=self.HEADER,
                 fg=self.FG_DIM).pack(side=tk.RIGHT, padx=S4)

    # ── Main area (video + sidebar) ───────────────────────────────────────────

    def _build_main_area(self):
        main = tk.Frame(self, bg=self.BG)
        main.pack(fill=tk.BOTH, expand=True, padx=self.S3, pady=self.S3)

        # Sidebar first (RIGHT), then video fills remaining LEFT space
        self._build_sidebar(main)
        self._build_video_panel(main)

    # ── Video panel ───────────────────────────────────────────────────────────

    def _build_video_panel(self, parent):
        outer = tk.Frame(parent, bg=self.BG)
        outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, self.S3))

        self._border = tk.Frame(outer, bg=self.ORANGE)
        self._border.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(
            self._border, bg="#09090a", highlightthickness=0
        )
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    # Four titled sections. The log is last and takes whatever height is left.

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=self.PANEL, width=self.SIDEBAR_W)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        sb.pack_propagate(False)

        self._build_shot_preview(sb)
        self._build_indicator(sb)
        self._build_controls(self._section(sb, "Camera on / off"))
        self._build_log(sb)

    def _section(self, parent, title: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=self.PANEL)
        frame.pack(fill=tk.X, padx=self.S3, pady=(self.S3, 0))
        tk.Label(frame, text=title.upper(), font=(FONT, 8, "bold"),
                 bg=self.PANEL, fg=self.FG_DIM).pack(anchor=tk.W,
                                                     pady=(0, self.S2))
        return frame

    # ── Detection light ───────────────────────────────────────────────────────
    # One small row: a light that blinks red while a cylinder is spotted,
    # and how sure the detector is about it.

    LIGHT_OFF = "#4a4a4c"
    LIGHT_ON  = "#ff3b30"
    LIGHT_DIM = "#8a1c16"

    def _build_indicator(self, parent):
        S2, S3 = self.S2, self.S3
        box = tk.Frame(parent, bg=self.CARD_BG)
        box.pack(fill=tk.X, padx=S3, pady=(S3, 0))
        row = tk.Frame(box, bg=self.CARD_BG)
        row.pack(fill=tk.X, padx=S3, pady=S2)

        self._light = tk.Canvas(row, width=18, height=18, bg=self.CARD_BG,
                                highlightthickness=0)
        self._light_dot = self._light.create_oval(
            2, 2, 16, 16, fill=self.LIGHT_OFF, outline="")
        self._light.pack(side=tk.LEFT)

        tk.Label(row, text="How sure", font=(FONT, 10, "bold"),
                 bg=self.CARD_BG, fg=self.FG_MID).pack(side=tk.LEFT,
                                                      padx=(S2, 0))
        self._sv_best = tk.StringVar(value="—")
        tk.Label(row, textvariable=self._sv_best, font=(MONO, 14, "bold"),
                 bg=self.CARD_BG, fg=self.FG).pack(side=tk.RIGHT)

    @property
    def light_color(self) -> str:
        return self._light.itemcget(self._light_dot, "fill")

    # ── Controls ──────────────────────────────────────────────────────────────

    def _build_controls(self, parent):
        cfg = dict(relief=tk.FLAT, cursor="hand2", bd=0)

        self._btn_start = tk.Button(
            parent, text="▶   START", font=(FONT, 12, "bold"),
            activebackground=self.GREEN_DK, activeforeground="white",
            command=self._on_start, **cfg,
        )
        self._btn_start.pack(fill=tk.X, ipady=self.S2)

        self._btn_stop = tk.Button(
            parent, text="■   STOP", font=(FONT, 12, "bold"),
            activebackground=self.RED_DK, activeforeground="white",
            command=self._on_stop, state=tk.DISABLED, **cfg,
        )
        self._btn_stop.pack(fill=tk.X, pady=(self.S2, 0), ipady=self.S2)
        self._set_power_look(running=False)

        # Loading indicator: only takes up space while a model is loading.
        self._lbl_loading = tk.Label(parent, text="", wraplength=240,
                                     justify=tk.LEFT,
                                     font=(FONT, 9, "italic"),
                                     bg=self.PANEL, fg=self.ORANGE_LT)

    def _set_power_look(self, running: bool):
        """Camera on: STOP lit red, START dark. Camera off: START lit green,
        STOP dark. Steady colours, no animation."""
        for name, btn in (("start", self._btn_start), ("stop", self._btn_stop)):
            lit = (name == "stop") == running
            bg, fg = (self.LIT[name], "white") if lit else self.DARK[name]
            # Disabled buttons keep these colours instead of greying out.
            btn.config(bg=bg, fg=fg, disabledforeground=fg)

    # ── Settings ──────────────────────────────────────────────────────────────

    def _build_settings(self, bar):
        S1, S2, S4 = self.S1, self.S2, self.S4
        bg = self.HEADER

        def label(text):
            tk.Label(bar, text=text, font=(FONT, 9), bg=bg,
                     fg=self.FG_MID).pack(side=tk.LEFT, padx=(0, S2))

        # Detection model dropdown
        label("What to look for")
        self._model_keys = available_model_keys()
        self._sv_model = tk.StringVar(value=MODEL_LABELS[self._model_key])
        self._cmb_model = ttk.Combobox(
            bar, textvariable=self._sv_model, state="readonly", width=28,
            values=[MODEL_LABELS[k] for k in self._model_keys],
            style="Dark.TCombobox", font=(FONT, 9),
        )
        self._cmb_model.pack(side=tk.LEFT, padx=(0, S4 * 2))
        self._cmb_model.bind("<<ComboboxSelected>>", self._on_model_selected)

        # Camera dropdown + rescan button
        label("Camera")
        self._sv_camera = tk.StringVar(value="Scanning for cameras…")
        self._cmb_camera = ttk.Combobox(
            bar, textvariable=self._sv_camera, state=tk.DISABLED, width=18,
            style="Dark.TCombobox", font=(FONT, 9),
        )
        self._cmb_camera.pack(side=tk.LEFT)
        self._cmb_camera.bind("<<ComboboxSelected>>", self._on_camera_selected)
        self._btn_rescan = tk.Button(
            bar, text="⟳ Find cameras", font=(FONT, 9),
            bg=self.CARD_BG, fg=self.FG,
            disabledforeground=self.FG_DIM,
            activebackground=self.SEP, activeforeground=self.ORANGE_LT,
            relief=tk.FLAT, bd=0, cursor="hand2", padx=S2,
            command=self._scan_cameras,
        )
        self._btn_rescan.pack(side=tk.LEFT, fill=tk.Y, padx=(S1, S4 * 2))

        # Confidence slider
        label("How sure before it alerts")
        self._sv_conf = tk.DoubleVar(value=DEFAULT_CONF)
        ttk.Scale(
            bar, from_=0.05, to=0.95, orient=tk.HORIZONTAL, length=160,
            variable=self._sv_conf, command=self._on_conf_change,
        ).pack(side=tk.LEFT)
        self._lbl_conf = tk.Label(bar, text=f"{DEFAULT_CONF:.0%}",
                                  font=(FONT, 9, "bold"),
                                  bg=bg, fg=self.ORANGE)
        self._lbl_conf.pack(side=tk.LEFT, padx=(S2, 0))

    def _set_loading(self, text: str):
        self._lbl_loading.config(text=text)
        if text:
            self._lbl_loading.pack(anchor=tk.W, pady=(self.S1, 0))
        else:
            self._lbl_loading.pack_forget()

    # ── Last screenshot preview ───────────────────────────────────────────────

    def _build_shot_preview(self, parent):
        self._shot_box = tk.Frame(parent, bg=self.PANEL)
        self._shot_box.pack(side=tk.BOTTOM, fill=tk.X,
                            padx=self.S3, pady=(0, self.S3))
        self._shot_path: Optional[Path] = None
        self._shot_photo: Optional[ImageTk.PhotoImage] = None
        self._shot_widgets_built = False

    def _show_last_shot(self, path: Path, frame: np.ndarray):
        S1, S2 = self.S1, self.S2
        if not self._shot_widgets_built:
            head = tk.Frame(self._shot_box, bg=self.PANEL)
            head.pack(fill=tk.X, pady=(self.S3, S2))
            tk.Label(head, text="LAST SCREENSHOT", font=(FONT, 8, "bold"),
                     bg=self.PANEL, fg=self.FG_DIM).pack(side=tk.LEFT)
            self._lbl_shot_time = tk.Label(head, font=(FONT, 8),
                                           bg=self.PANEL, fg=self.FG_DIM)
            self._lbl_shot_time.pack(side=tk.RIGHT)
            # Thin orange frame around the picture; click to open it.
            ring = tk.Frame(self._shot_box, bg=self.ORANGE)
            ring.pack()
            self._lbl_shot = tk.Label(ring, bg=self.CARD_BG, bd=0,
                                      cursor="hand2")
            self._lbl_shot.pack(padx=2, pady=2)
            self._lbl_shot.bind("<Button-1>", lambda _: self._open_last_shot())
            tk.Label(self._shot_box, text="Click the picture to open it",
                     font=(FONT, 8), bg=self.PANEL,
                     fg=self.FG_DIM).pack(pady=(S1, 0))
            self._shot_widgets_built = True

        width = max(self.SIDEBAR_W - 2 * self.S3 - 4, 80)
        h, w = frame.shape[:2]
        thumb = cv2.resize(frame, (width, max(h * width // w, 1)),
                           interpolation=cv2.INTER_AREA)
        img = Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
        self._shot_photo = ImageTk.PhotoImage(img)   # keep a reference
        self._lbl_shot.config(image=self._shot_photo)
        self._lbl_shot_time.config(text=datetime.now().strftime("%H:%M:%S"))
        self._shot_path = path

    def _open_last_shot(self):
        if self._shot_path and self._shot_path.exists() \
                and hasattr(os, "startfile"):
            os.startfile(self._shot_path)   # opens in the Windows Photos app

    # ── Detection log ─────────────────────────────────────────────────────────

    def _build_log(self, parent):
        S1, S2, S3 = self.S1, self.S2, self.S3

        # The log is hidden until "Show log" is clicked.
        bar = tk.Frame(parent, bg=self.PANEL)
        bar.pack(fill=tk.X, padx=S3, pady=(S3, 0))
        self._log_bar = bar

        # Shown under "Show log" only while a replay is playing.
        self._btn_continue = tk.Button(
            parent, text="▶   CONTINUE LIVE", font=(FONT, 12, "bold"),
            bg=self.GREEN, fg="white",
            activebackground=self.GREEN_DK, activeforeground="white",
            relief=tk.FLAT, bd=0, cursor="hand2",
            command=self._on_continue,
        )
        self._btn_log = tk.Button(
            bar, text="▸  Show log", font=(FONT, 9, "bold"), anchor=tk.W,
            bg=self.PANEL, fg=self.FG_MID,
            activebackground=self.PANEL, activeforeground=self.FG,
            relief=tk.FLAT, bd=0, cursor="hand2", padx=0,
            command=self._toggle_log,
        )
        self._btn_log.pack(side=tk.LEFT)
        self._btn_clear = tk.Button(
            bar, text="Clear", font=(FONT, 8),
            bg=self.PANEL, fg=self.FG_MID,
            activebackground=self.SEP, activeforeground=self.FG,
            relief=tk.FLAT, cursor="hand2", bd=0, padx=S1,
            command=self._clear_log,
        )

        self._log_frame = tk.Frame(parent, bg=self.PANEL)
        # height=4: ask for little, then expand into whatever space is left.
        self._log = tk.Text(
            self._log_frame, bg=self.CARD_BG, fg=self.FG, height=4,
            font=(MONO, 9), relief=tk.FLAT,
            wrap=tk.WORD, state=tk.DISABLED,
            padx=S2, pady=S1, spacing1=1, spacing3=1,
            insertbackground=self.ORANGE,
            selectbackground="#4a3020",
        )
        self._log.tag_configure("detect", foreground=self.ORANGE_LT)
        self._log.tag_configure("clear",  foreground=self.FG_DIM)
        self._log.tag_configure("snap",   foreground="#5dade2")
        self._log.tag_configure("warn",   foreground="#e8705f")
        self._log.tag_configure("div",    foreground=self.FG_DIM)

        sb = ttk.Scrollbar(self._log_frame, orient=tk.VERTICAL,
                           command=self._log.yview)
        self._log.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log.pack(fill=tk.BOTH, expand=True)

    @property
    def log_visible(self) -> bool:
        return self._log_frame.winfo_manager() != ""

    def _toggle_log(self):
        if self.log_visible:
            self._log_frame.pack_forget()
            self._btn_clear.pack_forget()
            self._btn_log.config(text="▸  Show log")
        else:
            self._log_frame.pack(fill=tk.BOTH, expand=True,
                                 padx=self.S3, pady=(self.S2, self.S3))
            self._btn_clear.pack(side=tk.RIGHT)
            self._btn_log.config(text="▾  Hide log")
            self._log.see(tk.END)

    # ── Backend ───────────────────────────────────────────────────────────────

    def _init_backend(self):
        self._model      = DetectionModel(MODEL_PATHS[self._model_key], conf=DEFAULT_CONF)
        self._video      = VideoSource(0)
        self._processor  = FrameProcessor()
        self._controller = SystemController(
            self._video, self._model, self._processor
        )
        self._display = DisplayManager(self._canvas)

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _on_start(self):
        if self._controller.is_running or self._starting:
            return
        if self._scanning:
            self._log_write("Still looking for cameras, try again in a moment.",
                            "div")
            return
        self._starting = True
        self._set_camera_controls()
        self._btn_start.config(state=tk.DISABLED)
        self._cmb_model.config(state=tk.DISABLED)
        if self._model.is_loaded or MODEL_PATHS[self._model_key].exists():
            self._sv_status.set("Getting ready…")
            self._set_loading("Getting ready, this takes a few seconds…")
        else:
            name = MODEL_PATHS[self._model_key].name
            self._sv_status.set("Downloading the detector (first time only)…")
            self._set_loading(
                f"Downloading {name} (first time only, about 6 MB)…"
            )
            self._log_write(f"◆ {name} not found, downloading it once…", "div")

        self._model.conf = self._sv_conf.get()
        self._video      = VideoSource(self._camera)
        self._controller = SystemController(
            self._video, self._model, self._processor
        )
        self._start_result = None
        threading.Thread(target=self._load_and_start, daemon=True).start()
        self.after(100, self._wait_for_start)

    def _load_and_start(self):
        # Runs on a worker thread: never touch tkinter here, only store the
        # result for _wait_for_start to pick up on the main thread.
        path = MODEL_PATHS[self._model_key]
        downloading = not self._model.is_loaded and not path.exists()
        try:
            if not self._model.is_loaded:
                self._model.load()
        except Exception as exc:
            if downloading:
                message = (f"⚠  Could not download {path.name}. Check the "
                           f"internet connection and press START again. ({exc})")
            else:
                message = f"⚠  Could not load model: {exc}"
            self._start_result = (False, "Detector problem", message)
            return
        if self._controller.start():
            self._start_result = (True, "", "")
        else:
            self._start_result = (
                False, "Camera problem",
                f"⚠  Could not open {camera_label(self._camera)}. Close other "
                "apps using it (Teams, Zoom, Camera), check Windows Settings › "
                "Privacy & security › Camera, then press START again.")

    def _wait_for_start(self):
        if self._start_result is None:
            self.after(100, self._wait_for_start)
            return
        self._post_start(*self._start_result)

    def _post_start(self, ok: bool, status: str, message: str):
        self._starting = False
        self._set_camera_controls()
        self._set_loading("")
        if ok:
            self._btn_stop.config(state=tk.NORMAL)
            self._set_power_look(running=True)
            self._sv_status.set("Watching")
            self._badge.config(text="● ON", fg=self.GREEN)
            self._replay.clear()
            self._btn_replay.set_enabled(False)
            self._btn_shot.set_enabled(True)
            self._reset_alert()
            self._last_log_count = -1
            self._last_log_time  = 0.0
            self._log_write(f"── session started · {camera_label(self._camera)} ──",
                            "div")
            self._poll_id  = self.after(30,   self._poll_frames)
            self._badge_id = self.after(900,  self._pulse_live_badge)
        else:
            self._btn_start.config(state=tk.NORMAL)
            self._set_power_look(running=False)
            self._cmb_model.config(state="readonly")
            self._sv_status.set(status)
            self._badge.config(text="● PROBLEM", fg=self.RED)
            self._log_write(message, "warn")
            if status == "Camera problem" and self._none_found:
                # The scan saw no camera and the laptop fallback failed too:
                # Windows can't see any camera, so other apps aren't the cause.
                status = "No camera found"
                self._sv_status.set(status)
                hint = ("Plug in a camera, or turn on the laptop camera "
                        "(camera key or privacy shutter), then press "
                        "⟳ Find cameras.")
            elif status == "Camera problem":
                hint = ("Close other apps that use the camera (Teams, Zoom), "
                        "then press START again.")
            elif "download" in message:
                hint = "Check the internet connection, then press START again."
            else:
                hint = "Try the other choice under “What to look for”."
            self._show_idle(f"⚠  {status}", hint, "#e8705f")

    def _on_stop(self):
        for attr in ("_poll_id", "_alert_id", "_badge_id"):
            after_id = getattr(self, attr)
            if after_id:
                self.after_cancel(after_id)
                setattr(self, attr, None)

        self._controller.stop()
        self._show_idle("Stopped", "Press START to watch again.")
        self._btn_stop.config(state=tk.DISABLED)
        self._set_power_look(running=False)
        self._btn_shot.set_enabled(False)
        self._btn_start.config(state=tk.NORMAL)
        self._cmb_model.config(state="readonly")
        self._sv_status.set("Stopped")
        self._sv_fps.set("—")
        self._sv_best.set("—")
        self._badge.config(text="● OFF", fg=self.RED)
        self._reset_alert()
        self._border.config(bg=self.ORANGE)
        self._log_write("── session ended ──", "div")
        self.title("HawkSight — Gas Cylinder Detection")

    @property
    def replaying(self) -> bool:
        return self._replay_frames is not None

    def _on_replay(self):
        frames = self._replay.frames()
        if not frames:
            return
        if self._replay_id:
            self.after_cancel(self._replay_id)
        self._replay_frames = frames
        self._replay_i = 0
        self._btn_continue.pack(after=self._log_bar, fill=tk.X,
                                padx=self.S3, pady=(self.S3, 0),
                                ipady=self.S2)
        self._log_write(f"⏪ replaying the last {self._replay.duration:.0f} s",
                        "div")
        self._replay_step()

    def _replay_step(self):
        frames = self._replay_frames
        t, frame = frames[self._replay_i]
        length = frames[-1][0]
        self._render(self._replay_overlay(frame, t, length))
        self._replay_i += 1
        if self._replay_i < len(frames):
            # Same gap as when it was recorded (kept between 15 ms and 1 s).
            gap = min(max(frames[self._replay_i][0] - t, 0.015), 1.0)
        else:
            self._replay_i = 0   # loop until "Continue live"
            gap = 1.0
        self._replay_id = self.after(int(gap * 1000), self._replay_step)

    @staticmethod
    def _replay_overlay(frame: np.ndarray, t: float, length: float) -> np.ndarray:
        """Red "REPLAY m:ss / m:ss" tag in the top-left corner."""
        out  = frame.copy()
        tag  = (f"REPLAY  {int(t) // 60}:{int(t) % 60:02d} / "
                f"{int(length) // 60}:{int(length) % 60:02d}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(out.shape[1] / 1280, 0.4) * 0.8
        (tw, th), _ = cv2.getTextSize(tag, font, scale, 2)
        cv2.rectangle(out, (12, 12), (12 + tw + 20, 12 + th + 20),
                      (40, 40, 200), -1)
        cv2.putText(out, tag, (22, 22 + th), font, scale, (255, 255, 255), 2,
                    cv2.LINE_AA)
        return out

    def _on_continue(self):
        if not self.replaying:
            return
        if self._replay_id:
            self.after_cancel(self._replay_id)
            self._replay_id = None
        self._replay_frames = None
        self._btn_continue.pack_forget()
        if self._controller.is_running:
            if self._last_frame is not None:
                self._render(self._last_frame)
        else:
            self._on_canvas_resize()

    def _render(self, frame: np.ndarray):
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        self._display.render(
            self._processor.resize_for_display(frame, w or 800, h or 500))

    def _flash_status(self, text: str, ms: int = 3000):
        """Show `text` in the status bar for a moment, then go back."""
        if self._status_id is None:
            self._status_before = self._sv_status.get()
        else:
            self.after_cancel(self._status_id)
        self._sv_status.set(text)

        def restore():
            self._status_id = None
            if self._sv_status.get() == text:
                self._sv_status.set(self._status_before)
        self._status_id = self.after(ms, restore)

    def _on_snapshot(self):
        if self._last_frame is None:
            return
        SNAP_DIR.mkdir(exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # to the ms
        path = SNAP_DIR / f"snap_{ts}.jpg"
        n = 1
        while path.exists():   # two snapshots in the same millisecond
            path = SNAP_DIR / f"snap_{ts}_{n}.jpg"
            n += 1
        if cv2.imwrite(str(path), self._last_frame):
            self._log_write(f"◎ {path.name}  saved", "snap")
            self._show_last_shot(path, self._last_frame)
            self._flash_status(f"◉  Screenshot saved in the snapshots folder ({path.name})")
        else:
            self._log_write(f"⚠  Could not save {path.name}", "warn")
            self._flash_status("⚠  Could not save the screenshot")

    def _on_model_selected(self, _=None):
        key = next((k for k in self._model_keys
                    if MODEL_LABELS[k] == self._sv_model.get()), None)
        if key is not None:
            self._on_model_switch(key)
        # Show the model actually in use (unchanged if the switch was refused).
        self._sv_model.set(MODEL_LABELS[self._model_key])
        self._cmb_model.selection_clear()

    def _on_model_switch(self, key: str):
        if key == self._model_key or self._controller.is_running:
            return
        path = MODEL_PATHS[key]
        # Stock weights are downloaded on first use; the custom model can't be.
        if not path.exists() and key != "yolo":
            return
        self._model_key = key
        self._model.switch(path)
        self._sv_model.set(MODEL_LABELS[key])
        self._sv_footer.set(f"{MODEL_LABELS[key]} · HawkSight v2.0 · CIS 4913")
        self._log_write(f"◆ model switched → {MODEL_LABELS[key]}", "div")

    # ── Cameras ───────────────────────────────────────────────────────────────

    def _set_camera_controls(self):
        # Locked while scanning or while a feed is starting up; otherwise the
        # camera can be changed, including while the feed is live.
        busy = self._scanning or self._starting
        self._btn_rescan.config(state=tk.DISABLED if busy else tk.NORMAL)
        self._cmb_camera.config(
            state=tk.DISABLED if busy or not self._cameras else "readonly")

    def _scan_cameras(self):
        if self._scanning or self._starting:
            return
        self._scanning = True
        self._set_camera_controls()
        self._sv_camera.set("Scanning for cameras…")
        # The live camera may refuse a second open, so don't probe it.
        live = (self._video.source,) if self._controller.is_running else ()
        found: list = []

        def work():
            try:
                found.append(list_cameras(assume_present=live))
            except Exception:
                found.append([])
        threading.Thread(target=work, daemon=True).start()
        self.after(100, self._wait_for_scan, found)

    def _wait_for_scan(self, found: list):
        if not found:
            self.after(100, self._wait_for_scan, found)
            return
        self._scanning = False
        self._cameras  = found[0]
        self._none_found = not self._cameras
        if self._cameras:
            n = len(self._cameras)
            self._log_write(f"◆ {n} camera{'s' if n != 1 else ''} found", "div")
        else:
            # The scan can miss a camera that will still open (busy, or only
            # reachable through Media Foundation), so offer the laptop camera.
            self._cameras = [LAPTOP_CAMERA]
            self._log_write("◆ No other camera found, using the laptop camera.",
                            "div")
        self._cmb_camera.config(values=[camera_label(i) for i in self._cameras])
        if self._camera not in self._cameras:
            self._camera = self._cameras[0]
        self._sv_camera.set(camera_label(self._camera))
        self._set_camera_controls()

    def _on_camera_selected(self, _=None):
        self._cmb_camera.selection_clear()
        label = self._sv_camera.get()
        index = next((i for i in self._cameras if camera_label(i) == label), None)
        if index is None or index == self._camera:
            self._sv_camera.set(camera_label(self._camera))
            return
        self._camera = index
        self._log_write(f"◆ camera switched → {camera_label(index)}", "div")
        if self._controller.is_running:
            # Release the old camera completely before opening the new one.
            self._on_stop()
            self._on_start()

    def _on_conf_change(self, _=None):
        v = self._sv_conf.get()
        self._lbl_conf.config(text=f"{v:.0%}")
        if self._model.is_loaded:
            self._model.conf = v

    def _on_canvas_resize(self, event=None):
        if self._controller.is_running or self.replaying:
            return
        if self._idle_msg is None:
            self._display.draw_placeholder()
        else:
            self._display.show_message(*self._idle_msg)

    def _show_idle(self, title: str, hint: str = "", color: str = "#a0a0a2"):
        self._idle_msg = (title, hint, color)
        if not self.replaying:   # a replay keeps the screen until "Continue"
            self._display.show_message(*self._idle_msg)

    # ── Frame polling ─────────────────────────────────────────────────────────

    def _poll_frames(self):
        data = self._controller.poll_frame()
        if data is not None:
            frame, result = data
            self._last_frame = frame
            self._replay.add(frame)
            if not self._btn_replay.enabled:
                self._btn_replay.set_enabled(True)
            if not self.replaying:
                self._render(frame)
            self._update_stats(result)
        if self._controller.is_running:
            self._poll_id = self.after(30, self._poll_frames)

    def _update_stats(self, result: DetectionResult):
        conf = result.best_confidence
        self._sv_best.set(f"{conf:.0%}" if conf > 0 else "—")
        fps = self._controller.fps
        self._sv_fps.set(f"{fps:.1f}" if fps > 0 else "—")

        # Window title
        n = result.count
        if n > 0:
            word = "cylinder" if n == 1 else "cylinders"
            self.title(f"HawkSight  —  {n} gas {word} spotted!")
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

    # ── Animations ────────────────────────────────────────────────────────────

    def _start_alert(self):
        self._alert_state = False
        self._alert_id = self.after(0, self._pulse_alert)

    def _reset_alert(self):
        if self._alert_id:
            self.after_cancel(self._alert_id)
            self._alert_id = None
        self._light.itemconfig(self._light_dot, fill=self.LIGHT_OFF)
        self._border.config(bg=self.ORANGE)

    def _pulse_alert(self):
        self._alert_state = not self._alert_state
        on = self._alert_state
        self._light.itemconfig(self._light_dot,
                               fill=self.LIGHT_ON if on else self.LIGHT_DIM)
        self._border.config(bg=self.ORANGE_LT if on else self.ORANGE_DK)
        self._alert_id = self.after(500, self._pulse_alert)

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
        if self._replay_id:
            self.after_cancel(self._replay_id)
        self._controller.stop()
        self.destroy()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = HawkSightApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
