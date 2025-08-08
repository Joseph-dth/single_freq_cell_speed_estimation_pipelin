#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import tkinter as tk
from tkinter import ttk, messagebox

try:
    from PIL import Image, ImageTk
except ImportError:
    raise ImportError("Pillow is required: pip install pillow")

# Optional: try to read a video frame without OpenCV
_HAS_IMAGEIO = True
try:
    import imageio
    import imageio.v2 as iio
except Exception:
    _HAS_IMAGEIO = False

# As a fallback for reading frame only (not GUI)
_HAS_CV2 = True
try:
    import cv2
except Exception:
    _HAS_CV2 = False

LABELS = ["left", "middle", "right"]
COLORS = {
    "left": "#ff0000",    # red
    "middle": "#ffff00",  # yellow
    "right": "#0066ff",   # blue
}
MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 800
HANDLE_SIZE_BASE = 6


def read_settings(path: Path) -> Dict[str, str]:
    settings: Dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                k, v = line.split(':', 1)
                settings[k.strip()] = v.strip()
    return settings


def write_settings(path: Path, settings: Dict[str, str]) -> None:
    lines = []
    for k, v in settings.items():
        lines.append(f"{k}: {v}\n")
    with open(path, 'w') as f:
        f.writelines(lines)


def find_video_for_settings(settings_path: Path) -> Path:
    base_name = settings_path.stem.replace('_setting', '')
    folder = settings_path.parent
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        p = folder / f"{base_name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No video found for base '{base_name}' next to settings")


def read_middle_frame(video_path: Path) -> Tuple[Image.Image, int, int]:
    # Try imageio first (no OpenCV window usage)
    if _HAS_IMAGEIO:
        try:
            rdr = iio.get_reader(str(video_path))
            nframes = rdr.count_frames()
            idx = max(0, nframes // 2)
            frame = rdr.get_data(idx)
            rdr.close()
            # frame is RGB or BGR? imageio typically RGB
            img = Image.fromarray(frame)
            w, h = img.size
            return img, w, h
        except Exception:
            pass
    # Fallback to OpenCV for reading only
    if _HAS_CV2:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Failed to read a representative frame")
        rgb = bgr[:, :, ::-1]
        img = Image.fromarray(rgb)
        w, h = img.size
        return img, w, h
    raise RuntimeError("No backend available to read video frame (install imageio or opencv-python)")


class RoiTkApp:
    def __init__(self, root: tk.Tk, settings_path: Path):
        self.root = root
        self.settings_path = settings_path
        self.settings = read_settings(settings_path)
        self.video_path = find_video_for_settings(settings_path)
        self.src_img, self.W, self.H = read_middle_frame(self.video_path)

        # scale image to fit
        self.scale = min(MAX_DISPLAY_W / self.W, MAX_DISPLAY_H / self.H, 1.0)
        self.disp_w = int(self.W * self.scale)
        self.disp_h = int(self.H * self.scale)
        self.display_img = self.src_img.resize((self.disp_w, self.disp_h), Image.BILINEAR) if self.scale < 1.0 else self.src_img
        self.photo = ImageTk.PhotoImage(self.display_img)

        self.active_label = tk.StringVar(value=LABELS[0])
        self.handle_sz = max(4, int(round(HANDLE_SIZE_BASE * self.scale)))

        # Rectangles in display coordinates: dict[label] = (x1,y1,x2,y2)
        self.rois = self._load_initial_rois()

        # Dragging state
        self.drag_mode: Optional[str] = None  # 'move' or 'resize'
        self.drag_handle: int = -1
        self.start_pt = (0, 0)
        self.start_rect = None

        self._build_ui()
        self._redraw()

    def _load_initial_rois(self) -> Dict[str, Tuple[int, int, int, int]]:
        ok = True
        rois: Dict[str, Tuple[int, int, int, int]] = {}
        for label in LABELS:
            x1 = self.settings.get(f"{label}_roi_x1")
            y1 = self.settings.get(f"{label}_roi_y1")
            x2 = self.settings.get(f"{label}_roi_x2")
            y2 = self.settings.get(f"{label}_roi_y2")
            if x1 and y1 and x2 and y2:
                ox1, oy1, ox2, oy2 = map(lambda v: int(float(v)), (x1, y1, x2, y2))
                dx1, dy1 = int(round(ox1 * self.scale)), int(round(oy1 * self.scale))
                dx2, dy2 = int(round(ox2 * self.scale)), int(round(oy2 * self.scale))
                rois[label] = self._norm_rect((dx1, dy1, dx2, dy2))
            else:
                ok = False
        if not ok:
            third = self.disp_w // 3
            rois = {
                "left": (0, 0, max(third - 1, 0), self.disp_h - 1),
                "middle": (third, 0, max(2 * third - 1, third), self.disp_h - 1),
                "right": (2 * third, 0, self.disp_w - 1, self.disp_h - 1)
            }
        return rois

    def _build_ui(self):
        self.root.title("ROI Setting (Tkinter)")
        self.root.geometry(f"{self.disp_w+260}x{max(self.disp_h, 400)}")

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(main)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, width=self.disp_w, height=self.disp_h, background="#000000")
        self.canvas.pack(fill=tk.BOTH, expand=False)
        self.canvas_img = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        sidebar = ttk.Frame(main, width=240)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(sidebar, text="Active ROI:").pack(anchor=tk.W, padx=10, pady=(12, 4))
        for lbl in LABELS:
            ttk.Radiobutton(sidebar, text=lbl, value=lbl, variable=self.active_label, command=self._redraw).pack(anchor=tk.W, padx=16)

        ttk.Separator(sidebar).pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(sidebar, text="Reset 3 Columns", command=self._reset_thirds).pack(fill=tk.X, padx=10)
        ttk.Button(sidebar, text="Save", command=self._save).pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(sidebar, text="Quit", command=self.root.destroy).pack(fill=tk.X, padx=10)

        info = (
            "使用說明:\n"
            "- 1/2/3 切換 left/middle/right\n"
            "- 拖曳矩形內移動; 拖四角縮放\n"
            "- Save 儲存到設定檔"
        )
        ttk.Label(sidebar, text=info, justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=10)

        # Bind events
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        self.root.bind("1", lambda e: self._set_active("left"))
        self.root.bind("2", lambda e: self._set_active("middle"))
        self.root.bind("3", lambda e: self._set_active("right"))

    def _set_active(self, lbl: str):
        self.active_label.set(lbl)
        self._redraw()

    @staticmethod
    def _norm_rect(rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = rect
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def _hit_test_handle(self, x: int, y: int, rect: Tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = rect
        handles = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        for idx, (cx, cy) in enumerate(handles):
            if abs(x - cx) <= self.handle_sz and abs(y - cy) <= self.handle_sz:
                return idx
        return -1

    def _on_mouse_down(self, event):
        lbl = self.active_label.get()
        rect = self.rois[lbl]
        h = self._hit_test_handle(event.x, event.y, rect)
        if h >= 0:
            self.drag_mode = 'resize'
            self.drag_handle = h
        elif rect[0] <= event.x <= rect[2] and rect[1] <= event.y <= rect[3]:
            self.drag_mode = 'move'
            self.drag_handle = -1
        else:
            self.drag_mode = 'resize'
            self.drag_handle = 3
            self.rois[lbl] = (event.x, event.y, event.x, event.y)
        self.start_pt = (event.x, event.y)
        self.start_rect = self.rois[lbl]
        self._redraw()

    def _on_mouse_move(self, event):
        if not self.drag_mode:
            return
        lbl = self.active_label.get()
        x1, y1, x2, y2 = self.start_rect
        dx = event.x - self.start_pt[0]
        dy = event.y - self.start_pt[1]
        if self.drag_mode == 'move':
            nx1, ny1, nx2, ny2 = x1 + dx, y1 + dy, x2 + dx, y2 + dy
            nx1, nx2 = max(0, min(nx1, nx2)), min(self.disp_w - 1, max(nx1, nx2))
            ny1, ny2 = max(0, min(ny1, ny2)), min(self.disp_h - 1, max(ny1, ny2))
            self.rois[lbl] = (nx1, ny1, nx2, ny2)
        else:
            hx1, hy1, hx2, hy2 = x1, y1, x2, y2
            if self.drag_handle == 0:   # TL
                hx1, hy1 = x1 + dx, y1 + dy
            elif self.drag_handle == 1: # TR
                hx2, hy1 = x2 + dx, y1 + dy
            elif self.drag_handle == 2: # BL
                hx1, hy2 = x1 + dx, y2 + dy
            elif self.drag_handle == 3: # BR
                hx2, hy2 = x2 + dx, y2 + dy
            hx1 = max(0, min(hx1, self.disp_w - 1)); hy1 = max(0, min(hy1, self.disp_h - 1))
            hx2 = max(0, min(hx2, self.disp_w - 1)); hy2 = max(0, min(hy2, self.disp_h - 1))
            self.rois[lbl] = self._norm_rect((hx1, hy1, hx2, hy2))
        self._redraw()

    def _on_mouse_up(self, event):
        self.drag_mode = None
        self.drag_handle = -1

    def _reset_thirds(self):
        third = self.disp_w // 3
        self.rois = {
            "left": (0, 0, max(third - 1, 0), self.disp_h - 1),
            "middle": (third, 0, max(2 * third - 1, third), self.disp_h - 1),
            "right": (2 * third, 0, self.disp_w - 1, self.disp_h - 1)
        }
        self._redraw()

    def _save(self):
        # convert to original resolution
        for lbl in LABELS:
            dx1, dy1, dx2, dy2 = self.rois[lbl]
            ox1 = int(round(dx1 / self.scale))
            oy1 = int(round(dy1 / self.scale))
            ox2 = int(round(dx2 / self.scale))
            oy2 = int(round(dy2 / self.scale))
            ox1 = max(0, min(ox1, self.W - 1)); ox2 = max(0, min(ox2, self.W - 1))
            oy1 = max(0, min(oy1, self.H - 1)); oy2 = max(0, min(oy2, self.H - 1))
            x1, y1, x2, y2 = self._norm_rect((ox1, oy1, ox2, oy2))
            self.settings[f"{lbl}_roi_x1"] = str(x1)
            self.settings[f"{lbl}_roi_y1"] = str(y1)
            self.settings[f"{lbl}_roi_x2"] = str(x2)
            self.settings[f"{lbl}_roi_y2"] = str(y2)
        self.settings["roi_crop_detection"] = self.settings.get("roi_crop_detection", "1")
        write_settings(self.settings_path, self.settings)
        messagebox.showinfo("Saved", f"Saved ROIs to {self.settings_path}")

    def _redraw(self):
        self.canvas.delete("roi")
        # background image already on canvas
        for lbl in LABELS:
            x1, y1, x2, y2 = self.rois[lbl]
            color = COLORS[lbl]
            width = 3 if lbl == self.active_label.get() else 2
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, tags="roi")
            # handles
            for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                self.canvas.create_rectangle(cx - self.handle_sz, cy - self.handle_sz,
                                             cx + self.handle_sz, cy + self.handle_sz,
                                             outline=color, fill=color, tags="roi")
            # label text
            self.canvas.create_text(x1 + 6, max(14, y1 + 12), text=lbl, anchor=tk.W, fill=color, font=("Arial", 12, "bold"), tags="roi")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Draw three ROIs and save to settings file (Tkinter GUI)")
    ap.add_argument("settings_file", help="Path to settings file")
    args = ap.parse_args()

    settings_path = Path(args.settings_file)
    root = tk.Tk()
    app = RoiTkApp(root, settings_path)
    root.mainloop()


if __name__ == "__main__":
    main() 