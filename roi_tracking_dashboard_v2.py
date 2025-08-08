#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox

# For popup dialogs to input precise times
try:
    import tkinter as tk
    from tkinter import simpledialog
    _HAS_TK = True
except Exception:
    _HAS_TK = False

# --- Constants ---
LABELS = ["left", "middle", "right"]

# --- Settings Loader (to get scale if available) ---

def read_settings(settings_path: Path) -> Dict[str, str]:
    settings: Dict[str, str] = {}
    if not settings_path.exists():
        return settings
    with open(settings_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                k, v = line.split(':', 1)
                settings[k.strip()] = v.strip()
    return settings

# --- Data IO ---

def load_csvs(settings_path: Path):
    base_name = settings_path.stem.replace('_setting', '')
    folder = settings_path.parent
    raw_csv = folder / f"{base_name}_raw.csv"
    kf_csv = folder / f"{base_name}_kf.csv"

    if not raw_csv.exists():
        raise FileNotFoundError(f"RAW CSV not found: {raw_csv}")
    df_raw = pd.read_csv(raw_csv)
    df_kf = pd.read_csv(kf_csv) if kf_csv.exists() else None
    return df_raw, df_kf, base_name, folder

# --- Helpers ---

def group_by_roi_and_id(df: pd.DataFrame) -> Dict[str, Dict[int, pd.DataFrame]]:
    groups: Dict[str, Dict[int, pd.DataFrame]] = {l: {} for l in LABELS}
    for (roi, cell_id), sub in df.groupby(['roi', 'cell_id']):
        roi = str(roi)
        if roi not in groups:
            continue
        groups[roi][int(cell_id)] = sub.sort_values(['timestamp'])
    return groups

def apply_kalman_1d(traj: pd.DataFrame, q_scale: float, r_scale: float) -> pd.DataFrame:
    # 1D constant-velocity KF on x only; returns traj with 'cell_x'
    xs = traj['cell_x'].values.astype(float)
    ts = traj['timestamp'].values.astype(float)
    if len(xs) == 0:
        return traj.copy()

    x = np.array([[xs[0]], [0.0]])
    P = np.eye(2) * 100.0
    out = []
    for i in range(len(xs)):
        dt = ts[i] - ts[i-1] if i > 0 else 0.033
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = max(1e-9, q_scale) * np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[max(1e-6, r_scale)]])
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        # update
        z = np.array([[xs[i]]])
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P
        out.append(float(x[0, 0]))
    df = traj.copy()
    df['cell_x'] = np.array(out)
    return df

def recompute_kf_from_raw(df_raw: pd.DataFrame, q_scale: float, r_scale: float) -> pd.DataFrame:
    parts = []
    for (roi, cell_id), sub in df_raw.groupby(['roi', 'cell_id']):
        smoothed = apply_kalman_1d(sub.sort_values('timestamp'), q_scale, r_scale)
        smoothed['roi'] = roi
        smoothed['cell_id'] = cell_id
        parts.append(smoothed[['roi', 'frame_number', 'timestamp', 'cell_id', 'cell_x', 'cell_y']])
    out = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame(columns=['roi','frame_number','timestamp','cell_id','cell_x','cell_y'])
    return out

def sample_x_at_time(traj_sorted: pd.DataFrame, t: float, clamp: bool = False) -> float:
    # Linear interpolate x at timestamp t.
    # If clamp=False and t outside [t_min, t_max], return NaN to indicate no coverage.
    times = traj_sorted['timestamp'].values.astype(float)
    xs = traj_sorted['cell_x'].values.astype(float)
    if len(times) == 0:
        return np.nan
    if t <= times[0]:
        return xs[0] if clamp else np.nan
    if t >= times[-1]:
        return xs[-1] if clamp else np.nan
    idx = np.searchsorted(times, t)
    if idx <= 0 or idx >= len(times):
        return np.nan
    t0, t1 = times[idx-1], times[idx]
    x0, x1 = xs[idx-1], xs[idx]
    if t1 == t0:
        return x0
    alpha = (t - t0) / (t1 - t0)
    return (1 - alpha) * x0 + alpha * x1

def first_crossing_time(traj_sorted: pd.DataFrame,
                        x_thr: float,
                        t1: float,
                        t2: float) -> float:
    """
    在 [t1, t2] 內找 'x <= x_thr' 的第一次時間點。
    使用線性內插；若未穿越回 np.nan。
    """
    times = traj_sorted['timestamp'].values.astype(float)
    xs = traj_sorted['cell_x'].values.astype(float)
    n = len(times)
    if n == 0:
        return np.nan
    if t2 < times[0] or t1 > times[-1]:
        return np.nan

    i_start = max(0, np.searchsorted(times, t1) - 1)
    i_end   = min(n - 1, np.searchsorted(times, t2))

    # 先掃單點
    for i in range(i_start, min(i_end + 1, n)):
        if times[i] < t1: 
            continue
        if times[i] > t2: 
            break
        if xs[i] <= x_thr:
            return times[i]

    # 再掃線段
    for i in range(i_start, i_end):
        t0, t1s = times[i], times[i+1]
        if t1s < t1 or t0 > t2:
            continue
        x0, x1v = xs[i], xs[i+1]
        crossed = (x0 > x_thr) and (x1v <= x_thr)
        if not crossed and not (x0 == x_thr or x1v == x_thr):
            continue
        if x0 == x_thr and (t0 >= t1 and t0 <= t2):
            return t0
        if x1v == x_thr and (t1s >= t1 and t1s <= t2):
            return t1s
        if x1v != x0:
            alpha = (x_thr - x0) / (x1v - x0)
            if 0.0 <= alpha <= 1.0:
                t_cross = t0 + alpha * (t1s - t0)
                if t_cross >= t1 and t_cross <= t2:
                    return float(t_cross)
    return np.nan

# --- Dashboard ---

def dashboard(settings_file: str):
    settings_path = Path(settings_file)
    settings = read_settings(settings_path)
    scale_microns_per_pixel: Optional[float] = None
    if 'scale_microns_per_pixel' in settings:
        try:
            scale_microns_per_pixel = float(settings['scale_microns_per_pixel'])
        except Exception:
            scale_microns_per_pixel = None

    df_raw, df_kf_loaded, base_name, folder = load_csvs(settings_path)

    # State
    use_kf = False
    q_init, r_init = 1.0, 1.0
    df_kf = df_kf_loaded if df_kf_loaded is not None else recompute_kf_from_raw(df_raw, q_init, r_init)
    
    # Track deleted lines: set of (roi, cell_id) tuples
    deleted_lines = set()

    def current_df() -> pd.DataFrame:
        return df_kf if use_kf else df_raw

    # Initial groups
    groups = group_by_roi_and_id(current_df())

    # Figure with 1 row x 3 cols
    fig, axes = plt.subplots(1, 3, figsize=(17, 8), sharex=False, sharey=False)
    fig.suptitle(f"ROI Tracking - {base_name}")
    if len(axes.shape) == 1:
        axes = axes.reshape(1, -1)

    # Plotter for trajectories
    def plot_groups(ax, groups_roi: Dict[int, pd.DataFrame], roi_name: str = ""):
        ax.clear()
        lines = []
        for cell_id, traj in groups_roi.items():
            if (roi_name, cell_id) in deleted_lines:
                continue
            ts = traj['timestamp'].values
            xs = traj['cell_x'].values
            line = ax.plot(ts, xs, linewidth=1, alpha=0.85, picker=True, pickradius=15)[0]
            line.roi_name = roi_name
            line.cell_id = cell_id
            lines.append(line)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('x (px)')
        return lines

    # Initial draw for data row
    for j, label in enumerate(LABELS):
        axes[0, j].set_title(("Data-" + label) + (" (KF)" if use_kf else " (RAW)"))
        plot_groups(axes[0, j], groups.get(label, {}), label)

    # Two vertical time lines across ROIs
    ts_min = float(current_df()['timestamp'].min()) if len(current_df()) else 0.0
    t_start = ts_min
    t_end = ts_min + 1.0
    vlines_tstart = [axes[0, j].axvline(t_start, color='cyan', linestyle='-', linewidth=1.8, alpha=0.9) for j in range(3)]
    vlines_tend   = [axes[0, j].axvline(t_end,   color='magenta', linestyle='-', linewidth=1.8, alpha=0.9) for j in range(3)]

    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.9, wspace=0.3)

    # UI controls
    ax_chk = plt.axes([0.07, 0.15, 0.12, 0.04])
    chk = CheckButtons(ax_chk, ["Use KF"], [use_kf])

    ax_q = plt.axes([0.22, 0.16, 0.25, 0.03])
    ax_r = plt.axes([0.22, 0.12, 0.25, 0.03])
    s_q = Slider(ax_q, 'KF Q', 0.001, 10.0, valinit=q_init, valstep=0.001)
    s_r = Slider(ax_r, 'KF R', 0.001, 10.0, valinit=r_init, valstep=0.001)

    ax_apply = plt.axes([0.49, 0.135, 0.12, 0.04])
    btn_apply = Button(ax_apply, 'Apply KF')

    # Period/Repeats inputs
    ax_period = plt.axes([0.66, 0.16, 0.09, 0.03])
    ax_repeats = plt.axes([0.76, 0.16, 0.07, 0.03])
    tb_period = TextBox(ax_period, 'Period(s)', initial="12.0")
    tb_repeats = TextBox(ax_repeats, 'Repeats', initial="8")

    # Min/max speed filter (px/s)
    ax_minspd = plt.axes([0.66, 0.12, 0.09, 0.03])
    ax_maxspd = plt.axes([0.76, 0.12, 0.09, 0.03])
    tb_minspd = TextBox(ax_minspd, 'Min v', initial="")
    tb_maxspd = TextBox(ax_maxspd, 'Max v', initial="")

    # Start/End time TextBoxes
    ax_tstart = plt.axes([0.22, 0.08, 0.08, 0.03])
    ax_tend   = plt.axes([0.31, 0.08, 0.08, 0.03])
    tb_tstart = TextBox(ax_tstart, 'Start(s)', initial=f"{t_start:.3f}")
    tb_tend   = TextBox(ax_tend,   'End(s)',   initial=f"{t_end:.3f}")

    # Bounds for initial scaling
    df_for_scale = current_df()
    if len(df_for_scale) > 0:
        t_data_min, t_data_max = df_for_scale['timestamp'].min(), df_for_scale['timestamp'].max()
        x_data_min, x_data_max = df_for_scale['cell_x'].min(), df_for_scale['cell_x'].max()
        t_range = t_data_max - t_data_min
        x_range = x_data_max - x_data_min
    else:
        t_data_min, t_data_max = 0, 100
        x_data_min, x_data_max = 0, 500
        t_range, x_range = 100, 500

    # --- Per-ROI thresholds (x1/x2) and global delta_x ---
    # Initialize per-ROI x1 as 25% quantile; delta_x as 25% of global range
    x_thresholds: Dict[str, Dict[str, float]] = {}
    hlines_x1: Dict[str, any] = {}
    hlines_x2: Dict[str, any] = {}

    def init_roi_thresholds():
        nonlocal x_thresholds, hlines_x1, hlines_x2, delta_x
        # initial delta_x（全域同距離）
        delta_x = max(1e-6, 0.25 * (x_data_max - x_data_min))

        x_thresholds = {}
        hlines_x1, hlines_x2 = {}, {}

        for roi, j in zip(LABELS, range(3)):
            roi_data = df_for_scale[df_for_scale['roi'] == roi] if len(df_for_scale) else pd.DataFrame()

            if len(roi_data) > 0:
                x_vals = roi_data['cell_x'].values.astype(float)
                roi_min, roi_max = float(np.min(x_vals)), float(np.max(x_vals))
                # x1 一開始放「較大」位置（上面）：用 75 分位數比較穩定
                x1_init = float(np.quantile(x_vals, 0.75))
            else:
                roi_min, roi_max = x_data_min, x_data_max
                x1_init = x_data_min + 0.75 * (x_data_max - x_data_min)

            # x2 比 x1 小（在下方），相距 delta_x
            x2_init = x1_init - delta_x

            # --- 邊界保護：確保 x1 > x2 且都在 [roi_min, roi_max] 內 ---
            # 若 x2 太低，先把 x2 拉回來，再用同距離推回 x1；若又超上界，再夾回可視範圍
            if x2_init < roi_min:
                # 讓 x2 稍離邊界，避免貼齊不好拖
                pad = 0.02 * (roi_max - roi_min) if roi_max > roi_min else 0.0
                x2_init = roi_min + pad
                x1_init = x2_init + delta_x
                if x1_init > roi_max:  # delta 太大裝不下 → 往下調 x1 並維持順序
                    x1_init = roi_max - pad
                    x2_init = x1_init - delta_x

            # 最後保證順序
            if not (x1_init > x2_init):
                x1_init = min(roi_max, x2_init + abs(delta_x))
                x2_init = max(roi_min, x1_init - abs(delta_x))

            x_thresholds[roi] = {'x1': x1_init, 'x2': x2_init}
            hlines_x1[roi] = axes[0, j].axhline(x1_init, color='green', linestyle='-',  linewidth=1.8, alpha=0.9)   # x1 上面（大）
            hlines_x2[roi] = axes[0, j].axhline(x2_init, color='orange', linestyle='--', linewidth=1.8, alpha=0.9)   # x2 下面（小）


    delta_x = 1.0  # will be overwritten by init
    init_roi_thresholds()

    # Zoom & pan controls
    ax_x_scale = plt.axes([0.07, 0.05, 0.12, 0.02])
    s_x_scale = Slider(ax_x_scale, 'X Zoom', 0.1, 10.0, valinit=1.0, valstep=0.1)
    ax_y_scale = plt.axes([0.07, 0.02, 0.12, 0.02])
    s_y_scale = Slider(ax_y_scale, 'Y Zoom', 0.1, 10.0, valinit=1.0, valstep=0.1)

    ax_x_left = plt.axes([0.21, 0.05, 0.025, 0.02])
    ax_x_right = plt.axes([0.24, 0.05, 0.025, 0.02])
    ax_y_up = plt.axes([0.27, 0.05, 0.025, 0.02])
    ax_y_down = plt.axes([0.30, 0.05, 0.025, 0.02])
    btn_x_left = Button(ax_x_left, '←')
    btn_x_right = Button(ax_x_right, '→')
    btn_y_up = Button(ax_y_up, '↑')
    btn_y_down = Button(ax_y_down, '↓')

    ax_select = plt.axes([0.34, 0.05, 0.06, 0.02])
    btn_select = Button(ax_select, 'Select')
    ax_delete = plt.axes([0.34, 0.02, 0.06, 0.02])
    btn_delete = Button(ax_delete, 'Delete')

    ax_autoscale = plt.axes([0.41, 0.05, 0.06, 0.02])
    btn_autoscale = Button(ax_autoscale, 'Auto Scale')

    ax_compute = plt.axes([0.42, 0.08, 0.11, 0.03])
    btn_compute = Button(ax_compute, 'Compute')

    ax_apply_filter = plt.axes([0.42, 0.04, 0.11, 0.03])
    btn_apply_filter = Button(ax_apply_filter, 'Apply Filter')

    ax_save = plt.axes([0.54, 0.04, 0.09, 0.03])
    btn_save = Button(ax_save, 'Save')

    # KF toggle
    def redraw_all_axes():
        """Redraw trajectories, vlines, and per-ROI x1/x2 lines."""
        for j, roi in enumerate(LABELS):
            axes[0, j].clear()
            plot_groups(axes[0, j], groups.get(roi, {}), roi)
            axes[0, j].set_title(("Data-" + roi) + (" (KF)" if use_kf else " (RAW)"))
            # re-add vlines
            vlines_tstart[j] = axes[0, j].axvline(t_start, color='cyan', linestyle='-', linewidth=1.8, alpha=0.9)
            vlines_tend[j]   = axes[0, j].axvline(t_end,   color='magenta', linestyle='-', linewidth=1.8, alpha=0.9)
            # re-add hlines for this roi
            hlines_x1[roi] = axes[0, j].axhline(x_thresholds[roi]['x1'], color='green', linestyle='-', linewidth=1.8, alpha=0.9)
            hlines_x2[roi] = axes[0, j].axhline(x_thresholds[roi]['x2'], color='orange', linestyle='--', linewidth=1.8, alpha=0.9)
        fig.canvas.draw_idle()

    def on_toggle_kf(label):
        nonlocal use_kf, groups
        use_kf = not use_kf
        groups = group_by_roi_and_id(current_df())
        redraw_all_axes()
    chk.on_clicked(on_toggle_kf)

    def on_apply_kf(event):
        nonlocal df_kf, groups
        qv = float(s_q.val)
        rv = float(s_r.val)
        df_kf = recompute_kf_from_raw(df_raw, qv, rv)
        if use_kf:
            groups = group_by_roi_and_id(df_kf)
            redraw_all_axes()
    btn_apply.on_clicked(on_apply_kf)

    # Dragging states
    drag_state = {"kind": None, "which": None}  # vertical t lines
    drag_state_h = {"active": False, "which": None, "roi": None}  # 'x1' or 'x2' for per-ROI

    # Modes
    select_mode = False
    delete_mode = False

    # Rectangle selection state for delete/select
    rect_selection = {
        'active': False,
        'start_x': None,
        'start_y': None,
        'current_x': None,
        'current_y': None,
        'plot_index': None,
        'rect_patch': None
    }

    def on_press(event):
        if event.inaxes is None:
            return

        # select/delete rectangle begin
        if select_mode or delete_mode:
            plot_index = None
            for j in range(3):
                if event.inaxes == axes[0, j]:
                    plot_index = j
                    break
            if plot_index is not None and event.xdata is not None and event.ydata is not None:
                rect_selection.update({
                    'active': True,
                    'start_x': event.xdata,
                    'start_y': event.ydata,
                    'current_x': event.xdata,
                    'current_y': event.ydata,
                    'plot_index': plot_index,
                    'mode': 'select' if select_mode else 'delete'
                })
                return

        # Horizontal line dragging (x1/x2) per-ROI
        if not select_mode and not delete_mode:
            for roi, j in zip(LABELS, range(3)):
                if event.inaxes == axes[0, j]:
                    y = event.ydata
                    if y is None:
                        break
                    x1_val = x_thresholds[roi]['x1']
                    x2_val = x_thresholds[roi]['x2']
                    # sensitivity: in data units; adjust if needed
                    if abs(y - x1_val) < 2.0:
                        drag_state_h.update({"active": True, "which": 'x1', "roi": roi})
                        return
                    if abs(y - x2_val) < 2.0:
                        drag_state_h.update({"active": True, "which": 'x2', "roi": roi})
                        return

        # Vertical time lines (only when not in select/delete mode)
        if not select_mode and not delete_mode:
            for j in range(3):
                if event.inaxes == axes[0, j]:
                    x = event.xdata
                    if x is None:
                        return
                    if abs(x - t_start) < 0.2:
                        drag_state.update({"kind": 'v', "which": 't_start'})
                    elif abs(x - t_end) < 0.2:
                        drag_state.update({"kind": 'v', "which": 't_end'})
                    return

    def on_motion(event):
        nonlocal t_start, t_end, delta_x

        # rectangle selection drawing
        if (select_mode or delete_mode) and rect_selection['active'] and event.inaxes is not None:
            if event.xdata is not None and event.ydata is not None:
                rect_selection['current_x'] = event.xdata
                rect_selection['current_y'] = event.ydata
                if rect_selection['rect_patch'] is not None:
                    rect_selection['rect_patch'].remove()
                plot_index = rect_selection['plot_index']
                ax = axes[0, plot_index]
                x_min = min(rect_selection['start_x'], rect_selection['current_x'])
                x_max = max(rect_selection['start_x'], rect_selection['current_x'])
                y_min = min(rect_selection['start_y'], rect_selection['current_y'])
                y_max = max(rect_selection['start_y'], rect_selection['current_y'])
                width = x_max - x_min
                height = y_max - y_min
                from matplotlib.patches import Rectangle
                rect_patch = Rectangle((x_min, y_min), width, height,
                                       linewidth=2,
                                       edgecolor=('blue' if select_mode else 'red'),
                                       facecolor=('blue' if select_mode else 'red'),
                                       alpha=0.3)
                ax.add_patch(rect_patch)
                rect_selection['rect_patch'] = rect_patch
                fig.canvas.draw_idle()
            return

        # Horizontal line dragging
        if drag_state_h["active"] and event.inaxes is not None:
            roi = drag_state_h["roi"]
            j = LABELS.index(roi)
            y = event.ydata
            if y is None:
                return
            if drag_state_h["which"] == 'x1':
                # move only this ROI's x1 and keep its delta
                delta_local = x_thresholds[roi]['x2'] - x_thresholds[roi]['x1']
                x_thresholds[roi]['x1'] = y
                x_thresholds[roi]['x2'] = y + delta_local
                hlines_x1[roi].set_ydata([y, y])
                hlines_x2[roi].set_ydata([y + delta_local, y + delta_local])
            elif drag_state_h["which"] == 'x2':
                # update global delta_x and propagate x2 to all ROI as x1 + delta_x
                new_delta = y - x_thresholds[roi]['x1']
                delta_x = float(new_delta)
                for r, jj in zip(LABELS, range(3)):
                    x_thresholds[r]['x2'] = x_thresholds[r]['x1'] + delta_x
                    hlines_x2[r].set_ydata([x_thresholds[r]['x2'], x_thresholds[r]['x2']])
            fig.canvas.draw_idle()
            return

        # Vertical time line dragging
        if drag_state["kind"] != 'v' or event.inaxes is None:
            return
        x = event.xdata
        if x is None:
            return
        if drag_state["which"] == 't_start':
            t_start = float(x)
            for j in range(3):
                vlines_tstart[j].set_xdata([t_start, t_start])
            tb_tstart.set_val(f"{t_start:.3f}")
        else:
            t_end = float(x)
            for j in range(3):
                vlines_tend[j].set_xdata([t_end, t_end])
            tb_tend.set_val(f"{t_end:.3f}")
        fig.canvas.draw_idle()

    def on_release(event):
        # finish rectangle selection
        if (select_mode or delete_mode) and rect_selection['active']:
            x_min = min(rect_selection['start_x'], rect_selection['current_x'])
            x_max = max(rect_selection['start_x'], rect_selection['current_x'])
            y_min = min(rect_selection['start_y'], rect_selection['current_y'])
            y_max = max(rect_selection['start_y'], rect_selection['current_y'])
            if (x_max - x_min) > 0.01 and (y_max - y_min) > 0.01:
                selected_plot_index = rect_selection['plot_index']
                if select_mode:
                    rect_width = x_max - x_min
                    rect_height = y_max - y_min
                    rect_center_x = (x_min + x_max) / 2
                    rect_center_y = (y_min + y_max) / 2
                    current_centers = []
                    for j in range(3):
                        xlim = axes[0, j].get_xlim()
                        ylim = axes[0, j].get_ylim()
                        center_x = (xlim[0] + xlim[1]) / 2
                        center_y = (ylim[0] + ylim[1]) / 2
                        current_centers.append((center_x, center_y))
                    selected_center_x, selected_center_y = current_centers[selected_plot_index]
                    offset_x = rect_center_x - selected_center_x
                    offset_y = rect_center_y - selected_center_y
                    for j in range(3):
                        new_center_x = current_centers[j][0] + offset_x
                        new_center_y = current_centers[j][1] + offset_y
                        axes[0, j].set_xlim(new_center_x - rect_width/2, new_center_x + rect_width/2)
                        axes[0, j].set_ylim(new_center_y - rect_height/2, new_center_y + rect_height/2)
                    print(f"Zoomed to rectangle: width={rect_width:.2f}, height={rect_height:.2f}")
                elif delete_mode:
                    roi_name = LABELS[selected_plot_index]
                    lines_to_delete = []
                    groups_local = group_by_roi_and_id(current_df())
                    for cell_id, traj in groups_local.get(roi_name, {}).items():
                        if (roi_name, cell_id) in deleted_lines:
                            continue
                        ts = traj['timestamp'].values
                        xs = traj['cell_x'].values
                        within_bounds = ((ts >= x_min) & (ts <= x_max) &
                                         (xs >= y_min) & (xs <= y_max))
                        if np.any(within_bounds):
                            lines_to_delete.append((roi_name, cell_id))
                    for roi, cid in lines_to_delete:
                        deleted_lines.add((roi, cid))
                    if lines_to_delete:
                        print(f"Deleted {len(lines_to_delete)} lines in rectangle from {roi_name}")
                        redraw_all_axes()
            if rect_selection['rect_patch'] is not None:
                rect_selection['rect_patch'].remove()
            rect_selection.update({
                'active': False,
                'start_x': None,
                'start_y': None,
                'current_x': None,
                'current_y': None,
                'plot_index': None,
                'rect_patch': None,
                'mode': None
            })
            fig.canvas.draw_idle()
            return

        drag_state.update({"kind": None, "which": None})
        drag_state_h.update({"active": False, "which": None, "roi": None})

    def on_pick(event):
        pass

    def on_key_press(event):
        nonlocal select_mode, delete_mode, drag_state
        if event.key == 'escape':
            if drag_state["kind"] is not None:
                drag_state.update({"kind": None, "which": None})
                print("Cancelled time line dragging")
            if rect_selection['active']:
                if rect_selection['rect_patch'] is not None:
                    rect_selection['rect_patch'].remove()
                rect_selection.update({
                    'active': False,
                    'start_x': None,
                    'start_y': None,
                    'current_x': None,
                    'current_y': None,
                    'plot_index': None,
                    'rect_patch': None,
                    'mode': None
                })
                print("Cancelled rectangle selection")
                fig.canvas.draw_idle()
            if select_mode:
                select_mode = False
                btn_select.label.set_text('Select')
                btn_select.color = 'lightgray'
                btn_select.hovercolor = 'gray'
                print("Exited select mode")
                fig.canvas.draw_idle()
            if delete_mode:
                delete_mode = False
                btn_delete.label.set_text('Delete')
                btn_delete.color = 'lightgray'
                btn_delete.hovercolor = 'gray'
                print("Exited delete mode")
                fig.canvas.draw_idle()

    cid_pick = fig.canvas.mpl_connect('pick_event', on_pick)
    cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
    cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
    cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key_press)

    # Build cycles helper
    def build_cycles() -> List[Tuple[float, float]]:
        try:
            period_s = float(tb_period.text)
        except Exception:
            period_s = 12.0
        try:
            repeats = int(float(tb_repeats.text))
        except Exception:
            repeats = 8
        return [(t_start + k * period_s, t_end + k * period_s) for k in range(repeats)]

    # Compute speeds with per-ROI thresholds and crossing times
    def compute_speeds(event=None):
        cycles = build_cycles()
        df = current_df()
        groups_local = group_by_roi_and_id(df)
        speed_rows = []
        for roi in LABELS:
            for _id, traj in groups_local.get(roi, {}).items():
                if (roi, _id) in deleted_lines:
                    continue
                traj_sorted = traj.sort_values('timestamp')
                x1 = x_thresholds[roi]['x1']
                x2 = x_thresholds[roi]['x2']
                for cycle_idx, (ts, te) in enumerate(cycles):
                    t_x1 = first_crossing_time(traj_sorted, x1, ts, te)
                    t_x2 = first_crossing_time(traj_sorted, x2, ts, te)
                    if np.isfinite(t_x1) and np.isfinite(t_x2) and (t_x2 > t_x1):
                        move_s = t_x2 - t_x1
                        v_px_s = (x2 - x1) / move_s if move_s > 0 else np.nan
                        v_um_s = v_px_s * scale_microns_per_pixel if (scale_microns_per_pixel is not None and np.isfinite(v_px_s)) else np.nan
                    else:
                        move_s = np.nan
                        v_px_s = np.nan
                        v_um_s = np.nan
                    speed_rows.append({
                        'roi': roi, 'cycle_index': cycle_idx, 'cell_id': int(_id),
                        't_start': ts, 't_end': te,
                        'x1': x1, 'x2': x2,
                        't_x1': t_x1, 't_x2': t_x2,
                        'move_s': move_s,
                        'speed_px_per_s': abs(v_px_s) if np.isfinite(v_px_s) else np.nan,
                        'speed_um_per_s': abs(v_um_s) if np.isfinite(v_um_s) else np.nan
                    })
        fig._speeds_df = pd.DataFrame(speed_rows)
        print(f"Computed speeds with per-ROI x1/x2 thresholds: {len(speed_rows)} rows")
    btn_compute.on_clicked(compute_speeds)

    # Apply filter by min/max speed (absolute) and refresh top plots
    def get_speed_bounds() -> Tuple[float, float]:
        def _parse(tb, default):
            try:
                return float(tb.text)
            except Exception:
                return default
        vmin = _parse(tb_minspd, -np.inf)
        vmax = _parse(tb_maxspd, np.inf)
        return vmin, vmax

    def apply_filter(event=None):
        # Clear deleted_lines when applying/reapplying filter? —— 保留現有刪除結果，不重置。
        speeds_df = getattr(fig, '_speeds_df', None)
        if speeds_df is None or len(speeds_df) == 0:
            print("No speeds computed yet.")
            return
        vmin, vmax = get_speed_bounds()
        abs_speeds = np.abs(speeds_df['speed_px_per_s'])
        passed = speeds_df[np.isfinite(speeds_df['speed_px_per_s']) & (abs_speeds >= vmin) & (abs_speeds <= vmax)]
        # 只重畫頂部圖（保留當前刪除線），視覺上仍顯示全部，但匯出與統計會用篩選集
        redraw_all_axes()
        fig.canvas.draw_idle()
        fig._speeds_filtered_df = passed.reset_index(drop=True)
        print(f"Filtered speeds: {len(passed)} rows kept (|v| in [{vmin},{vmax}] px/s)")

    btn_apply_filter.on_clicked(apply_filter)

    # Save outputs
    def on_save(event):
        base = base_name
        out_png = folder / f"{base}_roi_dashboard.png"
        fig.savefig(out_png, dpi=200, bbox_inches='tight')
        
        spd_f_df = getattr(fig, '_speeds_filtered_df', None)
        spd_df = getattr(fig, '_speeds_df', None)
        
        if isinstance(spd_f_df, pd.DataFrame) and len(spd_f_df):
            csv_path = folder / f"{base}_speeds_filtered.csv"
            spd_f_df.to_csv(csv_path, index=False)
            n_rows = len(spd_f_df)
            mean_speed_px = spd_f_df['speed_px_per_s'].mean()
            std_speed_px = spd_f_df['speed_px_per_s'].std()
            mean_speed_um = spd_f_df['speed_um_per_s'].mean()
            std_speed_um = spd_f_df['speed_um_per_s'].std()
            print("=" * 60)
            print("📊 SUMMARY STATISTICS (FILTERED)")
            print("=" * 60)
            print(f"📁 Saved files:")
            print(f"   • {out_png}")
            print(f"   • {csv_path}")
            print()
            print(f"📈 Data points: {n_rows}")
            print(f"📏 Speed (px/s): {mean_speed_px:.2f} ± {std_speed_px:.2f}")
            print(f"📏 Speed (μm/s): {mean_speed_um:.2f} ± {std_speed_um:.2f}")
            if np.isfinite(mean_speed_px) and mean_speed_px != 0:
                print(f"📊 Coefficient of Variation: {(std_speed_px/mean_speed_px)*100:.1f}%")
            print("=" * 60)
        elif isinstance(spd_df, pd.DataFrame) and len(spd_df):
            csv_path = folder / f"{base}_speeds_all.csv"
            spd_df.to_csv(csv_path, index=False)
            n_rows = len(spd_df)
            mean_speed_px = spd_df['speed_px_per_s'].mean()
            std_speed_px = spd_df['speed_px_per_s'].std()
            mean_speed_um = spd_df['speed_um_per_s'].mean()
            std_speed_um = spd_df['speed_um_per_s'].std()
            print("=" * 60)
            print("📊 SUMMARY STATISTICS (ALL COMPUTED)")
            print("=" * 60)
            print(f"📁 Saved files:")
            print(f"   • {out_png}")
            print(f"   • {csv_path}")
            print()
            print(f"📈 Data points: {n_rows}")
            print(f"📏 Speed (px/s): {mean_speed_px:.2f} ± {std_speed_px:.2f}")
            print(f"📏 Speed (μm/s): {mean_speed_um:.2f} ± {std_speed_um:.2f}")
            if np.isfinite(mean_speed_px) and mean_speed_px != 0:
                print(f"📊 Coefficient of Variation: {(std_speed_px/mean_speed_px)*100:.1f}%")
            print("=" * 60)
        else:
            print(f"Saved dashboard: {out_png}")
            print("ℹ️  No speed data computed yet. Click 'Compute' first to save CSV.")

    btn_save.on_clicked(on_save)

    # Scale slider callbacks
    def on_x_scale_change(val):
        zoom = val
        if len(df_for_scale) == 0:
            return
        current_xlim = axes[0, 0].get_xlim()
        t_center = (current_xlim[0] + current_xlim[1]) / 2
        t_half_range = (t_range if t_range > 0 else 1.0) / (2 * zoom)
        x_min = t_center - t_half_range
        x_max = t_center + t_half_range
        for j in range(3):
            axes[0, j].set_xlim(x_min, x_max)
        fig.canvas.draw_idle()
    
    def on_y_scale_change(val):
        zoom = val
        if len(df_for_scale) == 0:
            return
        y_half_range = (x_data_max - x_data_min if (x_data_max > x_data_min) else 1.0) / (2 * zoom)
        for j in range(3):
            current_ylim = axes[0, j].get_ylim()
            y_center = (current_ylim[0] + current_ylim[1]) / 2
            axes[0, j].set_ylim(y_center - y_half_range, y_center + y_half_range)
        fig.canvas.draw_idle()
    
    def on_x_offset_left(event):
        for j in range(3):
            xlim = axes[0, j].get_xlim()
            x_shift = (xlim[1] - xlim[0]) * 0.1
            axes[0, j].set_xlim(xlim[0] - x_shift, xlim[1] - x_shift)
        fig.canvas.draw_idle()
    
    def on_x_offset_right(event):
        for j in range(3):
            xlim = axes[0, j].get_xlim()
            x_shift = (xlim[1] - xlim[0]) * 0.1
            axes[0, j].set_xlim(xlim[0] + x_shift, xlim[1] + x_shift)
        fig.canvas.draw_idle()
    
    def on_y_offset_up(event):
        for j in range(3):
            ylim = axes[0, j].get_ylim()
            y_shift = (ylim[1] - ylim[0]) * 0.1
            axes[0, j].set_ylim(ylim[0] + y_shift, ylim[1] + y_shift)
        fig.canvas.draw_idle()
    
    def on_y_offset_down(event):
        for j in range(3):
            ylim = axes[0, j].get_ylim()
            y_shift = (ylim[1] - ylim[0]) * 0.1
            axes[0, j].set_ylim(ylim[0] - y_shift, ylim[1] - y_shift)
        fig.canvas.draw_idle()

    def on_select_toggle(event):
        nonlocal select_mode, delete_mode
        if select_mode:
            select_mode = False
            btn_select.label.set_text('Select')
            btn_select.color = 'lightgray'
            btn_select.hovercolor = 'gray'
            print("Select mode OFF")
        else:
            if delete_mode:
                delete_mode = False
                btn_delete.label.set_text('Delete')
                btn_delete.color = 'lightgray'
                btn_delete.hovercolor = 'gray'
            select_mode = True
            btn_select.label.set_text('Exit Sel')
            btn_select.color = 'blue'
            btn_select.hovercolor = 'darkblue'
            print("Select mode ON - Draw rectangles to zoom")
        if rect_selection['rect_patch'] is not None:
            rect_selection['rect_patch'].remove()
            rect_selection.update({
                'active': False,
                'start_x': None,
                'start_y': None,
                'current_x': None,
                'current_y': None,
                'plot_index': None,
                'rect_patch': None
            })
        fig.canvas.draw_idle()

    def on_delete_toggle(event):
        nonlocal delete_mode, select_mode
        if delete_mode:
            delete_mode = False
            btn_delete.label.set_text('Delete')
            btn_delete.color = 'lightgray'
            btn_delete.hovercolor = 'gray'
            print("Delete mode OFF")
        else:
            if select_mode:
                select_mode = False
                btn_select.label.set_text('Select')
                btn_select.color = 'lightgray'
                btn_select.hovercolor = 'gray'
            delete_mode = True
            btn_delete.label.set_text('Exit Del')
            btn_delete.color = 'red'
            btn_delete.hovercolor = 'darkred'
            print("Delete mode ON - Draw rectangles to delete cell lines")
        if rect_selection['rect_patch'] is not None:
            rect_selection['rect_patch'].remove()
            rect_selection.update({
                'active': False,
                'start_x': None,
                'start_y': None,
                'current_x': None,
                'current_y': None,
                'plot_index': None,
                'rect_patch': None
            })
        fig.canvas.draw_idle()

    s_x_scale.on_changed(on_x_scale_change)
    s_y_scale.on_changed(on_y_scale_change)
    btn_x_left.on_clicked(on_x_offset_left)
    btn_x_right.on_clicked(on_x_offset_right)
    btn_y_up.on_clicked(on_y_offset_up)
    btn_y_down.on_clicked(on_y_offset_down)
    btn_select.on_clicked(on_select_toggle)
    btn_delete.on_clicked(on_delete_toggle)

    def on_autoscale(event):
        df_for_autoscale = current_df()
        if len(df_for_autoscale) == 0:
            print("No data to auto-scale")
            return
        for j, roi in enumerate(LABELS):
            roi_data = df_for_autoscale[df_for_autoscale['roi'] == roi]
            if len(roi_data) > 0:
                roi_data_filtered = roi_data[~roi_data.apply(
                    lambda row: (roi, row['cell_id']) in deleted_lines, axis=1
                )]
                target = roi_data_filtered if len(roi_data_filtered) > 0 else roi_data
                t_min, t_max = target['timestamp'].min(), target['timestamp'].max()
                x_min, x_max = target['cell_x'].min(), target['cell_x'].max()
                t_padding = (t_max - t_min) * 0.05
                x_padding = (x_max - x_min) * 0.05
                axes[0, j].set_xlim(t_min - t_padding, t_max + t_padding)
                axes[0, j].set_ylim(x_min - x_padding, x_max + x_padding)
        s_x_scale.reset()
        s_y_scale.reset()
        fig.canvas.draw_idle()
        print("Auto-scaled all plots to fit data")

    btn_autoscale.on_clicked(on_autoscale)

    plt.show()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ROI Tracking Dashboard with KF toggle and x1/x2 crossing-based speed measurement")
    ap.add_argument("settings_file", help="Path to settings file to infer base paths")
    args = ap.parse_args()
    dashboard(args.settings_file)
