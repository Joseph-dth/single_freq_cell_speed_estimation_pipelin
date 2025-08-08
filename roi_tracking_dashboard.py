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


def canonicalize_tracks_per_roi(df_roi: pd.DataFrame, max_cells: int, merge_gap_s: float, merge_dist_px: float) -> Dict[int, pd.DataFrame]:
    # df_roi contains single ROI rows
    # Step 1: group segments by original id
    segs = {}
    for cid, sub in df_roi.groupby('cell_id'):
        sub_sorted = sub.sort_values('timestamp')
        start_t = float(sub_sorted['timestamp'].iloc[0])
        end_t = float(sub_sorted['timestamp'].iloc[-1])
        last_x = float(sub_sorted['cell_x'].iloc[-1])
        last_y = float(sub_sorted['cell_y'].iloc[-1]) if 'cell_y' in sub_sorted.columns else 0.0
        first_x = float(sub_sorted['cell_x'].iloc[0])
        first_y = float(sub_sorted['cell_y'].iloc[0]) if 'cell_y' in sub_sorted.columns else 0.0
        segs[int(cid)] = {
            'df': sub_sorted,
            'start_t': start_t,
            'end_t': end_t,
            'first_xy': (first_x, first_y),
            'last_xy': (last_x, last_y),
            'frames': len(sub_sorted)
        }
    # Step 2: sort by start time and merge sequentially
    seg_items = sorted(segs.items(), key=lambda kv: kv[1]['start_t'])
    clusters = []  # list of dict with 'ids', 'start_t', 'end_t', 'last_xy', 'frames'
    for oid, meta in seg_items:
        assigned = False
        for cl in clusters:
            time_gap = meta['start_t'] - cl['end_t']
            if 0 <= time_gap <= merge_gap_s:
                dx = meta['first_xy'][0] - cl['last_xy'][0]
                dy = meta['first_xy'][1] - cl['last_xy'][1]
                if (dx*dx + dy*dy) ** 0.5 <= merge_dist_px:
                    cl['ids'].append(oid)
                    cl['end_t'] = max(cl['end_t'], meta['end_t'])
                    cl['last_xy'] = meta['last_xy']
                    cl['frames'] += meta['frames']
                    assigned = True
                    break
        if not assigned:
            clusters.append({
                'ids': [oid],
                'start_t': meta['start_t'],
                'end_t': meta['end_t'],
                'last_xy': meta['last_xy'],
                'frames': meta['frames']
            })
    # Step 3: sort clusters by coverage (frames) and cap
    clusters.sort(key=lambda cl: cl['frames'], reverse=True)
    clusters = clusters[:max_cells]
    # Step 4: build mapping original id -> canonical id (0..N-1)
    mapping: Dict[int, int] = {}
    for canon_id, cl in enumerate(clusters):
        for oid in cl['ids']:
            mapping[oid] = canon_id
    # Step 5: assemble grouped dfs by canonical id
    canon_groups: Dict[int, pd.DataFrame] = {}
    for oid, canon_id in mapping.items():
        df_seg = segs[oid]['df']
        if canon_id not in canon_groups:
            canon_groups[canon_id] = df_seg.copy()
        else:
            canon_groups[canon_id] = pd.concat([canon_groups[canon_id], df_seg], axis=0, ignore_index=True)
    # sort each canon by time
    for k in list(canon_groups.keys()):
        canon_groups[k] = canon_groups[k].sort_values('timestamp')
    return canon_groups


def canonicalize_and_group(df: pd.DataFrame, max_cells_per_roi: int, merge_gap_s: float, merge_dist_px: float) -> Dict[str, Dict[int, pd.DataFrame]]:
    out: Dict[str, Dict[int, pd.DataFrame]] = {l: {} for l in LABELS}
    for roi, sub in df.groupby('roi'):
        roi = str(roi)
        if roi not in out:
            continue
        out[roi] = canonicalize_tracks_per_roi(sub, max_cells_per_roi, merge_gap_s, merge_dist_px)
    return out


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

    # Figure with 2 rows x 3 cols: top=data, bottom=preview windows
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=False, sharey=False)
    fig.suptitle(f"ROI Tracking - {base_name}")

    # Plotter for trajectories
    def plot_groups(ax, groups_roi: Dict[int, pd.DataFrame], roi_name: str = ""):
        ax.clear()
        lines = []
        for cell_id, traj in groups_roi.items():
            # Skip deleted lines
            if (roi_name, cell_id) in deleted_lines:
                continue
            ts = traj['timestamp'].values
            xs = traj['cell_x'].values
            line = ax.plot(ts, xs, linewidth=1, alpha=0.85)[0]
            lines.append(line)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('x (px)')
        return lines

    # Initial draw for data row
    for j, label in enumerate(LABELS):
        axes[0, j].set_title(("Data-" + label) + (" (KF)" if use_kf else " (RAW)"))
        plot_groups(axes[0, j], groups.get(label, {}), label)

    # Two time lines across ROIs
    ts_min = float(current_df()['timestamp'].min()) if len(current_df()) else 0.0
    t_start = ts_min
    t_end = ts_min + 1.0
    vlines_tstart = [axes[0, j].axvline(t_start, color='cyan', linestyle='-', linewidth=1.8, alpha=0.9) for j in range(3)]
    vlines_tend   = [axes[0, j].axvline(t_end,   color='magenta', linestyle='-', linewidth=1.8, alpha=0.9) for j in range(3)]

    # Bottom row will show the same curves + preview of replicated windows
    for j, label in enumerate(LABELS):
        axes[1, j].set_title("Windows-" + label)
        axes[1, j].grid(True, alpha=0.3)
        axes[1, j].set_xlabel('time (s)')
        axes[1, j].set_ylabel('x (px)')

    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.27, top=0.9, wspace=0.3, hspace=0.35)

    # UI controls
    ax_chk = plt.axes([0.07, 0.18, 0.12, 0.06])
    chk = CheckButtons(ax_chk, ["Use KF"], [use_kf])

    ax_q = plt.axes([0.22, 0.19, 0.25, 0.03])
    ax_r = plt.axes([0.22, 0.15, 0.25, 0.03])
    s_q = Slider(ax_q, 'KF Q', 0.001, 10.0, valinit=q_init, valstep=0.001)
    s_r = Slider(ax_r, 'KF R', 0.001, 10.0, valinit=r_init, valstep=0.001)

    ax_apply = plt.axes([0.49, 0.165, 0.12, 0.05])
    btn_apply = Button(ax_apply, 'Apply KF')

    # Period/Repeats inputs - more compact
    ax_period = plt.axes([0.66, 0.19, 0.09, 0.04])
    ax_repeats = plt.axes([0.76, 0.19, 0.07, 0.04])
    tb_period = TextBox(ax_period, 'Period(s)', initial="12.0")
    tb_repeats = TextBox(ax_repeats, 'Repeats', initial="8")

    # Move time for speed dt, and min/max speed filter (px/s) - more compact
    ax_movetime = plt.axes([0.66, 0.15, 0.08, 0.04])
    tb_movetime = TextBox(ax_movetime, 'Move(s)', initial="1.0")
    ax_minspd = plt.axes([0.75, 0.15, 0.07, 0.04])
    ax_maxspd = plt.axes([0.83, 0.15, 0.07, 0.04])
    tb_minspd = TextBox(ax_minspd, 'Min v', initial="")
    tb_maxspd = TextBox(ax_maxspd, 'Max v', initial="")

    # Start/End time TextBoxes (smaller)
    ax_tstart = plt.axes([0.22, 0.11, 0.08, 0.04])
    ax_tend   = plt.axes([0.31, 0.11, 0.08, 0.04])
    tb_tstart = TextBox(ax_tstart, 'Start(s)', initial=f"{t_start:.3f}")
    tb_tend   = TextBox(ax_tend,   'End(s)',   initial=f"{t_end:.3f}")

    # Buttons: more compact layout in two rows
    # Row 1: Preview and Compute
    ax_preview = plt.axes([0.42, 0.11, 0.09, 0.04])
    btn_preview = Button(ax_preview, 'Preview')
    
    ax_compute = plt.axes([0.52, 0.11, 0.11, 0.04])
    btn_compute = Button(ax_compute, 'Compute')
    
    # Row 2: Apply Filter and Save
    ax_apply_filter = plt.axes([0.42, 0.06, 0.11, 0.04])
    btn_apply_filter = Button(ax_apply_filter, 'Apply Filter')
    
    ax_save = plt.axes([0.54, 0.06, 0.09, 0.04])
    btn_save = Button(ax_save, 'Save')

    # KF toggle
    def on_toggle_kf(label):
        nonlocal use_kf, groups
        use_kf = not use_kf
        groups = group_by_roi_and_id(current_df())
        for j, roi in enumerate(LABELS):
            axes[0, j].set_title(("Data-" + roi) + (" (KF)" if use_kf else " (RAW)"))
            plot_groups(axes[0, j], groups.get(roi, {}), roi)
            vlines_tstart[j].set_xdata([t_start, t_start])
            vlines_tend[j].set_xdata([t_end, t_end])
        fig.canvas.draw_idle()
    chk.on_clicked(on_toggle_kf)

    def on_apply_kf(event):
        nonlocal df_kf, groups
        qv = float(s_q.val)
        rv = float(s_r.val)
        df_kf = recompute_kf_from_raw(df_raw, qv, rv)
        if use_kf:
            groups = group_by_roi_and_id(df_kf)
            for j, roi in enumerate(LABELS):
                axes[0, j].set_title(("Data-" + roi) + " (KF)")
                plot_groups(axes[0, j], groups.get(roi, {}), roi)
            fig.canvas.draw_idle()
    btn_apply.on_clicked(on_apply_kf)

    # Dragging for two time lines
    drag_state = {"kind": None, "which": None}  # which: 't_start' or 't_end'

    def on_press(event):
        if event.inaxes is None:
            return
        # Only handle dragging for top row plots
        for j in range(3):
            if event.inaxes == axes[0, j]:
                x = event.xdata
                if x is None:
                    return
                # Only start dragging if click is very close to time lines
                if abs(x - t_start) < 0.2:
                    drag_state.update({"kind": 'v', "which": 't_start'})
                elif abs(x - t_end) < 0.2:
                    drag_state.update({"kind": 'v', "which": 't_end'})
                return

    def on_motion(event):
        nonlocal t_start, t_end
        if drag_state["kind"] != 'v' or event.inaxes is None:
            return
        x = event.xdata
        if x is None:
            return
        if drag_state["which"] == 't_start':
            t_start = float(x)
            for j in range(3):
                vlines_tstart[j].set_xdata([t_start, t_start])
            # sync box
            tb_tstart.set_val(f"{t_start:.3f}")
        else:
            t_end = float(x)
            for j in range(3):
                vlines_tend[j].set_xdata([t_end, t_end])
            # sync box
            tb_tend.set_val(f"{t_end:.3f}")
        fig.canvas.draw_idle()

    def on_release(event):
        drag_state.update({"kind": None, "which": None})

    # Line deletion handler
    def on_pick(event):
        line = event.artist
        if hasattr(line, 'roi_name') and hasattr(line, 'cell_id'):
            roi_name = line.roi_name
            cell_id = line.cell_id
            deleted_lines.add((roi_name, cell_id))
            print(f"Deleted line: {roi_name} cell_id={cell_id}")
            # Redraw the affected filtered plot (bottom row)
            speeds_df = getattr(fig, '_speeds_df', None)
            if speeds_df is not None and len(speeds_df) > 0:
                vmin, vmax = get_speed_bounds()
                # Use absolute speed values for filtering
                abs_speeds = np.abs(speeds_df['speed_px_per_s'])
                passed = speeds_df[np.isfinite(speeds_df['speed_px_per_s']) & (abs_speeds >= vmin) & (abs_speeds <= vmax)]
                pass_pairs = set((r, int(cid)) for r, cid in zip(passed['roi'], passed['cell_id']))
                groups_local = group_by_roi_and_id(current_df())
                
                for j, label in enumerate(LABELS):
                    if label == roi_name:
                        ax = axes[1, j]
                        ax.clear()
                        # plot only those pairs that passed at least once and are not deleted
                        for _id, traj in groups_local.get(label, {}).items():
                            if (label, int(_id)) in pass_pairs and (label, _id) not in deleted_lines:
                                line = ax.plot(traj['timestamp'].values, traj['cell_x'].values, linewidth=1.6, alpha=0.95, picker=True, pickradius=15)[0]
                                # Store metadata for click handling
                                line.roi_name = label
                                line.cell_id = _id
                        for (ts, te) in build_cycles():
                            ax.axvline(ts, color='cyan', linestyle=':', linewidth=1.0, alpha=0.9)
                            ax.axvline(te, color='magenta', linestyle=':', linewidth=1.0, alpha=0.9)
                        ax.set_title(f"Filtered-{label} |v|∈[{vmin:.2f},{vmax:.2f}] px/s")
                        ax.grid(True, alpha=0.3)
                        ax.set_xlabel('time (s)'); ax.set_ylabel('x (px)')
                        break
            fig.canvas.draw_idle()

    cid_pick = fig.canvas.mpl_connect('pick_event', on_pick)
    cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
    cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
    cid_release = fig.canvas.mpl_connect('button_release_event', on_release)

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


    # Preview replicated windows on bottom row
    def run_preview(event=None):
        cycles = build_cycles()
        groups_local = group_by_roi_and_id(current_df())
        for j, roi in enumerate(LABELS):
            ax = axes[1, j]
            ax.clear()
            plot_groups(ax, groups_local.get(roi, {}), roi)
            for (ts, te) in cycles:
                ax.axvline(ts, color='cyan', linestyle=':', linewidth=1.0, alpha=0.9)
                ax.axvline(te, color='magenta', linestyle=':', linewidth=1.0, alpha=0.9)
            ax.set_title("Windows-" + roi)
        fig.canvas.draw_idle()
    btn_preview.on_clicked(run_preview)

    # Compute speeds for all cells (use move time as dt)
    def parse_move_time() -> float:
        try:
            mv = float(tb_movetime.text)
            return mv if mv > 0 else 1.0
        except Exception:
            return 1.0

    def compute_speeds(event=None):
        cycles = build_cycles()
        move_s = parse_move_time()
        df = current_df()
        groups_local = group_by_roi_and_id(df)
        speed_rows = []
        for roi in LABELS:
            for _id, traj in groups_local.get(roi, {}).items():
                # Skip deleted lines
                if (roi, _id) in deleted_lines:
                    continue
                traj_sorted = traj.sort_values('timestamp')
                for cycle_idx, (ts, te) in enumerate(cycles):
                    x_s = sample_x_at_time(traj_sorted, ts, clamp=False)
                    x_e = sample_x_at_time(traj_sorted, te, clamp=False)
                    v_px_s = (x_e - x_s) / move_s if (np.isfinite(x_s) and np.isfinite(x_e)) else np.nan
                    v_um_s = v_px_s * scale_microns_per_pixel if (scale_microns_per_pixel is not None) else np.nan
                    speed_rows.append({
                        'roi': roi, 'cycle_index': cycle_idx, 'cell_id': int(_id),
                        't_start': ts, 't_end': te,
                        'move_s': move_s,
                        'x_start': x_s, 'x_end': x_e,
                        'speed_px_per_s': abs(v_px_s) if np.isfinite(v_px_s) else np.nan,
                        'speed_um_per_s': abs(v_um_s) if np.isfinite(v_um_s) else np.nan
                    })
        fig._speeds_df = pd.DataFrame(speed_rows)
        print(f"Computed speeds: {len(speed_rows)} rows (dt=Move(s))")
    btn_compute.on_clicked(compute_speeds)

    # Apply filter by min/max speed and update bottom plots to show only passing IDs
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
        # Clear deleted_lines when applying/reapplying filter
        deleted_lines.clear()
        speeds_df = getattr(fig, '_speeds_df', None)
        if speeds_df is None or len(speeds_df) == 0:
            print("No speeds computed yet.")
            return
        vmin, vmax = get_speed_bounds()
        # Use absolute speed values for filtering
        abs_speeds = np.abs(speeds_df['speed_px_per_s'])
        passed = speeds_df[np.isfinite(speeds_df['speed_px_per_s']) & (abs_speeds >= vmin) & (abs_speeds <= vmax)]
        # collect passing (roi, cell_id)
        pass_pairs = set((r, int(cid)) for r, cid in zip(passed['roi'], passed['cell_id']))
        groups_local = group_by_roi_and_id(current_df())
        # Also refresh the top row plots
        for j, roi in enumerate(LABELS):
            axes[0, j].clear()
            plot_groups(axes[0, j], groups.get(roi, {}), roi)
            axes[0, j].set_title(("Data-" + roi) + (" (KF)" if use_kf else " (RAW)"))
            axes[0, j].axvline(t_start, color='cyan', linestyle='-', linewidth=1.8, alpha=0.9)
            axes[0, j].axvline(t_end, color='magenta', linestyle='-', linewidth=1.8, alpha=0.9)
            # Update vlines references
            vlines_tstart[j] = axes[0, j].axvline(t_start, color='cyan', linestyle='-', linewidth=1.8, alpha=0.9)
            vlines_tend[j] = axes[0, j].axvline(t_end, color='magenta', linestyle='-', linewidth=1.8, alpha=0.9)
        for j, roi in enumerate(LABELS):
            ax = axes[1, j]
            ax.clear()
            # plot only those pairs that passed at least once and are not deleted
            for _id, traj in groups_local.get(roi, {}).items():
                if (roi, int(_id)) in pass_pairs and (roi, _id) not in deleted_lines:
                    line = ax.plot(traj['timestamp'].values, traj['cell_x'].values, linewidth=1.6, alpha=0.95, picker=True, pickradius=15)[0]
                    # Store metadata for click handling
                    line.roi_name = roi
                    line.cell_id = _id
            for (ts, te) in build_cycles():
                ax.axvline(ts, color='cyan', linestyle=':', linewidth=1.0, alpha=0.9)
                ax.axvline(te, color='magenta', linestyle=':', linewidth=1.0, alpha=0.9)
            ax.set_title(f"Filtered-{roi} |v|∈[{vmin:.2f},{vmax:.2f}] px/s")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('time (s)'); ax.set_ylabel('x (px)')
        fig.canvas.draw_idle()
        fig._speeds_filtered_df = passed.reset_index(drop=True)
        print(f"Filtered speeds: {len(passed)} rows kept (using absolute values)")

    btn_apply_filter.on_clicked(apply_filter)

    # Removed: Merge IDs checkbox and its handler

    # Save outputs
    def on_save(event):
        base = base_name
        out_png = folder / f"{base}_roi_dashboard.png"
        fig.savefig(out_png, dpi=200, bbox_inches='tight')
        
        # Only save filtered speeds CSV
        spd_f_df = getattr(fig, '_speeds_filtered_df', None)
        if isinstance(spd_f_df, pd.DataFrame) and len(spd_f_df):
            csv_path = folder / f"{base}_speeds_filtered.csv"
            spd_f_df.to_csv(csv_path, index=False)
            
            # Calculate and display summary statistics
            n_rows = len(spd_f_df)
            mean_speed_px = spd_f_df['speed_px_per_s'].mean()
            std_speed_px = spd_f_df['speed_px_per_s'].std()
            mean_speed_um = spd_f_df['speed_um_per_s'].mean()
            std_speed_um = spd_f_df['speed_um_per_s'].std()
            
            print("=" * 60)
            print("📊 SUMMARY STATISTICS")
            print("=" * 60)
            print(f"📁 Saved files:")
            print(f"   • {out_png}")
            print(f"   • {csv_path}")
            print()
            print(f"📈 Data points: {n_rows}")
            print(f"📏 Speed (px/s): {mean_speed_px:.2f} ± {std_speed_px:.2f}")
            print(f"📏 Speed (μm/s): {mean_speed_um:.2f} ± {std_speed_um:.2f}")
            print(f"📊 Coefficient of Variation: {(std_speed_px/mean_speed_px)*100:.1f}%")
            print("=" * 60)
        else:
            print(f"Saved dashboard: {out_png}")
            print("⚠️  No filtered speeds data to save. Run Apply Filter first.")

    btn_save.on_clicked(on_save)

    plt.show()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ROI Tracking Dashboard with KF toggle and two-point speed measurement")
    ap.add_argument("settings_file", help="Path to settings file to infer base paths")
    args = ap.parse_args()
    dashboard(args.settings_file) 