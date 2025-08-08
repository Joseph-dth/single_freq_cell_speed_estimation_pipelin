#!/usr/bin/env python3
"""
Integrated Cell Analysis Application
Combines all cell tracking pipeline functionality into a single tkinter-based GUI.

Features:
1. Scale calibration
2. Parameter tuning with position setting
3. ROI configuration
4. Cell tracking analysis
5. Results dashboard

Author: Integrated from ex_setting_v3.py, roi_setting_gui_tk.py, 
        cell_tracking_analysis.py, and roi_tracking_dashboard_v2.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import threading
import subprocess

try:
    from PIL import Image, ImageTk
except ImportError:
    messagebox.showerror("Missing Dependency", "Pillow is required: pip install pillow")
    sys.exit(1)

# Import the cell tracking analysis module
try:
    import cell_tracking_analysis
except ImportError:
    cell_tracking_analysis = None
    print("Warning: cell_tracking_analysis.py not found - analysis will be simulated")

# Import the ROI tracking dashboard module  
try:
    import roi_tracking_dashboard
except ImportError:
    roi_tracking_dashboard = None
    print("Warning: roi_tracking_dashboard_v2.py not found - dashboard will use basic visualization")

class IntegratedCellAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated Cell Analysis Pipeline")
        self.root.geometry("1200x800")
        
        # Application state
        self.video_path = None
        self.settings_path = None
        self.current_frame = None
        self.scale_ratio = None
        self.settings = {}
        
        # Analysis components
        self.cell_tracker = None
        self.dashboard_data = None
        self.dashboard_csv_path = None
        
        # GUI components
        self.create_menu()
        self.create_main_interface()
        
    def create_menu(self):
        """Create the main menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Video", command=self.load_video)
        file_menu.add_command(label="Load Settings", command=self.load_settings)
        file_menu.add_command(label="Save Settings", command=self.save_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Scale Calibration", command=self.show_scale_calibration)
        analysis_menu.add_command(label="Parameter Tuning", command=self.show_parameter_tuning)
        analysis_menu.add_command(label="ROI Configuration", command=self.show_roi_configuration)
        analysis_menu.add_command(label="Run Cell Tracking", command=self.run_cell_tracking)
        analysis_menu.add_command(label="View Dashboard", command=self.show_dashboard)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def create_main_interface(self):
        """Create the main interface with tabs for different functions"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # Tab 1: File Management
        self.create_file_tab()
        
        # Tab 2: Scale Calibration
        self.create_scale_tab()
        
        # Tab 3: Parameter Tuning
        self.create_parameter_tab()
        
        # Tab 4: ROI Configuration
        self.create_roi_tab()
        
        # Tab 5: Cell Tracking
        self.create_tracking_tab()
        
        # Tab 6: Results Dashboard
        self.create_dashboard_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please load a video file to begin")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_file_tab(self):
        """Create file management tab"""
        file_frame = ttk.Frame(self.notebook)
        self.notebook.add(file_frame, text="File Management")
        
        # Video section
        video_group = ttk.LabelFrame(file_frame, text="Video File", padding=10)
        video_group.pack(fill=tk.X, padx=10, pady=5)
        
        self.video_path_var = tk.StringVar()
        video_entry = ttk.Entry(video_group, textvariable=self.video_path_var, width=60)
        video_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(video_group, text="Browse", command=self.load_video).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(video_group, text="Quick Select", command=self.quick_select_video).pack(side=tk.LEFT)
        
        # Settings section
        settings_group = ttk.LabelFrame(file_frame, text="Settings File", padding=10)
        settings_group.pack(fill=tk.X, padx=10, pady=5)
        
        self.settings_path_var = tk.StringVar()
        settings_entry = ttk.Entry(settings_group, textvariable=self.settings_path_var, width=60)
        settings_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(settings_group, text="Browse", command=self.load_settings).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(settings_group, text="Save", command=self.save_settings).pack(side=tk.LEFT)
        
        # Current settings display
        settings_display_group = ttk.LabelFrame(file_frame, text="Current Settings", padding=10)
        settings_display_group.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollable text widget for settings
        text_frame = ttk.Frame(settings_display_group)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.settings_text = tk.Text(text_frame, height=15, width=80)
        settings_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.settings_text.yview)
        self.settings_text.configure(yscrollcommand=settings_scrollbar.set)
        
        self.settings_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        settings_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_scale_tab(self):
        """Create scale calibration tab"""
        scale_frame = ttk.Frame(self.notebook)
        self.notebook.add(scale_frame, text="Scale Calibration")
        
        # Instructions
        instructions = ttk.Label(scale_frame, text="Scale Calibration Instructions:\n1. Load a video first\n2. Click two points on a known distance\n3. Enter the actual length in microns\n4. Click 'Set Scale' to calibrate", justify=tk.LEFT)
        instructions.pack(pady=10)
        
        # Scale input frame
        input_frame = ttk.Frame(scale_frame)
        input_frame.pack(pady=10)
        
        ttk.Label(input_frame, text="Actual Length (μm):").pack(side=tk.LEFT, padx=5)
        self.scale_length_var = tk.StringVar()
        scale_entry = ttk.Entry(input_frame, textvariable=self.scale_length_var, width=15)
        scale_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(input_frame, text="Set Scale", command=self.set_scale).pack(side=tk.LEFT, padx=5)
        ttk.Button(input_frame, text="Reset", command=self.reset_scale).pack(side=tk.LEFT, padx=5)
        ttk.Button(input_frame, text="Refresh Image", command=self.refresh_scale_image).pack(side=tk.LEFT, padx=5)
        
        # Scale display
        self.scale_display_var = tk.StringVar()
        self.scale_display_var.set("Scale: Not set")
        scale_display = ttk.Label(scale_frame, textvariable=self.scale_display_var, font=('Arial', 12))
        scale_display.pack(pady=5)
        
        # Image display area
        self.scale_image_frame = ttk.LabelFrame(scale_frame, text="Video Frame", padding=10)
        self.scale_image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Canvas for image display and point selection
        self.scale_canvas = tk.Canvas(self.scale_image_frame, bg='black', width=640, height=480)
        self.scale_canvas.pack(expand=True)
        
        # Initialize scale calibration variables
        self.scale_points = []  # List to store clicked points
        self.scale_image_id = None
        self.scale_point_ids = []  # Track point markers on canvas
        
        # Bind canvas click event
        self.scale_canvas.bind("<Button-1>", self.on_scale_canvas_click)
        
    def create_parameter_tab(self):
        """Create parameter tuning tab"""
        param_frame = ttk.Frame(self.notebook)
        self.notebook.add(param_frame, text="Parameter Tuning")
        
        # Controls frame
        control_frame = ttk.Frame(param_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Left side - parameters
        param_left_frame = ttk.Frame(control_frame)
        param_left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Image processing parameters
        img_proc_frame = ttk.LabelFrame(param_left_frame, text="Image Processing", padding=5)
        img_proc_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Parameters in a more compact grid
        params = [
            ("Blur Kernel Size:", "blur_kernel_size", "5"),
            ("Threshold Value:", "threshold_value", "120"),
            ("Min Cell Area:", "min_cell_area", "25"),
            ("Max Cell Area:", "max_cell_area", "500"),
        ]
        
        self.param_vars = {}
        for i, (label, key, default) in enumerate(params):
            ttk.Label(img_proc_frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=2, pady=1)
            var = tk.StringVar(value=default)
            self.param_vars[key] = var
            ttk.Entry(img_proc_frame, textvariable=var, width=8).grid(row=i, column=1, padx=2, pady=1)
        
        # Experiment parameters (simplified)
        exp_frame = ttk.LabelFrame(param_left_frame, text="Experiment Settings", padding=5)
        exp_frame.pack(fill=tk.X, pady=(0, 5))
        
        exp_params = [
            ("Period (seconds):", "period_seconds", "8.0"),
            ("Repeat Count:", "repeat_count", "8"),
        ]
        
        for i, (label, key, default) in enumerate(exp_params):
            ttk.Label(exp_frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=2, pady=1)
            var = tk.StringVar(value=default)
            self.param_vars[key] = var
            ttk.Entry(exp_frame, textvariable=var, width=8).grid(row=i, column=1, padx=2, pady=1)
        
        # Buttons
        button_frame = ttk.Frame(param_left_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Apply Parameters", command=self.apply_parameters).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Refresh Image", command=self.refresh_param_image).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Test Detection", command=self.test_cell_detection).pack(fill=tk.X, pady=2)
        
        # Right side - image display
        image_frame = ttk.LabelFrame(control_frame, text="Parameter Testing", padding=5)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        self.param_canvas = tk.Canvas(image_frame, bg='black', width=640, height=480)
        self.param_canvas.pack(expand=True)
        
        # Initialize parameter tuning variables
        self.param_image_id = None
        
    def create_roi_tab(self):
        """Create ROI configuration tab"""
        roi_frame = ttk.Frame(self.notebook)
        self.notebook.add(roi_frame, text="ROI Configuration")
        
        # Top control frame
        control_frame = ttk.Frame(roi_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Left side controls
        left_controls = ttk.Frame(control_frame)
        left_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(left_controls, text="Active ROI:").pack(side=tk.LEFT, padx=5)
        
        self.active_roi_var = tk.StringVar(value="left")
        self.active_roi_var.trace_add('write', self.on_roi_selection_changed)
        
        for roi in ["left", "middle", "right"]:
            ttk.Radiobutton(left_controls, text=roi.capitalize(), value=roi, 
                          variable=self.active_roi_var).pack(side=tk.LEFT, padx=5)
        
        # Right side controls
        right_controls = ttk.Frame(control_frame)
        right_controls.pack(side=tk.RIGHT)
        
        ttk.Button(right_controls, text="Reset ROIs", command=self.reset_rois).pack(side=tk.RIGHT, padx=5)
        ttk.Button(right_controls, text="Refresh Image", command=self.refresh_roi_image).pack(side=tk.RIGHT, padx=5)
        ttk.Button(right_controls, text="Save ROIs", command=self.save_roi_settings).pack(side=tk.RIGHT, padx=5)
        
        # Main content frame with two sections
        main_content_frame = ttk.Frame(roi_frame)
        main_content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left side - Image display with ROI editing
        image_frame = ttk.LabelFrame(main_content_frame, text="ROI Editor - Click and drag to define regions", padding=5)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Canvas for ROI editing
        self.roi_canvas = tk.Canvas(image_frame, bg='black', width=640, height=480)
        self.roi_canvas.pack(expand=True)
        
        # Right side - ROI information and coordinates
        info_frame = ttk.LabelFrame(main_content_frame, text="ROI Information", padding=5)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # ROI coordinates display
        coord_frame = ttk.LabelFrame(info_frame, text="Coordinates", padding=5)
        coord_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.roi_text = tk.Text(coord_frame, height=15, width=35)
        roi_text_scroll = ttk.Scrollbar(coord_frame, orient="vertical", command=self.roi_text.yview)
        self.roi_text.configure(yscrollcommand=roi_text_scroll.set)
        self.roi_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        roi_text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(info_frame, text="Instructions", padding=5)
        instructions_frame.pack(fill=tk.X, pady=(5, 0))
        
        instructions_text = """ROI Editing:
• Select an ROI using radio buttons
• Click and drag on image to define rectangular region
• Drag corners to resize
• Drag inside to move
• Reset button restores default layout
• Save button updates settings file"""
        
        ttk.Label(instructions_frame, text=instructions_text, justify=tk.LEFT, wraplength=250).pack(anchor=tk.W)
        
        # Initialize ROI data
        self.roi_rects = {}  # Store ROI rectangles {roi_name: (x1, y1, x2, y2)}
        self.roi_image_id = None
        self.roi_canvas_rects = {}  # Canvas rectangle IDs
        self.roi_canvas_handles = {}  # Canvas handle IDs
        self.roi_drag_data = {
            'dragging': False,
            'drag_type': None,  # 'move' or 'resize' 
            'start_pos': None,
            'start_rect': None,
            'handle_index': -1
        }
        
        # ROI colors
        self.roi_colors = {
            'left': '#ff0000',    # Red
            'middle': '#ffff00',  # Yellow  
            'right': '#0066ff'    # Blue
        }
        
        # Bind canvas events
        self.roi_canvas.bind("<Button-1>", self.on_roi_canvas_click)
        self.roi_canvas.bind("<B1-Motion>", self.on_roi_canvas_drag)
        self.roi_canvas.bind("<ButtonRelease-1>", self.on_roi_canvas_release)
        
    def create_tracking_tab(self):
        """Create cell tracking analysis tab"""
        tracking_frame = ttk.Frame(self.notebook)
        self.notebook.add(tracking_frame, text="Cell Tracking")
        
        # Top control frame
        control_frame = ttk.LabelFrame(tracking_frame, text="Analysis Controls", padding=5)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Analysis options in one row
        options_frame = ttk.Frame(control_frame)
        options_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(options_frame, text="Max Frames (0=all):").pack(side=tk.LEFT, padx=5)
        self.max_frames_var = tk.StringVar(value="0")
        ttk.Entry(options_frame, textvariable=self.max_frames_var, width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(options_frame, text="Target FPS:").pack(side=tk.LEFT, padx=5)
        self.target_fps_var = tk.StringVar(value="")
        ttk.Entry(options_frame, textvariable=self.target_fps_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # Analysis buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(button_frame, text="Preview Detection", command=self.preview_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Start Analysis", command=self.start_cell_tracking).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Analysis", command=self.stop_cell_tracking).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=2)
        
        # Main content frame with two columns
        main_content_frame = ttk.Frame(tracking_frame)
        main_content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left side - Video preview
        video_frame = ttk.LabelFrame(main_content_frame, text="Detection Preview", padding=5)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Canvas for video display
        self.tracking_canvas = tk.Canvas(video_frame, bg='black', width=480, height=360)
        self.tracking_canvas.pack(expand=True)
        
        # Initialize tracking variables
        self.tracking_image_id = None
        
        # Right side - Results
        results_frame = ttk.LabelFrame(main_content_frame, text="Analysis Results", padding=5)
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(results_frame, height=20, width=40)
        results_scroll = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_dashboard_tab(self):
        """Create results dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Results Dashboard")
        
        # Dashboard controls
        control_frame = ttk.LabelFrame(dashboard_frame, text="Interactive Dashboard", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Open Dashboard", command=self.load_results, 
                  style="Accent.TButton").pack(side=tk.LEFT, padx=10)
        ttk.Button(control_frame, text="Export Data", command=self.export_data).pack(side=tk.LEFT, padx=5)
        
        # Instructions and info
        info_frame = ttk.LabelFrame(dashboard_frame, text="Dashboard Features", padding=15)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        instructions = """
The Interactive Dashboard provides advanced analysis and visualization tools:

🔍 ANALYSIS FEATURES:
  • ROI-based cell tracking visualization
  • Kalman filter toggle (raw vs filtered data)
  • Interactive speed measurement between two points
  • Cell trajectory analysis with time controls
  • Statistical summaries and filtering options

📊 VISUALIZATION OPTIONS:
  • Real-time plot updates with parameter controls
  • Cell position scatter plots by ROI
  • Speed distribution histograms
  • Customizable time range selection
  • Zoom and pan functionality

💾 EXPORT CAPABILITIES:
  • Export filtered datasets as CSV
  • Save high-quality dashboard images
  • Export speed measurements
  • Generate analysis reports

📋 WORKFLOW:
  1. Complete cell tracking analysis first
  2. Click 'Open Dashboard' to launch interactive window
  3. Use dashboard controls to explore your data
  4. Export results and visualizations as needed

The dashboard will automatically detect and load your CSV results
based on the current settings file.
        """
        
        text_widget = tk.Text(info_frame, height=20, width=80, wrap=tk.WORD, 
                             font=('Arial', 10), bg='#f8f8f8', relief=tk.FLAT)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, instructions.strip())
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Status info
        status_frame = ttk.Frame(dashboard_frame)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.dashboard_status_var = tk.StringVar()
        self.dashboard_status_var.set("Dashboard ready - complete analysis first, then click 'Open Dashboard'")
        ttk.Label(status_frame, textvariable=self.dashboard_status_var, 
                 font=('Arial', 9), foreground='#666').pack()
        
    # File management methods
    def load_video(self):
        """Load video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            initialdir="/Users/josephsyu/Documents/Need_Backup/2025_summer/Lab/Lab_Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            # Verify it's actually a file
            if os.path.isfile(file_path):
                self.video_path = file_path
                self.video_path_var.set(file_path)
                self.status_var.set(f"Loaded video: {os.path.basename(file_path)}")
                success = self.load_video_frame()
                if success:
                    messagebox.showinfo("Success", f"Successfully loaded video: {os.path.basename(file_path)}")
                else:
                    messagebox.showerror("Error", "Failed to load video frame. Please check if the file is a valid video.")
            else:
                messagebox.showerror("Error", "Selected path is not a valid file. Please select a video file.")
            
    def load_video_frame(self):
        """Load a frame from the video for processing"""
        if not self.video_path:
            return False
            
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Error: Cannot open video file {self.video_path}")
                return False
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if frame_count == 0:
                print(f"Error: Video has no frames")
                cap.release()
                return False
                
            # Get middle frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                self.current_frame = frame
                print(f"Successfully loaded video: {frame_count} frames, {fps:.2f} FPS")
                cap.release()
                
                # Auto-refresh scale image if on scale calibration tab
                try:
                    if hasattr(self, 'scale_canvas') and hasattr(self, 'notebook'):
                        current_tab = self.notebook.select()
                        if current_tab == self.notebook.tabs()[1]:  # Scale calibration tab (index 1)
                            self.root.after(100, self.refresh_scale_image)
                except:
                    pass  # Ignore errors during auto-refresh
                
                return True
            else:
                print(f"Error: Cannot read frame from video")
                cap.release()
                return False
                
        except Exception as e:
            print(f"Error loading video: {str(e)}")
            return False
    
    def quick_select_video(self):
        """Show a dialog with available video files for quick selection"""
        video_dir = "/Users/josephsyu/Documents/Need_Backup/2025_summer/Lab/Lab_Video"
        
        # Find all video files in the directory and subdirectories
        video_files = []
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, video_dir)
                    video_files.append((relative_path, full_path))
        
        if not video_files:
            messagebox.showwarning("No Videos", "No video files found in the directory.")
            return
        
        # Create selection dialog
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Select Video File")
        selection_window.geometry("500x400")
        selection_window.transient(self.root)
        selection_window.grab_set()
        
        # List of videos
        ttk.Label(selection_window, text="Available Video Files:", font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Scrollable listbox
        list_frame = ttk.Frame(selection_window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        listbox = tk.Listbox(list_frame, height=15)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        for rel_path, _ in video_files:
            listbox.insert(tk.END, rel_path)
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons
        button_frame = ttk.Frame(selection_window)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        def select_video():
            selection = listbox.curselection()
            if selection:
                selected_file = video_files[selection[0]][1]
                self.video_path = selected_file
                self.video_path_var.set(selected_file)
                self.status_var.set(f"Selected: {os.path.basename(selected_file)}")
                success = self.load_video_frame()
                if success:
                    messagebox.showinfo("Success", f"Successfully loaded: {os.path.basename(selected_file)}")
                selection_window.destroy()
            else:
                messagebox.showwarning("No Selection", "Please select a video file.")
        
        ttk.Button(button_frame, text="Select", command=select_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=selection_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # Double-click to select
        def on_double_click(event):
            select_video()
        listbox.bind('<Double-Button-1>', on_double_click)
        
        # Center the dialog
        selection_window.update_idletasks()
        x = (selection_window.winfo_screenwidth() // 2) - (selection_window.winfo_width() // 2)
        y = (selection_window.winfo_screenheight() // 2) - (selection_window.winfo_height() // 2)
        selection_window.geometry(f"+{x}+{y}")
            
    def load_settings(self):
        """Load settings file"""
        file_path = filedialog.askopenfilename(
            title="Select Settings File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            self.settings_path = file_path
            self.settings_path_var.set(file_path)
            self.read_settings_file()
            self.status_var.set(f"Loaded settings: {os.path.basename(file_path)}")
            
    def read_settings_file(self):
        """Read settings from file and update GUI"""
        if not self.settings_path:
            return
            
        self.settings = {}
        try:
            with open(self.settings_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and ':' in line:
                        key, value = line.split(':', 1)
                        self.settings[key.strip()] = value.strip()
            
            # Update GUI with loaded settings
            self.update_settings_display()
            self.update_parameter_gui()
            
            # If ROI canvas exists, reload ROI settings
            if hasattr(self, 'roi_canvas'):
                self.setup_default_rois()
                if hasattr(self, 'roi_img_offset_x'):  # Only redraw if image is loaded
                    self.redraw_roi_rectangles()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {str(e)}")
            
    def update_settings_display(self):
        """Update the settings display text widget"""
        self.settings_text.delete(1.0, tk.END)
        for key, value in self.settings.items():
            self.settings_text.insert(tk.END, f"{key}: {value}\n")
            
    def update_parameter_gui(self):
        """Update parameter GUI with loaded settings"""
        for key, var in self.param_vars.items():
            if key in self.settings:
                var.set(self.settings[key])
                
    def save_settings(self):
        """Save current settings to file"""
        if not self.settings_path:
            file_path = filedialog.asksaveasfilename(
                title="Save Settings File",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                self.settings_path = file_path
                self.settings_path_var.set(file_path)
            else:
                return
                
        # Update settings from GUI
        self.update_settings_from_gui()
        
        try:
            with open(self.settings_path, 'w') as f:
                f.write("# Cell Analysis Settings\n")
                f.write("# Generated by Integrated Cell Analysis App\n\n")
                
                for key, value in self.settings.items():
                    f.write(f"{key}: {value}\n")
                    
            self.status_var.set(f"Settings saved: {os.path.basename(self.settings_path)}")
            messagebox.showinfo("Success", "Settings saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
            
    def update_settings_from_gui(self):
        """Update settings dictionary from GUI values"""
        # Update from parameter variables
        for key, var in self.param_vars.items():
            self.settings[key] = var.get()
            
        # Add scale if set
        if self.scale_ratio:
            self.settings['scale_microns_per_pixel'] = str(self.scale_ratio)
            
    # Scale calibration methods
    def refresh_scale_image(self):
        """Refresh the image display in scale calibration tab"""
        if self.current_frame is None:
            messagebox.showwarning("No Video", "Please load a video file first.")
            return
        
        # Convert OpenCV image to PIL and then to tkinter
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize image to fit canvas while maintaining aspect ratio
        canvas_width = self.scale_canvas.winfo_width() if self.scale_canvas.winfo_width() > 1 else 640
        canvas_height = self.scale_canvas.winfo_height() if self.scale_canvas.winfo_height() > 1 else 480
        
        # Calculate resize ratio
        img_width, img_height = pil_image.size
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Resize and convert to PhotoImage
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.scale_photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.scale_canvas.delete("all")
        self.scale_image_id = self.scale_canvas.create_image(
            canvas_width // 2, canvas_height // 2, 
            anchor=tk.CENTER, image=self.scale_photo
        )
        
        # Store image dimensions for coordinate calculation
        self.scale_img_width = new_width
        self.scale_img_height = new_height
        self.scale_img_ratio = ratio
        self.scale_img_offset_x = (canvas_width - new_width) // 2
        self.scale_img_offset_y = (canvas_height - new_height) // 2
        
        # Reset points
        self.scale_points = []
        self.scale_point_ids = []
        
        self.status_var.set("Image refreshed - click two points to set scale")
    
    def on_scale_canvas_click(self, event):
        """Handle click events on the scale calibration canvas"""
        if self.scale_image_id is None:
            messagebox.showwarning("No Image", "Please refresh the image first.")
            return
        
        # Convert canvas coordinates to image coordinates
        canvas_x = event.x
        canvas_y = event.y
        
        # Check if click is within image bounds
        if (canvas_x < self.scale_img_offset_x or 
            canvas_x > self.scale_img_offset_x + self.scale_img_width or
            canvas_y < self.scale_img_offset_y or
            canvas_y > self.scale_img_offset_y + self.scale_img_height):
            return
        
        # Convert to image coordinates
        img_x = canvas_x - self.scale_img_offset_x
        img_y = canvas_y - self.scale_img_offset_y
        
        # Store point (in original image coordinates)
        original_x = img_x / self.scale_img_ratio
        original_y = img_y / self.scale_img_ratio
        
        if len(self.scale_points) < 2:
            self.scale_points.append((original_x, original_y))
            
            # Draw point marker on canvas
            marker_id = self.scale_canvas.create_oval(
                canvas_x - 5, canvas_y - 5, canvas_x + 5, canvas_y + 5,
                fill='red', outline='white', width=2
            )
            self.scale_point_ids.append(marker_id)
            
            # Add point label
            label_id = self.scale_canvas.create_text(
                canvas_x + 15, canvas_y - 15, 
                text=f"P{len(self.scale_points)}", 
                fill='red', font=('Arial', 12, 'bold')
            )
            self.scale_point_ids.append(label_id)
            
            if len(self.scale_points) == 2:
                # Draw line between points
                p1_canvas = (
                    int(self.scale_points[0][0] * self.scale_img_ratio) + self.scale_img_offset_x,
                    int(self.scale_points[0][1] * self.scale_img_ratio) + self.scale_img_offset_y
                )
                p2_canvas = (
                    int(self.scale_points[1][0] * self.scale_img_ratio) + self.scale_img_offset_x,
                    int(self.scale_points[1][1] * self.scale_img_ratio) + self.scale_img_offset_y
                )
                
                line_id = self.scale_canvas.create_line(
                    p1_canvas[0], p1_canvas[1], p2_canvas[0], p2_canvas[1],
                    fill='blue', width=3
                )
                self.scale_point_ids.append(line_id)
                
                # Calculate pixel distance
                pixel_distance = math.sqrt(
                    (self.scale_points[1][0] - self.scale_points[0][0]) ** 2 +
                    (self.scale_points[1][1] - self.scale_points[0][1]) ** 2
                )
                
                self.status_var.set(f"Two points selected - distance: {pixel_distance:.1f} pixels")
                messagebox.showinfo("Points Selected", 
                    f"Selected two points!\nPixel distance: {pixel_distance:.1f}\nNow enter the actual length and click 'Set Scale'.")
        else:
            # Reset if already have 2 points
            self.reset_scale_points()
            self.on_scale_canvas_click(event)  # Add the new point
    
    def reset_scale_points(self):
        """Reset the selected points"""
        self.scale_points = []
        # Remove point markers from canvas
        for point_id in self.scale_point_ids:
            self.scale_canvas.delete(point_id)
        self.scale_point_ids = []
        
    def set_scale(self):
        """Set the scale calibration"""
        if len(self.scale_points) != 2:
            messagebox.showwarning("Incomplete", "Please select exactly two points first.")
            return
        
        try:
            length = float(self.scale_length_var.get())
            if length <= 0:
                raise ValueError("Length must be positive")
                
            # Calculate pixel distance
            pixel_distance = math.sqrt(
                (self.scale_points[1][0] - self.scale_points[0][0]) ** 2 +
                (self.scale_points[1][1] - self.scale_points[0][1]) ** 2
            )
            
            if pixel_distance == 0:
                messagebox.showerror("Error", "The two points are at the same location.")
                return
                
            self.scale_ratio = length / pixel_distance
            self.scale_display_var.set(f"Scale: {self.scale_ratio:.4f} μm/pixel")
            self.status_var.set("Scale calibration set successfully!")
            
            messagebox.showinfo("Success", 
                f"Scale calibration set!\n{self.scale_ratio:.4f} μm/pixel\nPixel distance: {pixel_distance:.1f} px\nActual length: {length} μm")
            
        except ValueError as e:
            messagebox.showerror("Error", "Please enter a valid positive length in microns.")
            
    def reset_scale(self):
        """Reset scale calibration"""
        self.scale_ratio = None
        self.scale_display_var.set("Scale: Not set")
        self.scale_length_var.set("")
        self.reset_scale_points()
        self.status_var.set("Scale calibration reset")
        
    # Parameter tuning methods
    def refresh_param_image(self):
        """Refresh the image display in parameter tuning tab"""
        if self.current_frame is None:
            messagebox.showwarning("No Video", "Please load a video file first.")
            return
        
        try:
            # Show original image
            self.display_image_on_canvas(self.param_canvas, self.current_frame, "param")
            self.status_var.set("Parameter image refreshed - click 'Test Detection' to see results")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh image: {str(e)}")
    
    def apply_parameters(self):
        """Apply current parameters and show detection results"""
        if self.current_frame is None:
            self.refresh_param_image()
            return
        
        # Automatically run detection with new parameters
        self.test_cell_detection()
    
    def test_cell_detection(self):
        """Test cell detection with current parameters"""
        if self.current_frame is None:
            messagebox.showwarning("No Video", "Please load a video file first.")
            return
        
        try:
            print("Starting cell detection test...")
            
            # Get parameters
            blur_kernel = int(self.param_vars["blur_kernel_size"].get())
            threshold_val = int(self.param_vars["threshold_value"].get())
            min_area = int(self.param_vars["min_cell_area"].get())
            max_area = int(self.param_vars["max_cell_area"].get())
            
            print(f"Parameters: blur={blur_kernel}, threshold={threshold_val}, area={min_area}-{max_area}")
            
            # Process image
            frame = self.current_frame.copy()
            centers = self.detect_cells_in_frame_simple(frame, blur_kernel, threshold_val, min_area, max_area)
            
            print(f"Detected {len(centers)} cells: {centers[:5]}...")  # Show first 5
            
            # Draw detected cells with better visibility
            for i, center in enumerate(centers):
                # Draw larger, more visible circles
                cv2.circle(frame, center, 12, (0, 255, 0), 3)
                cv2.circle(frame, center, 4, (255, 255, 255), -1)  # White center dot
                
                # Draw cell number and coordinates
                cv2.putText(frame, f"#{i+1}", 
                          (center[0] - 10, center[1] - 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f"({center[0]},{center[1]})", 
                          (center[0] + 15, center[1] + 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            print("Drawing cells on frame completed")
            
            # Display result
            self.display_image_on_canvas(self.param_canvas, frame, "param")
            self.status_var.set(f"✅ Detected {len(centers)} cells with current parameters")
            
            print("Image displayed on canvas")
            
        except Exception as e:
            error_msg = f"Detection failed: {str(e)}"
            print(f"Error: {error_msg}")
            messagebox.showerror("Error", error_msg)
    
    def detect_cells_in_frame_simple(self, frame, blur_kernel, threshold_val, min_area, max_area):
        """Simple cell detection for parameter testing"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Ensure blur kernel is odd
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed = cv2.erode(thresh, kernel, iterations=2)
        processed = cv2.bitwise_not(processed)
        
        # Blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 5
        params.maxThreshold = 255
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area
        params.filterByCircularity = True
        params.minCircularity = 0.6
        params.filterByConvexity = False
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        params.maxInertiaRatio = 1.0
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(processed)
        centers = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
        
        return sorted(centers, key=lambda c: c[0])
    
    def display_image_on_canvas(self, canvas, frame, prefix):
        """Display image on a canvas with proper scaling"""
        try:
            print(f"Displaying image on canvas with prefix: {prefix}")
            
            # Convert OpenCV image to PIL and then to tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Get canvas dimensions
            canvas.update_idletasks()
            canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else 640
            canvas_height = canvas.winfo_height() if canvas.winfo_height() > 1 else 480
            
            print(f"Canvas dimensions: {canvas_width}x{canvas_height}")
            print(f"Original image size: {pil_image.size}")
            
            # Calculate resize ratio
            img_width, img_height = pil_image.size
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            print(f"Resizing to: {new_width}x{new_height}, ratio: {ratio}")
            
            # Resize and convert to PhotoImage
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Store photo reference to prevent garbage collection
            setattr(self, f"{prefix}_photo", photo)
            
            # Clear canvas and display image
            canvas.delete("all")
            image_id = canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=photo)
            print(f"Image created with ID: {image_id}")
            
        except Exception as e:
            print(f"Error in display_image_on_canvas: {str(e)}")
            raise
    
    def preview_detection(self):
        """Preview cell detection on tracking tab"""
        if self.current_frame is None:
            messagebox.showwarning("No Video", "Please load a video file first.")
            return
        
        try:
            # Use current parameters
            blur_kernel = int(self.param_vars.get("blur_kernel_size", tk.StringVar(value="5")).get())
            threshold_val = int(self.param_vars.get("threshold_value", tk.StringVar(value="120")).get())
            min_area = int(self.param_vars.get("min_cell_area", tk.StringVar(value="25")).get())
            max_area = int(self.param_vars.get("max_cell_area", tk.StringVar(value="500")).get())
            
            # Process image
            frame = self.current_frame.copy()
            centers = self.detect_cells_in_frame_simple(frame, blur_kernel, threshold_val, min_area, max_area)
            
            # Draw detected cells with better visibility
            for i, center in enumerate(centers):
                # Draw larger, more visible circles
                cv2.circle(frame, center, 12, (0, 255, 0), 3)
                cv2.circle(frame, center, 4, (255, 255, 255), -1)  # White center dot
                
                # Draw cell number and coordinates
                cv2.putText(frame, f"#{i+1}", 
                          (center[0] - 10, center[1] - 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f"({center[0]},{center[1]})", 
                          (center[0] + 15, center[1] + 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Display result on tracking canvas
            self.display_image_on_canvas(self.tracking_canvas, frame, "tracking")
            self.status_var.set(f"Preview: Detected {len(centers)} cells")
            
        except Exception as e:
            messagebox.showerror("Error", f"Preview failed: {str(e)}")
        
    # ROI methods - Enhanced functionality
    def refresh_roi_image(self):
        """Refresh the image display in ROI configuration tab"""
        if self.current_frame is None:
            messagebox.showwarning("No Video", "Please load a video file first.")
            return
        
        try:
            # Convert OpenCV image to PIL and then to tkinter
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Resize image to fit canvas while maintaining aspect ratio
            canvas_width = self.roi_canvas.winfo_width() if self.roi_canvas.winfo_width() > 1 else 640
            canvas_height = self.roi_canvas.winfo_height() if self.roi_canvas.winfo_height() > 1 else 480
            
            # Calculate resize ratio
            img_width, img_height = pil_image.size
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            # Resize and convert to PhotoImage
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.roi_photo = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and display image
            self.roi_canvas.delete("all")
            self.roi_image_id = self.roi_canvas.create_image(
                canvas_width // 2, canvas_height // 2, 
                anchor=tk.CENTER, image=self.roi_photo
            )
            
            # Store image dimensions for coordinate calculation
            self.roi_img_width = new_width
            self.roi_img_height = new_height
            self.roi_img_ratio = ratio
            self.roi_img_offset_x = (canvas_width - new_width) // 2
            self.roi_img_offset_y = (canvas_height - new_height) // 2
            
            # Setup default ROIs if not already set
            if not self.roi_rects:
                self.setup_default_rois()
            
            # Redraw ROI rectangles
            self.redraw_roi_rectangles()
            
            self.status_var.set("ROI image refreshed - select ROI and click/drag to edit")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh ROI image: {str(e)}")
    
    def setup_default_rois(self):
        """Setup ROI rectangles - load from settings if available, otherwise use defaults"""
        if not hasattr(self, 'roi_img_width') or not self.roi_img_width:
            # Use default canvas dimensions
            img_width = 640
            img_height = 480
            scale_factor = 1.0
        else:
            img_width = self.roi_img_width
            img_height = self.roi_img_height
            scale_factor = self.roi_img_ratio if hasattr(self, 'roi_img_ratio') else 1.0
        
        # Try to load ROI settings from existing settings
        rois_loaded = False
        if self.settings:
            try:
                for roi_name in ['left', 'middle', 'right']:
                    x1_key = f"{roi_name}_roi_x1"
                    y1_key = f"{roi_name}_roi_y1" 
                    x2_key = f"{roi_name}_roi_x2"
                    y2_key = f"{roi_name}_roi_y2"
                    
                    if all(key in self.settings for key in [x1_key, y1_key, x2_key, y2_key]):
                        # Convert from original image coordinates to display coordinates
                        orig_x1 = int(float(self.settings[x1_key]))
                        orig_y1 = int(float(self.settings[y1_key]))
                        orig_x2 = int(float(self.settings[x2_key]))
                        orig_y2 = int(float(self.settings[y2_key]))
                        
                        # Scale to display coordinates
                        disp_x1 = int(orig_x1 * scale_factor)
                        disp_y1 = int(orig_y1 * scale_factor)
                        disp_x2 = int(orig_x2 * scale_factor)
                        disp_y2 = int(orig_y2 * scale_factor)
                        
                        # Ensure within bounds
                        disp_x1 = max(0, min(disp_x1, img_width - 1))
                        disp_y1 = max(0, min(disp_y1, img_height - 1))
                        disp_x2 = max(0, min(disp_x2, img_width - 1))
                        disp_y2 = max(0, min(disp_y2, img_height - 1))
                        
                        if not hasattr(self, 'roi_rects'):
                            self.roi_rects = {}
                        self.roi_rects[roi_name] = (disp_x1, disp_y1, disp_x2, disp_y2)
                        rois_loaded = True
                
                if rois_loaded:
                    self.status_var.set("ROI settings loaded from settings file")
                    
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not load ROI settings from file: {e}")
                rois_loaded = False
        
        # If no ROI settings loaded, create default 3-column layout
        if not rois_loaded or not hasattr(self, 'roi_rects') or not self.roi_rects:
            col_width = img_width // 3
            self.roi_rects = {
                'left': (0, 0, col_width - 1, img_height - 1),
                'middle': (col_width, 0, col_width * 2 - 1, img_height - 1),
                'right': (col_width * 2, 0, img_width - 1, img_height - 1)
            }
            self.status_var.set("Default ROI layout created")
        
        # Update coordinates display
        self.update_roi_coordinates_display()
    
    def redraw_roi_rectangles(self):
        """Draw ROI rectangles on the canvas"""
        # Clear existing ROI graphics
        self.roi_canvas.delete("roi")
        self.roi_canvas_rects = {}
        self.roi_canvas_handles = {}
        
        if not hasattr(self, 'roi_img_offset_x'):
            return
            
        for roi_name, (x1, y1, x2, y2) in self.roi_rects.items():
            # Convert to canvas coordinates
            canvas_x1 = x1 + self.roi_img_offset_x
            canvas_y1 = y1 + self.roi_img_offset_y
            canvas_x2 = x2 + self.roi_img_offset_x
            canvas_y2 = y2 + self.roi_img_offset_y
            
            color = self.roi_colors[roi_name]
            active_roi = self.active_roi_var.get()
            
            # Draw main rectangle
            line_width = 3 if roi_name == active_roi else 2
            rect_id = self.roi_canvas.create_rectangle(
                canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                outline=color, width=line_width, tags="roi"
            )
            self.roi_canvas_rects[roi_name] = rect_id
            
            # Draw corner handles for active ROI
            if roi_name == active_roi:
                handle_size = 6
                handles = []
                corners = [(canvas_x1, canvas_y1), (canvas_x2, canvas_y1), 
                          (canvas_x1, canvas_y2), (canvas_x2, canvas_y2)]
                for cx, cy in corners:
                    handle_id = self.roi_canvas.create_rectangle(
                        cx - handle_size, cy - handle_size,
                        cx + handle_size, cy + handle_size,
                        fill=color, outline='white', width=1, tags="roi"
                    )
                    handles.append(handle_id)
                self.roi_canvas_handles[roi_name] = handles
            
            # Draw ROI label
            label_x = canvas_x1 + 5
            label_y = canvas_y1 + 15
            self.roi_canvas.create_text(
                label_x, label_y, text=roi_name.upper(), 
                anchor=tk.W, fill=color, font=("Arial", 10, "bold"), tags="roi"
            )
    
    def on_roi_selection_changed(self, *args):
        """Handle ROI selection change"""
        self.redraw_roi_rectangles()
    
    def on_roi_canvas_click(self, event):
        """Handle mouse click on ROI canvas"""
        if self.roi_image_id is None:
            messagebox.showwarning("No Image", "Please refresh the image first.")
            return
        
        active_roi = self.active_roi_var.get()
        
        # Convert canvas coordinates to image coordinates
        if not hasattr(self, 'roi_img_offset_x'):
            return
            
        canvas_x = event.x
        canvas_y = event.y
        
        # Check if click is within image bounds
        if (canvas_x < self.roi_img_offset_x or 
            canvas_x > self.roi_img_offset_x + self.roi_img_width or
            canvas_y < self.roi_img_offset_y or
            canvas_y > self.roi_img_offset_y + self.roi_img_height):
            return
        
        img_x = canvas_x - self.roi_img_offset_x
        img_y = canvas_y - self.roi_img_offset_y
        
        # Check if clicking on a handle of the active ROI
        if active_roi in self.roi_rects:
            x1, y1, x2, y2 = self.roi_rects[active_roi]
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            
            for i, (corner_x, corner_y) in enumerate(corners):
                if abs(img_x - corner_x) <= 8 and abs(img_y - corner_y) <= 8:
                    # Clicked on a handle - start resize
                    self.roi_drag_data = {
                        'dragging': True,
                        'drag_type': 'resize',
                        'start_pos': (img_x, img_y),
                        'start_rect': (x1, y1, x2, y2),
                        'handle_index': i
                    }
                    return
            
            # Check if clicking inside the rectangle - start move
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                self.roi_drag_data = {
                    'dragging': True,
                    'drag_type': 'move',
                    'start_pos': (img_x, img_y),
                    'start_rect': (x1, y1, x2, y2),
                    'handle_index': -1
                }
                return
        
        # Click outside existing ROI - create new ROI
        self.roi_rects[active_roi] = (img_x, img_y, img_x, img_y)
        self.roi_drag_data = {
            'dragging': True,
            'drag_type': 'resize',
            'start_pos': (img_x, img_y),
            'start_rect': (img_x, img_y, img_x, img_y),
            'handle_index': 3  # Bottom-right corner
        }
        self.redraw_roi_rectangles()
    
    def on_roi_canvas_drag(self, event):
        """Handle mouse drag on ROI canvas"""
        if not self.roi_drag_data['dragging']:
            return
            
        active_roi = self.active_roi_var.get()
        
        # Convert to image coordinates
        canvas_x = max(self.roi_img_offset_x, min(event.x, self.roi_img_offset_x + self.roi_img_width))
        canvas_y = max(self.roi_img_offset_y, min(event.y, self.roi_img_offset_y + self.roi_img_height))
        img_x = canvas_x - self.roi_img_offset_x
        img_y = canvas_y - self.roi_img_offset_y
        
        start_x, start_y = self.roi_drag_data['start_pos']
        x1, y1, x2, y2 = self.roi_drag_data['start_rect']
        
        if self.roi_drag_data['drag_type'] == 'move':
            # Move the entire rectangle
            dx = img_x - start_x
            dy = img_y - start_y
            
            new_x1 = max(0, min(x1 + dx, self.roi_img_width - (x2 - x1)))
            new_y1 = max(0, min(y1 + dy, self.roi_img_height - (y2 - y1)))
            new_x2 = new_x1 + (x2 - x1)
            new_y2 = new_y1 + (y2 - y1)
            
            self.roi_rects[active_roi] = (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
            
        elif self.roi_drag_data['drag_type'] == 'resize':
            # Resize by dragging corner
            handle_idx = self.roi_drag_data['handle_index']
            
            new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
            
            if handle_idx == 0:  # Top-left
                new_x1, new_y1 = img_x, img_y
            elif handle_idx == 1:  # Top-right
                new_x2, new_y1 = img_x, img_y
            elif handle_idx == 2:  # Bottom-left
                new_x1, new_y2 = img_x, img_y
            elif handle_idx == 3:  # Bottom-right
                new_x2, new_y2 = img_x, img_y
            
            # Ensure coordinates are in correct order and within bounds
            new_x1 = max(0, min(new_x1, self.roi_img_width - 1))
            new_y1 = max(0, min(new_y1, self.roi_img_height - 1))
            new_x2 = max(0, min(new_x2, self.roi_img_width - 1))
            new_y2 = max(0, min(new_y2, self.roi_img_height - 1))
            
            # Normalize rectangle (ensure x1 < x2, y1 < y2)
            x1, x2 = min(new_x1, new_x2), max(new_x1, new_x2)
            y1, y2 = min(new_y1, new_y2), max(new_y1, new_y2)
            
            self.roi_rects[active_roi] = (int(x1), int(y1), int(x2), int(y2))
        
        self.redraw_roi_rectangles()
        self.update_roi_coordinates_display()
    
    def on_roi_canvas_release(self, event):
        """Handle mouse release on ROI canvas"""
        self.roi_drag_data['dragging'] = False
    
    def update_roi_coordinates_display(self):
        """Update the ROI coordinates text display"""
        self.roi_text.delete(1.0, tk.END)
        
        # Calculate original image coordinates if we have scaling info
        if hasattr(self, 'roi_img_ratio') and self.roi_img_ratio:
            scale_factor = 1.0 / self.roi_img_ratio
        else:
            scale_factor = 1.0
        
        roi_info = "Current ROI Settings:\n\n"
        
        for roi_name in ['left', 'middle', 'right']:
            if roi_name in self.roi_rects:
                x1, y1, x2, y2 = self.roi_rects[roi_name]
                
                # Convert to original image coordinates
                orig_x1 = int(x1 * scale_factor)
                orig_y1 = int(y1 * scale_factor)
                orig_x2 = int(x2 * scale_factor)
                orig_y2 = int(y2 * scale_factor)
                
                roi_info += f"{roi_name.upper()} ROI:\n"
                roi_info += f"  Top-left: ({orig_x1}, {orig_y1})\n"
                roi_info += f"  Bottom-right: ({orig_x2}, {orig_y2})\n"
                roi_info += f"  Size: {orig_x2-orig_x1} x {orig_y2-orig_y1}\n"
                roi_info += f"  Display: ({x1}, {y1}) to ({x2}, {y2})\n\n"
            else:
                roi_info += f"{roi_name.upper()} ROI: Not defined\n\n"
        
        self.roi_text.insert(tk.END, roi_info)
    
    def save_roi_settings(self):
        """Save ROI settings to the settings file and automatically reload"""
        if not self.roi_rects:
            messagebox.showwarning("No ROIs", "Please define ROI regions first.")
            return
            
        if not self.settings_path:
            # If no settings file exists, create one
            if not self.video_path:
                messagebox.showwarning("No Video", "Please load a video file first to create settings.")
                return
                
            # Auto-generate settings filename based on video
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            settings_filename = f"{video_name}_setting.txt"
            self.settings_path = os.path.join(os.path.dirname(self.video_path), settings_filename)
            self.settings_path_var.set(self.settings_path)
        
        try:
            # Calculate scale factor for original coordinates
            if hasattr(self, 'roi_img_ratio') and self.roi_img_ratio:
                scale_factor = 1.0 / self.roi_img_ratio
            else:
                scale_factor = 1.0
            
            # Update settings with ROI coordinates
            for roi_name in ['left', 'middle', 'right']:
                if roi_name in self.roi_rects:
                    x1, y1, x2, y2 = self.roi_rects[roi_name]
                    
                    # Convert to original image coordinates
                    orig_x1 = int(x1 * scale_factor)
                    orig_y1 = int(y1 * scale_factor)
                    orig_x2 = int(x2 * scale_factor)
                    orig_y2 = int(y2 * scale_factor)
                    
                    self.settings[f"{roi_name}_roi_x1"] = str(orig_x1)
                    self.settings[f"{roi_name}_roi_y1"] = str(orig_y1)
                    self.settings[f"{roi_name}_roi_x2"] = str(orig_x2)
                    self.settings[f"{roi_name}_roi_y2"] = str(orig_y2)
            
            # Add ROI detection flag
            self.settings["roi_crop_detection"] = "1"
            
            # Update settings from GUI parameters
            self.update_settings_from_gui()
            
            # Write the settings file
            with open(self.settings_path, 'w') as f:
                f.write("# Cell Analysis Settings\n")
                f.write("# Generated by Integrated Cell Analysis App\n")
                f.write("# ROI Configuration Settings\n\n")
                
                for key, value in self.settings.items():
                    f.write(f"{key}: {value}\n")
            
            # Automatically reload the settings to update the display
            self.read_settings_file()
            
            # Update both ROI coordinates display and main settings display
            self.update_roi_coordinates_display()
            
            messagebox.showinfo("Success", 
                f"ROI settings saved to {os.path.basename(self.settings_path)}\n"
                f"Settings automatically reloaded and updated in interface.")
            self.status_var.set(f"ROI settings saved and reloaded: {os.path.basename(self.settings_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save ROI settings: {str(e)}")

    def reset_rois(self):
        """Reset ROI configuration to default"""
        # Default ROI setup
        default_rois = {
            'left': (0, 0, 213, 480),
            'middle': (214, 0, 427, 480),
            'right': (428, 0, 640, 480)
        }
        
        roi_text = "Default ROI Configuration:\n\n"
        for roi, (x1, y1, x2, y2) in default_rois.items():
            roi_text += f"{roi.capitalize()} ROI: ({x1}, {y1}) to ({x2}, {y2})\n"
            
        self.roi_text.delete(1.0, tk.END)
        self.roi_text.insert(tk.END, roi_text)
        
    # Cell tracking methods
    def start_cell_tracking(self):
        """Start cell tracking analysis using the real analysis module"""
        if not self.video_path:
            messagebox.showerror("Error", "Please load a video file first")
            return
            
        if not self.settings_path:
            messagebox.showerror("Error", "Please load or create settings first")
            return
        
        # Check if settings file exists
        if not os.path.exists(self.settings_path):
            messagebox.showerror("Error", f"Settings file not found: {self.settings_path}")
            return
            
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Starting cell tracking analysis...\n")
        self.results_text.update()
        
        self.status_var.set("Cell tracking analysis in progress...")
        self.progress_var.set(0)
        
        # Disable the start button to prevent multiple runs
        self.analysis_running = True
        
        # Run analysis in a separate thread to avoid blocking the GUI
        def run_analysis():
            try:
                if cell_tracking_analysis is not None:
                    # Run the real analysis
                    self.results_text.insert(tk.END, f"Using settings file: {self.settings_path}\n")
                    self.results_text.insert(tk.END, f"Processing video: {self.video_path}\n")
                    self.results_text.update()
                    
                    # Get analysis parameters from GUI
                    max_frames = int(self.max_frames_var.get()) if self.max_frames_var.get() else 0
                    target_fps = float(self.target_fps_var.get()) if self.target_fps_var.get() else None
                    
                    # Create a temporary sys.argv for the analysis module
                    original_argv = sys.argv.copy()
                    try:
                        sys.argv = ['cell_tracking_analysis.py', self.settings_path]
                        if max_frames > 0:
                            sys.argv.extend(['--max-frames', str(max_frames)])
                        if target_fps:
                            sys.argv.extend(['--target-fps', str(target_fps)])
                        
                        # Run the main analysis function
                        cell_tracking_analysis.main()
                        
                        # Restore original argv
                        sys.argv = original_argv
                        
                        # Update GUI on completion
                        self.root.after(0, lambda: self.analysis_completed(True, "Analysis completed successfully!"))
                        
                    except Exception as e:
                        # Restore original argv
                        sys.argv = original_argv
                        error_msg = f"Analysis failed: {str(e)}"
                        self.root.after(0, lambda: self.analysis_completed(False, error_msg))
                        
                else:
                    # Fall back to simulation if module not available
                    self.root.after(0, lambda: self.simulate_progress())
                    
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.root.after(0, lambda: self.analysis_completed(False, error_msg))
        
        # Start analysis in background thread
        threading.Thread(target=run_analysis, daemon=True).start()
    
    def analysis_completed(self, success, message):
        """Handle analysis completion"""
        self.analysis_running = False
        self.progress_var.set(100 if success else 0)
        
        self.results_text.insert(tk.END, f"\n{message}\n")
        
        if success:
            self.status_var.set("Cell tracking analysis completed successfully")
            
            # Look for output files
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            video_dir = os.path.dirname(self.video_path)
            
            output_files = []
            csv_files = []
            for ext in ['_raw.csv', '_kf.csv', '_analysis.mp4', '_visualization.png']:
                output_file = os.path.join(video_dir, f"{video_name}{ext}")
                if os.path.exists(output_file):
                    output_files.append(output_file)
                    if ext.endswith('.csv'):
                        csv_files.append(output_file)
            
            if output_files:
                self.results_text.insert(tk.END, "\nOutput files generated:\n")
                for file_path in output_files:
                    self.results_text.insert(tk.END, f"  - {os.path.basename(file_path)}\n")
            
            # Update dashboard status
            if csv_files and hasattr(self, 'dashboard_status_var'):
                self.dashboard_status_var.set("✅ Analysis complete! Click 'Open Dashboard' to explore results")
        else:
            self.status_var.set("Cell tracking analysis failed")
            if hasattr(self, 'dashboard_status_var'):
                self.dashboard_status_var.set("❌ Analysis failed - please check settings and try again")
        
        self.results_text.see(tk.END)
    
    def simulate_progress(self):
        """Fallback simulation when analysis module not available"""
        current = self.progress_var.get()
        if current < 100:
            self.progress_var.set(current + 10)
            self.root.after(200, self.simulate_progress)
        else:
            self.analysis_completed(True, "Analysis simulation completed!")
            
    def stop_cell_tracking(self):
        """Stop cell tracking analysis"""
        if hasattr(self, 'analysis_running') and self.analysis_running:
            self.status_var.set("Analysis stop requested (may take a moment)...")
            # Note: In a more advanced implementation, you would set a flag that the analysis thread checks
        else:
            self.status_var.set("Analysis stopped")
        self.progress_var.set(0)
        
    def run_cell_tracking(self):
        """Menu command for cell tracking"""
        self.notebook.select(4)  # Switch to tracking tab
        self.start_cell_tracking()
        
    # Dashboard methods
    def load_results(self):
        """Auto-detect results and open the real dashboard"""
        # Check if we have a settings file to work with
        if not self.settings_path:
            messagebox.showwarning("No Settings", 
                "Please load or create a settings file first.\n"
                "The dashboard needs a settings file to locate the corresponding CSV results.")
            return
        
        if not os.path.exists(self.settings_path):
            messagebox.showerror("Error", f"Settings file not found: {self.settings_path}")
            return
        
        # Check if CSV results exist
        video_name = os.path.splitext(os.path.basename(self.video_path))[0] if self.video_path else None
        if not video_name:
            # Try to infer from settings file name
            settings_name = os.path.basename(self.settings_path)
            video_name = settings_name.replace('_setting.txt', '')
        
        video_dir = os.path.dirname(self.settings_path)
        raw_csv = os.path.join(video_dir, f"{video_name}_raw.csv")
        kf_csv = os.path.join(video_dir, f"{video_name}_kf.csv")
        
        if not (os.path.exists(raw_csv) or os.path.exists(kf_csv)):
            messagebox.showwarning("No Results Found", 
                f"No CSV results found for '{video_name}'.\n"
                f"Please run cell tracking analysis first to generate:\n"
                f"• {video_name}_raw.csv\n"
                f"• {video_name}_kf.csv")
            return
        
        # Launch the real dashboard
        self.open_dashboard()
    
    def generate_dashboard(self):
        """Open the real roi_tracking_dashboard instead of generating inline"""
        self.open_dashboard()
    
    def open_dashboard(self):
        """Open the real roi_tracking_dashboard_v2.py as a separate process"""
        if not self.settings_path:
            messagebox.showwarning("No Settings", "Please load a settings file first.")
            return
            
        try:
            # Get the original matplotlib dashboard script path
            dashboard_script = os.path.join(os.path.dirname(__file__), 'roi_tracking_dashboard_v2.py')
            
            if not os.path.exists(dashboard_script):
                # Try current directory
                dashboard_script = 'roi_tracking_dashboard_v2.py'
                if not os.path.exists(dashboard_script):
                    messagebox.showerror("Error", 
                        "roi_tracking_dashboard_v2.py not found.\n"
                        "Please ensure the dashboard script is in the same directory.")
                    return
            
            # Launch dashboard in separate process
            cmd = [sys.executable, dashboard_script, self.settings_path]
            
            # Launch in background
            subprocess.Popen(cmd, 
                           cwd=os.path.dirname(os.path.abspath(self.settings_path)),
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            self.status_var.set("Dashboard opened in separate window")
            messagebox.showinfo("Dashboard Launched", 
                f"The interactive dashboard has been opened in a separate window.\n"
                f"Using settings file: {os.path.basename(self.settings_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open dashboard: {str(e)}")
    
    def export_data(self):
        """Export tracking data - redirect to dashboard"""
        messagebox.showinfo("Export Data", 
            "To export tracking data, please use the 'Load Results' button\n"
            "to open the interactive dashboard, which has full export functionality.")
        self.load_results()
            
    # Menu methods
    def show_scale_calibration(self):
        """Show scale calibration tab"""
        self.notebook.select(1)
        
    def show_parameter_tuning(self):
        """Show parameter tuning tab"""
        self.notebook.select(2)
        
    def show_roi_configuration(self):
        """Show ROI configuration tab"""
        self.notebook.select(3)
        
    def show_dashboard(self):
        """Show results dashboard tab"""
        self.notebook.select(5)
        
    def show_about(self):
        """Show about dialog"""
        about_text = """
Integrated Cell Analysis Application
Version 1.0

This application combines all aspects of the cell analysis pipeline:
- Scale calibration
- Parameter tuning
- ROI configuration  
- Cell tracking analysis
- Results visualization

Integrated from:
- ex_setting_v3.py
- roi_setting_gui_tk.py
- cell_tracking_analysis.py
- roi_tracking_dashboard_v2.py
        """
        messagebox.showinfo("About", about_text.strip())
    
    def on_tab_changed(self, _event):
        """Handle tab change events"""
        try:
            selected_tab = self.notebook.select()
            current_tab_index = self.notebook.index(selected_tab)
            
            # Auto-refresh images when switching to different tabs
            if self.current_frame is not None:
                if current_tab_index == 1:  # Scale Calibration tab
                    self.root.after(100, self.refresh_scale_image)
                elif current_tab_index == 2:  # Parameter Tuning tab
                    self.root.after(100, self.test_cell_detection)  # Show detection results
                elif current_tab_index == 3:  # ROI Configuration tab
                    self.root.after(100, self.refresh_roi_image)
                elif current_tab_index == 4:  # Cell Tracking tab
                    self.root.after(100, self.preview_detection)
        except:
            pass  # Ignore errors

def main():
    """Main function to run the application"""
    root = tk.Tk()
    IntegratedCellAnalysisApp(root)
    
    # Handle window closing
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()