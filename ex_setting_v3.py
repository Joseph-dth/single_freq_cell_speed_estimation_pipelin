import cv2
import numpy as np
import os
import math
import time
import subprocess
import sys

# --- Default Analysis Parameters ---
default_settings = {
    # Experiment Timing (in seconds)
    "period_seconds": 8.0,
    "repeat_count": 8,
    "idle_time_seconds": 9.0,

    # Image Processing Parameters
    "blur_kernel_size": 5,
    "threshold_value": 120,
    "min_cell_area": 25,
    "max_cell_area": 500,

    # Analysis Logic
    "line_x_tolerance_pixels": 1,
}

# --- Global state variables ---
app_state = {
    'video_path': 'A549_1.mp4',
    'current_step': 1,  # 1: scale, 2: parameters, 3: complete
    'scale_data': {
        'line_points': [],
        'input_text': "",
        'input_active': False,
        'scale_ratio': None
    },
    'tuner_data': {
        'current_frame_pos': 0,
        'buttons': [],
        'mouse_coords': (0, 0),
        'final_params': None,
        'idle_time_seconds': 8,
        'period_seconds': 8,
        'repeat_count': 8,
        'line_x_tolerance_pixels': 1,
        'p1_positions': None,  # {'left_x': x, 'right_x': x}
        'p2_positions': None,   # {'left_x': x, 'right_x': x}
        'is_playing': False,
        'play_fps': 30,  # Playback speed
        'last_frame_time': 0
    },
    'settings_input': {
        'idle_active': False,
        'period_active': False,
        'repeat_active': False, 
        'tolerance_active': False,
        'idle_text': '4.0',
        'period_text': '12.0',
        'repeat_text': '10',
        'tolerance_text': '20'
    },
    'running': True
}

def euclidean_dist(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# --- Clipboard Utilities ---
def get_clipboard_text():
    """Get text from clipboard (cross-platform)."""
    try:
        if sys.platform == "darwin":  # macOS
            result = subprocess.run(['pbpaste'], capture_output=True, text=True)
            return result.stdout if result.returncode == 0 else ""
        elif sys.platform == "win32":  # Windows
            result = subprocess.run(['powershell', '-command', 'Get-Clipboard'], capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else ""
        else:  # Linux
            try:
                result = subprocess.run(['xclip', '-selection', 'clipboard', '-o'], capture_output=True, text=True)
                return result.stdout if result.returncode == 0 else ""
            except FileNotFoundError:
                # Fallback to xsel
                result = subprocess.run(['xsel', '--clipboard', '--output'], capture_output=True, text=True)
                return result.stdout if result.returncode == 0 else ""
    except Exception:
        return ""

def set_clipboard_text(text):
    """Set text to clipboard (cross-platform)."""
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(['pbcopy'], input=text, text=True, check=True)
        elif sys.platform == "win32":  # Windows
            subprocess.run(['powershell', '-command', f'Set-Clipboard -Value "{text}"'], check=True)
        else:  # Linux
            try:
                subprocess.run(['xclip', '-selection', 'clipboard'], input=text, text=True, check=True)
            except FileNotFoundError:
                # Fallback to xsel
                subprocess.run(['xsel', '--clipboard', '--input'], input=text, text=True, check=True)
    except Exception:
        pass

# --- Cell Detection Functions ---
def find_cells_and_get_processed_image(frame, blur_k, thresh_val, erode_iter, min_area, max_area, min_circularity, min_inertia, max_inertia):
    """Process frame and detect cells using blob detection."""
    blur_k = int(blur_k) * 2 + 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed_image = cv2.erode(thresh, kernel_close, iterations=int(erode_iter))
    processed_image = cv2.bitwise_not(processed_image)

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 5
    params.maxThreshold = 255
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = True
    params.minCircularity = min_circularity / 100.0
    params.filterByConvexity = False
    params.filterByInertia = True
    params.minInertiaRatio = min_inertia / 100.0
    params.maxInertiaRatio = max_inertia / 100.0

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(processed_image)
    centers = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
    
    return sorted(centers, key=lambda c: c[0]), processed_image, gray

def detect_columns(centers, frame_width):
    """Detect the 3 vertical cell columns and return left, middle, right positions."""
    if len(centers) < 6:  # Reduced minimum requirement
        return None, None, None
    
    # Collect all x-coordinates
    x_coords = [x for x, y in centers]
    
    # Create histogram with more bins for better resolution
    bins = max(30, frame_width // 15)  # More granular bins
    hist, bin_edges = np.histogram(x_coords, bins=bins)
    
    # Find peaks (columns) - more sensitive peak detection
    peak_indices = []
    min_height = max(2, len(centers) // 10)  # Adaptive minimum height
    
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] >= min_height:
            peak_indices.append(i)
    
    # Sort by histogram value and take the strongest peaks
    peak_indices.sort(key=lambda i: hist[i], reverse=True)
    
    print(f"Column detection: Found {len(peak_indices)} peaks for {len(centers)} cells")
    
    if len(peak_indices) >= 3:
        # Convert bin indices to x-coordinates and sort left to right
        x_positions = [bin_edges[i] + (bin_edges[1] - bin_edges[0]) / 2 for i in peak_indices[:3]]
        x_positions.sort()
        left, middle, right = int(x_positions[0]), int(x_positions[1]), int(x_positions[2])
        print(f"3 columns detected: Left={left}, Middle={middle}, Right={right}")
        return left, middle, right
    elif len(peak_indices) >= 2:
        # Two columns detected - try to infer middle position
        x_positions = [bin_edges[i] + (bin_edges[1] - bin_edges[0]) / 2 for i in peak_indices[:2]]
        x_positions.sort()
        left, right = int(x_positions[0]), int(x_positions[1])
        # Estimate middle position
        middle = int((left + right) / 2)
        print(f"2 columns detected: Left={left}, Right={right}, Estimated Middle={middle}")
        return left, middle, right
    elif len(peak_indices) >= 1:
        # Only one strong peak - try to estimate other positions
        main_pos = int(bin_edges[peak_indices[0]] + (bin_edges[1] - bin_edges[0]) / 2)
        # Estimate positions based on typical spacing
        estimated_spacing = frame_width // 3
        if main_pos < frame_width // 3:  # Left column detected
            left = main_pos
            middle = left + estimated_spacing
            right = middle + estimated_spacing
        elif main_pos > 2 * frame_width // 3:  # Right column detected
            right = main_pos
            middle = right - estimated_spacing
            left = middle - estimated_spacing
        else:  # Middle column detected
            middle = main_pos
            left = middle - estimated_spacing
            right = middle + estimated_spacing
        print(f"1 column detected: Main={main_pos}, Estimated Left={left}, Middle={middle}, Right={right}")
        return int(left), int(middle), int(right)
    
    print("No columns detected reliably")
    return None, None, None

def draw_annotations(frame, centers):
    """Draw circles and coordinates on detected cells."""
    for center in centers:
        cv2.circle(frame, center, 8, (0, 255, 0), 2)
        coord_text = f"({center[0]},{center[1]})"
        cv2.putText(frame, coord_text, (center[0] + 15, center[1] + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return frame

# --- Position Setting GUI ---
def run_position_setting(position_type, frame, centers, frame_width, frame_height):
    """Interactive GUI for setting P1 or P2 positions by dragging column lines."""
    # Detect columns
    left_col, middle_col, right_col = detect_columns(centers, frame_width)
    
    if left_col is None or right_col is None:
        print(f"Error: Could not detect columns for {position_type} setting")
        return None
    
    # Initialize draggable vertical lines (only x-coordinates matter)
    # Always include middle line, estimate if not detected
    if middle_col is None:
        middle_col = (left_col + right_col) // 2  # Estimate middle position
        print(f"Middle column estimated at x={middle_col}")
    
    draggable_lines = {
        'left': {'x': left_col, 'dragging': False},
        'middle': {'x': middle_col, 'dragging': False},
        'right': {'x': right_col, 'dragging': False}
    }
    
    window_name = f"Set {position_type} Position - Drag column lines"
    cv2.namedWindow(window_name)
    
    # Create display frame with annotations
    display_frame = draw_annotations(frame.copy(), centers)
    
    def position_mouse_callback(event, x, y, flags, param):
        nonlocal draggable_lines
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near a line
            for line_id, line_data in draggable_lines.items():
                if abs(x - line_data['x']) < 20:  # 20 pixel tolerance
                    line_data['dragging'] = True
                    break
                    
        elif event == cv2.EVENT_LBUTTONUP:
            # Stop dragging all lines
            for line_data in draggable_lines.values():
                line_data['dragging'] = False
                
        elif event == cv2.EVENT_MOUSEMOVE:
            # Drag lines (only x-coordinate)
            for line_data in draggable_lines.values():
                if line_data['dragging']:
                    line_data['x'] = x
    
    cv2.setMouseCallback(window_name, position_mouse_callback)
    
    while True:
        # Create display
        display = display_frame.copy()
        
        # Draw column lines
        for line_id, line_data in draggable_lines.items():
            if line_id == 'left':
                color = (0, 255, 0)  # Green for left
                label = "LEFT"
            elif line_id == 'right':
                color = (0, 0, 255)  # Red for right
                label = "RIGHT"  
            else:  # middle
                color = (255, 255, 0)  # Yellow for middle (will be saved)
                label = "MIDDLE"
            
            # Draw vertical line (only x-coordinate matters)
            cv2.line(display, (line_data['x'], 0), (line_data['x'], frame_height), color, 2)
            
            # Draw label
            cv2.putText(display, f"{label} (x={line_data['x']})", 
                       (line_data['x'] + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw instructions
        cv2.putText(display, f"Setting {position_type} Position", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "Drag the vertical lines to set LEFT, MIDDLE, and RIGHT positions", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "All three positions will be saved to settings file", (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, "Press ENTER to confirm, ESC to cancel", (10, frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter - confirm
            positions = {
                'left_x': draggable_lines['left']['x'],
                'middle_x': draggable_lines['middle']['x'],  # Always include middle
                'right_x': draggable_lines['right']['x']
            }
            cv2.destroyWindow(window_name)
            return positions
            
        elif key == 27:  # Escape - cancel
            cv2.destroyWindow(window_name)
            return None
    
    cv2.destroyWindow(window_name)
    return None

# --- Step 1: Scale Calibration GUI (same as v2) ---
def run_scale_calibration():
    """GUI for setting scale with line drawing and text input."""
    video_path = app_state['video_path']
    
    # Get user video path via GUI input
    input_canvas = np.zeros((200, 600, 3), dtype=np.uint8)
    input_text = ""
    input_active = True
    
    cv2.namedWindow("Video Path Input")
    
    def input_mouse_callback(event, x, y, flags, param):
        nonlocal input_active
        input_active = 20 <= x <= 580 and 80 <= y <= 120
    
    cv2.setMouseCallback("Video Path Input", input_mouse_callback)
    
    while True:
        input_canvas[:] = (50, 50, 50)
        cv2.putText(input_canvas, "Enter video file path (or press Enter for default):", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(input_canvas, f"Default: {app_state['video_path']}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(input_canvas, "Tip: Cmd+V paste clipboard, Cmd+A clear, Backspace/Delete edit", 
                   (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        box_color = (255, 255, 255) if input_active else (150, 150, 150)
        cv2.rectangle(input_canvas, (20, 80), (580, 120), box_color, 2)
        cv2.putText(input_canvas, input_text, (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(input_canvas, "Press Enter to continue, Esc to quit", 
                   (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Video Path Input", input_canvas)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # Esc
            cv2.destroyAllWindows()
            return None
        elif key == 13:  # Enter
            if input_text.strip():
                video_path = input_text.strip()
                app_state['video_path'] = video_path
            break
        elif input_active:
            if key == 8 or key == 127:  # Backspace or Delete (macOS)
                if len(input_text) > 0:
                    input_text = input_text[:-1]
            elif key == 1:  # Ctrl+A - Select all (clear text for easy replacement)
                input_text = ""
            elif key == 3 or key == 99:  # Ctrl+C or alternative - Copy text to clipboard
                if input_text:
                    set_clipboard_text(input_text)
                    print(f"Copied to clipboard: '{input_text}'")
            elif key == 22 or key == 118:  # Ctrl+V or alternative - Paste from clipboard  
                clipboard_text = get_clipboard_text().strip()
                if clipboard_text:
                    input_text = clipboard_text
                    print(f"Pasted from clipboard: '{input_text}'")
            elif 32 <= key <= 126:  # Printable characters
                input_text += chr(key)
    
    cv2.destroyAllWindows()
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return None

    # Load middle frame from video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return None
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read frame from video")
        return None

    clone = frame.copy()
    pristine_frame = frame.copy()
    h, w, _ = frame.shape
    control_h = 120
    canvas = np.zeros((h + control_h, w, 3), dtype=np.uint8)
    
    # GUI elements
    input_box = (20, h + 50, 200, 30)
    buttons = [
        (240, h + 50, 100, 30, "Set Scale"),
        (360, h + 50, 80, 30, "Reset"),
        (w - 100, h + 50, 80, 30, "Next")
    ]
    
    window_name = "Step 1: Set Scale Calibration"
    cv2.namedWindow(window_name)
    
    def scale_mouse_callback(event, x, y, flags, param):
        nonlocal clone, pristine_frame
        scale_data = app_state['scale_data']
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Handle buttons
            for bx, by, bw, bh, text in buttons:
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    if text == "Set Scale":
                        if len(scale_data['line_points']) == 2 and scale_data['input_text']:
                            try:
                                actual_len = float(scale_data['input_text'])
                                p1, p2 = scale_data['line_points']
                                px_len = euclidean_dist(p1, p2)
                                if px_len > 0:
                                    scale_data['scale_ratio'] = actual_len / px_len
                                    print(f"Scale set: {scale_data['scale_ratio']:.4f} microns/pixel")
                            except ValueError:
                                print("Invalid number in input box.")
                        else:
                            print("Please draw a line and enter length first.")
                    elif text == "Reset":
                        scale_data['line_points'] = []
                        scale_data['input_text'] = ""
                        clone = pristine_frame.copy()
                    elif text == "Next":
                        if scale_data['scale_ratio'] is not None:
                            app_state['current_step'] = 2
                            return
                        else:
                            print("Please set scale first.")
                    return
            
            # Handle input box
            ib_x, ib_y, ib_w, ib_h = input_box
            scale_data['input_active'] = ib_x <= x <= ib_x + ib_w and ib_y <= y <= ib_y + ib_h
            
            # Handle line drawing on image area
            if len(scale_data['line_points']) < 2 and y < h:
                scale_data['line_points'].append((x, y))
                cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(clone, f"Pt{len(scale_data['line_points'])}", 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if len(scale_data['line_points']) == 2:
                    cv2.line(clone, scale_data['line_points'][0], scale_data['line_points'][1], (255, 0, 0), 2)
    
    cv2.setMouseCallback(window_name, scale_mouse_callback)
    
    while app_state['current_step'] == 1 and app_state['running']:
        scale_data = app_state['scale_data']
        
        # Update canvas
        canvas[:h, :] = clone
        canvas[h:, :] = (50, 50, 50)
        
        # Instructions
        cv2.putText(canvas, "1. Click two points to draw a line", (20, h + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, "2. Click input box and enter actual length in microns", (20, h + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw buttons
        for bx, by, bw, bh, text in buttons:
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
            cv2.putText(canvas, text, (bx + 10, by + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw input box
        ib_x, ib_y, ib_w, ib_h = input_box
        box_color = (255, 255, 255) if scale_data['input_active'] else (150, 150, 150)
        cv2.rectangle(canvas, (ib_x, ib_y), (ib_x + ib_w, ib_y + ib_h), box_color, 2)
        cv2.putText(canvas, scale_data['input_text'], (ib_x + 5, ib_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show scale if set
        if scale_data['scale_ratio'] is not None:
            cv2.putText(canvas, f"Scale: {scale_data['scale_ratio']:.4f} microns/pixel", 
                       (460, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # Esc
            app_state['running'] = False
            break
        elif scale_data['input_active']:
            if key == 8 or key == 127:  # Backspace or Delete (macOS)
                if len(scale_data['input_text']) > 0:
                    scale_data['input_text'] = scale_data['input_text'][:-1]
            elif key == 1:  # Ctrl+A - Select all (clear text for easy replacement)
                scale_data['input_text'] = ""
            elif key == 3:  # Ctrl+C - Copy text to clipboard
                if scale_data['input_text']:
                    set_clipboard_text(scale_data['input_text'])
                    print(f"Copied to clipboard: '{scale_data['input_text']}'")
            elif key == 22:  # Ctrl+V - Paste from clipboard
                clipboard_text = get_clipboard_text().strip()
                if clipboard_text and (clipboard_text.replace('.', '').isdigit() or clipboard_text.isdigit()):
                    scale_data['input_text'] = clipboard_text
                    print(f"Pasted from clipboard: '{clipboard_text}'")
            elif key == 13:  # Enter - trigger Set Scale
                pass  # Will be handled by clicking Set Scale button
            elif 32 <= key <= 126:  # Printable characters
                char = chr(key)
                if char.isdigit() or (char == '.' and '.' not in scale_data['input_text']):
                    scale_data['input_text'] += char
    
    cv2.destroyAllWindows()
    return app_state['scale_data']['scale_ratio']

# --- Step 2: Parameter Tuning GUI with Position Setting ---
def run_parameter_tuner():
    """Interactive GUI for parameter tuning and position selection."""
    video_path = app_state['video_path']
    tuner_data = app_state['tuner_data']
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Window setup
    display_area_width = frame_width * 2
    button_panel_width = 250
    input_panel_width = 150  # Extra space for input boxes
    main_window_width = display_area_width + button_panel_width + input_panel_width

    main_window_name = "Step 2: Parameter Tuning & Position Selection"
    controls_window_name = "Controls"
    cv2.namedWindow(main_window_name)
    cv2.namedWindow(controls_window_name)

    def tuner_mouse_callback(event, x, y, flags, param):
        tuner_data['mouse_coords'] = (x, y)
        settings_input = app_state['settings_input']
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Handle buttons
            for (bx, by, bw, bh, text, action) in tuner_data['buttons']:
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    action()
                    tuner_data['current_frame_pos'] = max(0, min(tuner_data['current_frame_pos'], total_frames - 1))
                    cv2.setTrackbarPos("Frame", controls_window_name, tuner_data['current_frame_pos'])
                    break
            
            # Handle settings input boxes with proper spacing
            idle_box = (settings_panel_x, settings_panel_y, input_box_width, input_box_height)
            period_box = (settings_panel_x, settings_panel_y + label_spacing, input_box_width, input_box_height)
            repeat_box = (settings_panel_x, settings_panel_y + label_spacing * 2, input_box_width, input_box_height) 
            tolerance_box = (settings_panel_x, settings_panel_y + label_spacing * 3, input_box_width, input_box_height)
            
            settings_input['idle_active'] = (idle_box[0] <= x <= idle_box[0] + idle_box[2] and 
                                            idle_box[1] <= y <= idle_box[1] + idle_box[3])
            settings_input['period_active'] = (period_box[0] <= x <= period_box[0] + period_box[2] and 
                                              period_box[1] <= y <= period_box[1] + period_box[3])
            settings_input['repeat_active'] = (repeat_box[0] <= x <= repeat_box[0] + repeat_box[2] and 
                                              repeat_box[1] <= y <= repeat_box[1] + repeat_box[3])
            settings_input['tolerance_active'] = (tolerance_box[0] <= x <= tolerance_box[0] + tolerance_box[2] and 
                                                  tolerance_box[1] <= y <= tolerance_box[1] + tolerance_box[3])

    cv2.setMouseCallback(main_window_name, tuner_mouse_callback)

    # Create trackbars
    def nothing(x): pass
    cv2.createTrackbar("Frame", controls_window_name, 0, total_frames - 1, nothing)
    cv2.createTrackbar("Blur Kernel", controls_window_name, 2, 10, nothing)
    cv2.createTrackbar("Threshold", controls_window_name, 110, 255, nothing)
    cv2.createTrackbar("Erode Iterations", controls_window_name, 2, 10, nothing)
    cv2.createTrackbar("Min Area", controls_window_name, 25, 1000, nothing)
    cv2.createTrackbar("Max Area", controls_window_name, 500, 5000, nothing)
    cv2.createTrackbar("Min Circularity", controls_window_name, 60, 100, nothing)
    cv2.createTrackbar("Min Inertia", controls_window_name, 1, 100, nothing)
    cv2.createTrackbar("Max Inertia", controls_window_name, 100, 100, nothing)
    
    # Settings input boxes coordinates - position to the right of buttons
    settings_panel_x = display_area_width + 240  # Move further right to avoid buttons
    settings_panel_y = 70  # Start higher up with more space
    input_box_width = 120
    input_box_height = 30  # Taller input boxes
    label_spacing = 50  # More space between input sections

    # Button actions
    def action_play_pause():
        tuner_data['is_playing'] = not tuner_data['is_playing']
        tuner_data['last_frame_time'] = time.time()
        print(f"Video {'playing' if tuner_data['is_playing'] else 'paused'}")
    
    def action_next_frame(): 
        tuner_data['is_playing'] = False
        tuner_data['current_frame_pos'] += 1
    def action_prev_frame(): 
        tuner_data['is_playing'] = False
        tuner_data['current_frame_pos'] -= 1
    def action_next_10_frames(): 
        tuner_data['is_playing'] = False
        tuner_data['current_frame_pos'] += 10
    def action_prev_10_frames(): 
        tuner_data['is_playing'] = False
        tuner_data['current_frame_pos'] -= 10
    def action_next_50_frames(): 
        tuner_data['is_playing'] = False
        tuner_data['current_frame_pos'] += 50
    def action_prev_50_frames(): 
        tuner_data['is_playing'] = False
        tuner_data['current_frame_pos'] -= 50
    def action_next_sec(): 
        tuner_data['is_playing'] = False
        tuner_data['current_frame_pos'] += int(fps)
    def action_prev_sec(): 
        tuner_data['is_playing'] = False
        tuner_data['current_frame_pos'] -= int(fps)
    
    def action_set_p1_position():
        # Get current frame and process it
        cap.set(cv2.CAP_PROP_POS_FRAMES, tuner_data['current_frame_pos'])
        ret, frame = cap.read()
        if ret:
            # Get current parameters
            blur_k = cv2.getTrackbarPos("Blur Kernel", controls_window_name)
            thresh_val = cv2.getTrackbarPos("Threshold", controls_window_name)
            erode_iter = cv2.getTrackbarPos("Erode Iterations", controls_window_name)
            min_area = cv2.getTrackbarPos("Min Area", controls_window_name)
            max_area = cv2.getTrackbarPos("Max Area", controls_window_name)
            min_circ = cv2.getTrackbarPos("Min Circularity", controls_window_name)
            min_inertia = cv2.getTrackbarPos("Min Inertia", controls_window_name)
            max_inertia = cv2.getTrackbarPos("Max Inertia", controls_window_name)
            
            centers, _, _ = find_cells_and_get_processed_image(
                frame, blur_k, thresh_val, erode_iter, min_area, max_area, min_circ, min_inertia, max_inertia)
            
            positions = run_position_setting("P1", frame, centers, frame_width, frame_height)
            if positions:
                tuner_data['p1_positions'] = positions
                print(f"P1 positions set: Left(x={positions['left_x']}), Middle(x={positions['middle_x']}), Right(x={positions['right_x']})")
    
    def action_set_p2_position():
        # Get current frame and process it
        cap.set(cv2.CAP_PROP_POS_FRAMES, tuner_data['current_frame_pos'])
        ret, frame = cap.read()
        if ret:
            # Get current parameters
            blur_k = cv2.getTrackbarPos("Blur Kernel", controls_window_name)
            thresh_val = cv2.getTrackbarPos("Threshold", controls_window_name)
            erode_iter = cv2.getTrackbarPos("Erode Iterations", controls_window_name)
            min_area = cv2.getTrackbarPos("Min Area", controls_window_name)
            max_area = cv2.getTrackbarPos("Max Area", controls_window_name)
            min_circ = cv2.getTrackbarPos("Min Circularity", controls_window_name)
            min_inertia = cv2.getTrackbarPos("Min Inertia", controls_window_name)
            max_inertia = cv2.getTrackbarPos("Max Inertia", controls_window_name)
            
            centers, _, _ = find_cells_and_get_processed_image(
                frame, blur_k, thresh_val, erode_iter, min_area, max_area, min_circ, min_inertia, max_inertia)
            
            positions = run_position_setting("P2", frame, centers, frame_width, frame_height)
            if positions:
                tuner_data['p2_positions'] = positions
                print(f"P2 positions set: Left(x={positions['left_x']}), Middle(x={positions['middle_x']}), Right(x={positions['right_x']})")
    
    def action_generate_file():
        if app_state['scale_data']['scale_ratio'] is not None:
            if tuner_data['p1_positions'] and tuner_data['p2_positions']:
                app_state['current_step'] = 3
            else:
                print("Please set both P1 and P2 positions first.")
        else:
            print("Scale not set. Please complete step 1 first.")

    # Define button layout
    btn_x = display_area_width + 20
    btn_y, btn_w, btn_h, btn_margin = 20, 210, 35, 8
    button_layout = [
        ("PLAY/PAUSE (SPACE)", action_play_pause),
        ("Prev Sec (-1s)", action_prev_sec),
        ("Next Sec (+1s)", action_next_sec),
        ("Prev 50 Frames", action_prev_50_frames),
        ("Next 50 Frames", action_next_50_frames),
        ("Prev 10 Frames", action_prev_10_frames),
        ("Next 10 Frames", action_next_10_frames),
        ("Prev Frame (-1)", action_prev_frame),
        ("Next Frame (+1)", action_next_frame),
        ("SET P1 POSITION", action_set_p1_position),
        ("SET P2 POSITION", action_set_p2_position),
        ("GENERATE FILE", action_generate_file),
    ]
    
    tuner_data['buttons'] = []
    for i, (text, action) in enumerate(button_layout):
        y_pos = btn_y + i * (btn_h + btn_margin)
        tuner_data['buttons'].append((btn_x, y_pos, btn_w, btn_h, text, action))

    while app_state['current_step'] == 2 and app_state['running']:
        # Handle video playback
        if tuner_data['is_playing']:
            current_time = time.time()
            if current_time - tuner_data['last_frame_time'] > (1.0 / tuner_data['play_fps']):
                tuner_data['current_frame_pos'] += 1
                if tuner_data['current_frame_pos'] >= total_frames:
                    tuner_data['current_frame_pos'] = total_frames - 1
                    tuner_data['is_playing'] = False
                tuner_data['last_frame_time'] = current_time
                cv2.setTrackbarPos("Frame", controls_window_name, tuner_data['current_frame_pos'])
        
        # Read trackbar values
        blur_k = cv2.getTrackbarPos("Blur Kernel", controls_window_name)
        thresh_val = cv2.getTrackbarPos("Threshold", controls_window_name)
        erode_iter = cv2.getTrackbarPos("Erode Iterations", controls_window_name)
        min_area = cv2.getTrackbarPos("Min Area", controls_window_name)
        max_area = cv2.getTrackbarPos("Max Area", controls_window_name)
        min_circ = cv2.getTrackbarPos("Min Circularity", controls_window_name)
        min_inertia = cv2.getTrackbarPos("Min Inertia", controls_window_name)
        max_inertia = cv2.getTrackbarPos("Max Inertia", controls_window_name)
        
        # Update frame position from trackbar (unless playing)
        if not tuner_data['is_playing']:
            tuner_data['current_frame_pos'] = cv2.getTrackbarPos("Frame", controls_window_name)

        cap.set(cv2.CAP_PROP_POS_FRAMES, tuner_data['current_frame_pos'])
        ret, original_frame = cap.read()
        if not ret: break

        # Process image
        centers, processed_img, gray_img = find_cells_and_get_processed_image(
            original_frame, blur_k, thresh_val, erode_iter, min_area, max_area, min_circ, min_inertia, max_inertia)
        
        # Create display
        display_original = draw_annotations(original_frame.copy(), centers)
        display_processed = draw_annotations(cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR), centers)

        # Assemble main window
        main_canvas = np.zeros((frame_height, main_window_width, 3), dtype=np.uint8)
        main_canvas[:frame_height, :frame_width] = display_original
        main_canvas[:frame_height, frame_width:display_area_width] = display_processed

        # Draw info
        timestamp = tuner_data['current_frame_pos'] / fps
        playback_status = "PLAYING" if tuner_data['is_playing'] else "PAUSED"
        info_text = f"Time: {timestamp:.2f}s | Frame: {tuner_data['current_frame_pos']} | {playback_status}"
        cv2.putText(main_canvas, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        count_text = f"Cells Detected: {len(centers)}"
        cv2.putText(main_canvas, count_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show position status
        p1_status = "P1: SET" if tuner_data['p1_positions'] else "P1: NOT SET"
        p2_status = "P2: SET" if tuner_data['p2_positions'] else "P2: NOT SET"
        cv2.putText(main_canvas, f"{p1_status} | {p2_status}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw settings input panel
        settings_input = app_state['settings_input']
        
        # Idle time input
        idle_box = (settings_panel_x, settings_panel_y, input_box_width, input_box_height)
        cv2.putText(main_canvas, "Idle Time (s):", (settings_panel_x, settings_panel_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        box_color = (255, 255, 255) if settings_input['idle_active'] else (200, 200, 200)
        # Fill box background to clear previous text
        cv2.rectangle(main_canvas, idle_box[:2], (idle_box[0] + idle_box[2], idle_box[1] + idle_box[3]), (30, 30, 30), -1)
        cv2.rectangle(main_canvas, idle_box[:2], (idle_box[0] + idle_box[2], idle_box[1] + idle_box[3]), box_color, 2)
        # Show text or cursor
        display_text = settings_input['idle_text'] if settings_input['idle_text'] else "4.0"
        text_color = (255, 255, 255) if settings_input['idle_text'] else (180, 180, 180)
        cv2.putText(main_canvas, display_text, (idle_box[0] + 3, idle_box[1] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Period input
        period_box = (settings_panel_x, settings_panel_y + label_spacing, input_box_width, input_box_height)
        cv2.putText(main_canvas, "Period (s):", (settings_panel_x, settings_panel_y + label_spacing - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        box_color = (255, 255, 255) if settings_input['period_active'] else (200, 200, 200)
        # Fill box background to clear previous text
        cv2.rectangle(main_canvas, period_box[:2], (period_box[0] + period_box[2], period_box[1] + period_box[3]), (30, 30, 30), -1)
        cv2.rectangle(main_canvas, period_box[:2], (period_box[0] + period_box[2], period_box[1] + period_box[3]), box_color, 2)
        # Show text or cursor
        display_text = settings_input['period_text'] if settings_input['period_text'] else "12.0"
        text_color = (255, 255, 255) if settings_input['period_text'] else (180, 180, 180)
        cv2.putText(main_canvas, display_text, (period_box[0] + 3, period_box[1] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Repeat count input
        repeat_box = (settings_panel_x, settings_panel_y + label_spacing * 2, input_box_width, input_box_height)
        cv2.putText(main_canvas, "Repeat Count:", (settings_panel_x, settings_panel_y + label_spacing * 2 - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        box_color = (255, 255, 255) if settings_input['repeat_active'] else (200, 200, 200)
        # Fill box background to clear previous text
        cv2.rectangle(main_canvas, repeat_box[:2], (repeat_box[0] + repeat_box[2], repeat_box[1] + repeat_box[3]), (30, 30, 30), -1)
        cv2.rectangle(main_canvas, repeat_box[:2], (repeat_box[0] + repeat_box[2], repeat_box[1] + repeat_box[3]), box_color, 2)
        # Show text or cursor  
        display_text = settings_input['repeat_text'] if settings_input['repeat_text'] else "10"
        text_color = (255, 255, 255) if settings_input['repeat_text'] else (180, 180, 180)
        cv2.putText(main_canvas, display_text, (repeat_box[0] + 3, repeat_box[1] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Tolerance input
        tolerance_box = (settings_panel_x, settings_panel_y + label_spacing * 3, input_box_width, input_box_height)
        cv2.putText(main_canvas, "X Tolerance (px):", (settings_panel_x, settings_panel_y + label_spacing * 3 - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        box_color = (255, 255, 255) if settings_input['tolerance_active'] else (200, 200, 200)
        # Fill box background to clear previous text
        cv2.rectangle(main_canvas, tolerance_box[:2], (tolerance_box[0] + tolerance_box[2], tolerance_box[1] + tolerance_box[3]), (30, 30, 30), -1)
        cv2.rectangle(main_canvas, tolerance_box[:2], (tolerance_box[0] + tolerance_box[2], tolerance_box[1] + tolerance_box[3]), box_color, 2)
        # Show text or cursor
        display_text = settings_input['tolerance_text'] if settings_input['tolerance_text'] else "20"
        text_color = (255, 255, 255) if settings_input['tolerance_text'] else (180, 180, 180)
        cv2.putText(main_canvas, display_text, (tolerance_box[0] + 3, tolerance_box[1] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        mx, my = tuner_data['mouse_coords']
        if 0 <= my < frame_height:
            if 0 <= mx < frame_width:
                pixel_val = original_frame[my, mx]
                cursor_text = f"Cursor: ({mx}, {my}) | Val: {pixel_val}"
                cv2.putText(main_canvas, cursor_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

        # Draw buttons
        for (bx, by, bw, bh, text, _) in tuner_data['buttons']:
            if text in ["SET P1 POSITION", "SET P2 POSITION", "GENERATE FILE"]:
                color = (100, 150, 100)
            elif text == "PLAY/PAUSE (SPACE)":
                color = (0, 150, 0) if tuner_data['is_playing'] else (150, 100, 0)  # Green when playing, orange when paused
            else:
                color = (80, 80, 80)
            
            cv2.rectangle(main_canvas, (bx, by), (bx + bw, by + bh), color, -1)
            cv2.rectangle(main_canvas, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = bx + (bw - text_size[0]) // 2
            text_y = by + (bh + text_size[1]) // 2
            cv2.putText(main_canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Update settings from input boxes
        try:
            tuner_data['idle_time_seconds'] = float(settings_input['idle_text'])
            tuner_data['period_seconds'] = float(settings_input['period_text'])
            tuner_data['repeat_count'] = int(settings_input['repeat_text'])
            tuner_data['line_x_tolerance_pixels'] = int(settings_input['tolerance_text'])
        except ValueError:
            pass  # Keep existing values if invalid input
        
        # Store final parameters
        tuner_data['final_params'] = {
            'blur_kernel_size': blur_k * 2 + 1,
            'threshold_value': thresh_val,
            'erode_iterations': erode_iter,
            'min_cell_area': min_area,
            'max_cell_area': max_area,
            'min_circularity': min_circ / 100.0,
            'min_inertia': min_inertia / 100.0,
            'max_inertia': max_inertia / 100.0,
            'idle_time_seconds': tuner_data['idle_time_seconds'],
            'period_seconds': tuner_data['period_seconds'],
            'repeat_count': tuner_data['repeat_count'],
            'line_x_tolerance_pixels': tuner_data['line_x_tolerance_pixels']
        }

        # Create dummy control canvas
        controls_canvas = np.zeros((1, 500, 3), dtype=np.uint8)

        cv2.imshow(main_window_name, main_canvas)
        cv2.imshow(controls_window_name, controls_canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or cv2.getWindowProperty(main_window_name, cv2.WND_PROP_VISIBLE) < 1:
            app_state['running'] = False
        elif key == 32:  # Spacebar - play/pause
            action_play_pause()
        elif key != 255:  # Handle text input for settings boxes
            settings_input = app_state['settings_input']
            # Debug print to see what keys are being pressed
            print(f"Key pressed: {key} ({'delete' if key == 8 or key == 127 else chr(key) if 32 <= key <= 126 else 'special'})")
            
            if settings_input['idle_active']:
                if key == 8 or key == 127:  # Backspace or Delete (macOS)
                    if len(settings_input['idle_text']) > 0:
                        settings_input['idle_text'] = settings_input['idle_text'][:-1]
                        print(f"Idle text now: '{settings_input['idle_text']}'")
                elif key == 1:  # Ctrl+A - Select all (clear text)
                    settings_input['idle_text'] = ""
                elif key == 22:  # Ctrl+V - Paste from clipboard
                    clipboard_text = get_clipboard_text().strip()
                    if clipboard_text and (clipboard_text.replace('.', '').isdigit() or clipboard_text.isdigit()):
                        settings_input['idle_text'] = clipboard_text
                        print(f"Pasted idle text: '{clipboard_text}'")
                elif 32 <= key <= 126:  # Printable characters
                    char = chr(key)
                    if char.isdigit() or (char == '.' and '.' not in settings_input['idle_text']):
                        settings_input['idle_text'] += char
                        print(f"Idle text now: '{settings_input['idle_text']}'")
            elif settings_input['period_active']:
                if key == 8 or key == 127:  # Backspace or Delete (macOS)
                    if len(settings_input['period_text']) > 0:
                        settings_input['period_text'] = settings_input['period_text'][:-1]
                        print(f"Period text now: '{settings_input['period_text']}'")
                elif key == 1:  # Ctrl+A - Select all (clear text)
                    settings_input['period_text'] = ""
                elif key == 22:  # Ctrl+V - Paste from clipboard
                    clipboard_text = get_clipboard_text().strip()
                    if clipboard_text and (clipboard_text.replace('.', '').isdigit() or clipboard_text.isdigit()):
                        settings_input['period_text'] = clipboard_text
                        print(f"Pasted period text: '{clipboard_text}'")
                elif 32 <= key <= 126:  # Printable characters
                    char = chr(key)
                    if char.isdigit() or (char == '.' and '.' not in settings_input['period_text']):
                        settings_input['period_text'] += char
                        print(f"Period text now: '{settings_input['period_text']}'")
            elif settings_input['repeat_active']:
                if key == 8 or key == 127:  # Backspace or Delete (macOS)
                    if len(settings_input['repeat_text']) > 0:
                        settings_input['repeat_text'] = settings_input['repeat_text'][:-1]
                        print(f"Repeat text now: '{settings_input['repeat_text']}'")
                elif key == 1:  # Ctrl+A - Select all (clear text)
                    settings_input['repeat_text'] = ""
                elif key == 22:  # Ctrl+V - Paste from clipboard
                    clipboard_text = get_clipboard_text().strip()
                    if clipboard_text and clipboard_text.isdigit():
                        settings_input['repeat_text'] = clipboard_text
                        print(f"Pasted repeat text: '{clipboard_text}'")
                elif 32 <= key <= 126:  # Printable characters
                    char = chr(key)
                    if char.isdigit():
                        settings_input['repeat_text'] += char
                        print(f"Repeat text now: '{settings_input['repeat_text']}'")
            elif settings_input['tolerance_active']:
                if key == 8 or key == 127:  # Backspace or Delete (macOS)
                    if len(settings_input['tolerance_text']) > 0:
                        settings_input['tolerance_text'] = settings_input['tolerance_text'][:-1]
                        print(f"Tolerance text now: '{settings_input['tolerance_text']}'")
                elif key == 1:  # Ctrl+A - Select all (clear text)
                    settings_input['tolerance_text'] = ""
                elif key == 22:  # Ctrl+V - Paste from clipboard
                    clipboard_text = get_clipboard_text().strip()
                    if clipboard_text and clipboard_text.isdigit():
                        settings_input['tolerance_text'] = clipboard_text
                        print(f"Pasted tolerance text: '{clipboard_text}'")
                elif 32 <= key <= 126:  # Printable characters
                    char = chr(key)
                    if char.isdigit():
                        settings_input['tolerance_text'] += char
                        print(f"Tolerance text now: '{settings_input['tolerance_text']}'")
            
            # Show which input is active
            active_inputs = [k for k, v in settings_input.items() if k.endswith('_active') and v]
            print(f"Active inputs: {active_inputs}")

    cap.release()
    cv2.destroyAllWindows()
    return tuner_data['final_params']

# --- Step 3: File Generation ---
def generate_settings_file():
    """Generate the settings file with all collected parameters."""
    video_path = app_state['video_path']
    scale_ratio = app_state['scale_data']['scale_ratio']
    params = app_state['tuner_data']['final_params']
    p1_pos = app_state['tuner_data']['p1_positions']
    p2_pos = app_state['tuner_data']['p2_positions']
    
    if scale_ratio is None or params is None or p1_pos is None or p2_pos is None:
        print("Missing required data. Cannot generate settings file.")
        return False

    video_basename = os.path.basename(video_path)
    output_filename = os.path.splitext(video_basename)[0] + "_setting.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_filename)

    with open(output_path, "w") as f:
        f.write("# Experiment Timing (in seconds)\n")
        f.write(f"period_seconds: {default_settings['period_seconds']}\n")
        f.write(f"repeat_count: {default_settings['repeat_count']}\n")
        f.write(f"idle_time_seconds: {params['idle_time_seconds']:.4f}\n\n")

        f.write("# Position Coordinates (manually set - x-coordinates only)\n")
        f.write(f"left_p1_x: {p1_pos['left_x']}\n")
        f.write(f"left_p2_x: {p2_pos['left_x']}\n")
        f.write(f"middle_p1_x: {p1_pos['middle_x']}\n")
        f.write(f"middle_p2_x: {p2_pos['middle_x']}\n")
        f.write(f"right_p1_x: {p1_pos['right_x']}\n")
        f.write(f"right_p2_x: {p2_pos['right_x']}\n\n")

        f.write("# Scaling & Measurement\n")
        f.write("# This value is generated by the scale calibration step\n")
        f.write(f"scale_microns_per_pixel: {scale_ratio:.4f}\n\n")

        f.write("# Image Processing Parameters\n")
        f.write(f"blur_kernel_size: {params['blur_kernel_size']}\n")
        f.write(f"threshold_value: {params['threshold_value']}\n")
        f.write(f"erode_iterations: {params['erode_iterations']}\n")
        f.write(f"min_cell_area: {params['min_cell_area']}\n")
        f.write(f"max_cell_area: {params['max_cell_area']}\n")
        f.write(f"min_circularity: {params['min_circularity']:.2f}\n")
        f.write(f"min_inertia_ratio: {params['min_inertia']:.2f}\n")
        f.write(f"max_inertia_ratio: {params['max_inertia']:.2f}\n\n")

        f.write("# Analysis Logic\n")
        f.write(f"line_x_tolerance_pixels: {default_settings['line_x_tolerance_pixels']}\n")

    print(f"\n✅ Settings file saved to: {output_path}")
    return True

# --- Main Application Flow ---
def main():
    """Main function to run the complete GUI application."""
    print("=== Cell Analysis Settings Generator v3 ===")
    print("This tool allows manual positioning and configurable experiment parameters.")
    print()

    # Step 1: Scale Calibration
    print("Step 1: Scale Calibration")
    scale_ratio = run_scale_calibration()
    
    if not app_state['running'] or scale_ratio is None:
        print("Scale calibration cancelled.")
        return

    # Step 2: Parameter Tuning
    print("\nStep 2: Parameter Tuning & Position Selection")
    final_params = run_parameter_tuner()
    
    if not app_state['running'] or app_state['current_step'] != 3:
        print("Parameter tuning cancelled.")
        return

    # Step 3: Generate Settings File
    print("\nStep 3: Generating Settings File")
    success = generate_settings_file()
    
    if success:
        print("\n🎉 Settings file generation complete!")
        print(f"Video: {app_state['video_path']}")
        print(f"Scale: {scale_ratio:.4f} microns/pixel")
        p1_pos = app_state['tuner_data']['p1_positions']
        p2_pos = app_state['tuner_data']['p2_positions']
        print(f"P1 Positions - Left: x={p1_pos['left_x']}, Middle: x={p1_pos['middle_x']}, Right: x={p1_pos['right_x']}")
        print(f"P2 Positions - Left: x={p2_pos['left_x']}, Middle: x={p2_pos['middle_x']}, Right: x={p2_pos['right_x']}")
    else:
        print("Failed to generate settings file.")

if __name__ == "__main__":
    main()