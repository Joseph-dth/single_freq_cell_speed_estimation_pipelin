#!/usr/bin/env python3
"""
Cell Tracking Analysis Tool

This script takes a settings.txt file and analyzes cell movement throughout a video.
Outputs:
1. CSV file recording each cell position in each frame
2. Visualization showing cell paths over time with P1/P2 reference lines
3. Annotated MP4 video showing detected cells

Usage: python cell_tracking_analysis.py <settings_file.txt>
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import argparse
from pathlib import Path
import time
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from typing import Dict, List, Tuple, Optional


class KalmanTracker:
    """Individual Kalman filter tracker for a single cell."""
    
    count = 0
    
    def __init__(self, bbox, process_noise_scale: float = 1.0, measurement_noise_scale: float = 1.0):
        """Initialize Kalman filter tracker with initial bounding box [x, y, w, h].
        Args:
            bbox: [x, y, w, h] or [x, y] center
            process_noise_scale: multiplier for process noise Q
            measurement_noise_scale: multiplier for measurement noise R
        """
        # Convert center point to bounding box format for compatibility
        if len(bbox) == 2:  # [x, y] center point
            x, y = bbox
            bbox = [x-10, y-10, 20, 20]  # Convert to [x, y, w, h] with default size
        
        # Initialize Kalman filter with 7 state variables [x, y, s, r, vx, vy, vs]
        # x, y: center position, s: scale (area), r: aspect ratio, vx, vy, vs: velocities
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0], 
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0  # Higher uncertainty for scale and aspect ratio
        self.kf.R *= float(measurement_noise_scale)
        
        # Process noise
        self.kf.Q[-1, -1] *= 0.01   # Scale velocity noise
        self.kf.Q[4:, 4:] *= 0.01   # Velocity noise
        self.kf.Q *= float(process_noise_scale)
        
        # Covariance matrix
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty for initial velocities
        self.kf.P *= 10.0
        
        # Initialize state
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox):
        """Update tracker with new detection."""
        if len(bbox) == 2:  # Convert center point to bbox
            x, y = bbox
            bbox = [x-10, y-10, 20, 20]
            
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        
    def predict(self):
        """Predict next state and return predicted bounding box."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]
        
    def get_state(self):
        """Return current state as bounding box."""
        return self.convert_x_to_bbox(self.kf.x)
        
    def get_center(self):
        """Return current center position."""
        bbox = self.get_state()
        return [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
    
    @staticmethod
    def convert_bbox_to_z(bbox):
        """Convert bounding box [x, y, w, h] to measurement vector [x, y, s, r]."""
        w, h = bbox[2], bbox[3]
        x = bbox[0] + w/2.0
        y = bbox[1] + h/2.0
        s = w * h  # Scale (area)
        r = w / float(h)  # Aspect ratio
        return np.array([x, y, s, r]).reshape((4, 1))
        
    @staticmethod  
    def convert_x_to_bbox(x):
        """Convert state vector to bounding box [x, y, w, h]."""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([x[0] - w/2., x[1] - h/2., w, h]).flatten()


class MultiObjectTracker:
    """Multi-object tracker using SORT algorithm."""
    
    def __init__(self, max_disappeared=5, max_distance=50, process_noise_scale: float = 1.0, measurement_noise_scale: float = 1.0):
        """Initialize tracker.
        
        Args:
            max_disappeared: Maximum frames a tracker can be unmatched before deletion
            max_distance: Maximum distance for matching detections to trackers
            process_noise_scale: Q scale for internal Kalman filters
            measurement_noise_scale: R scale for internal Kalman filters
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale
        self.trackers = []
        self.frame_count = 0
        
    def update(self, detections):
        """Update trackers with new detections.
        
        Args:
            detections: List of [x, y] center points
            
        Returns:
            List of tracked objects with their IDs and positions
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        to_del = []
        ret = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        # Remove invalid trackers
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            detections, self.trackers, self.max_distance)
            
        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]])
            
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanTracker(detections[i], self.process_noise_scale, self.measurement_noise_scale)
            self.trackers.append(trk)
            
        # Return results
        for trk in self.trackers:
            if (trk.time_since_update < 1) and (trk.hit_streak >= 1 or self.frame_count <= 1):
                center = trk.get_center()
                ret.append([center[0], center[1], trk.id])
                
        # Remove dead trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.time_since_update > self.max_disappeared:
                self.trackers.pop(i-1)
            i -= 1
            
        return ret
        
    @staticmethod
    def associate_detections_to_trackers(detections, trackers, distance_threshold):
        """Associate detections to trackers using Hungarian algorithm."""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
            
        # Compute distance matrix
        distance_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                trk_center = trk.get_center()
                distance_matrix[d, t] = np.sqrt((det[0] - trk_center[0])**2 + (det[1] - trk_center[1])**2)
                
        # Use Hungarian algorithm for assignment
        if distance_matrix.size > 0:
            matched_indices = linear_sum_assignment(distance_matrix)
            matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))
        else:
            matched_indices = np.empty(shape=(0, 2))
            
        # Filter out matches that are too far
        matches = []
        for m in matched_indices:
            if distance_matrix[m[0], m[1]] > distance_threshold:
                continue
            matches.append(m.reshape(1, 2))
            
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
            
        # Find unmatched detections and trackers
        unmatched_detections = []
        for d, det in enumerate(detections):
            if len(matches) == 0 or d not in matches[:, 0]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if len(matches) == 0 or t not in matches[:, 1]:
                unmatched_trackers.append(t)
                
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class SimpleTracker:
    """A simple nearest-neighbor multi-object tracker without Kalman filtering."""
    def __init__(self, max_disappeared: int = 5, max_distance: float = 50.0,
                 reid_max_gap_frames: int = 60, reid_max_distance: float = 60.0):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.reid_max_gap_frames = reid_max_gap_frames
        self.reid_max_distance = reid_max_distance
        self.next_id = 0
        # Active tracks: track_id -> {"centroid": (x,y), "disappeared": int, "last_frame": int, "history": List[(frame,timestamp,x,y)]}
        self.tracks: Dict[int, Dict] = {}
        # Recently retired tracks for re-identification: track_id -> {"centroid": (x,y), "last_frame": int}
        self.retired: Dict[int, Dict] = {}

    def update(self, detections: List[List[float]], frame_num: int, timestamp: float) -> List[List[float]]:
        # Return list of [x, y, track_id] for current frame (assigned using nearest neighbor)
        if len(self.tracks) == 0:
            for det in detections:
                self._create_track_with_id(None, det[0], det[1], frame_num, timestamp)
            # Return with the created ids in order of detections
            assigned = []
            for det in detections:
                # Find the closest newly created track in this frame (exact position match)
                for tid, tr in self.tracks.items():
                    if tr.get("last_frame") == frame_num and tr["centroid"] == (det[0], det[1]) and tid not in [a[2] for a in assigned]:
                        assigned.append([det[0], det[1], tid])
                        break
            return assigned

        # Build cost matrix between detections and existing tracks
        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid]["centroid"] for tid in track_ids], dtype=np.float32)
        dets = np.array(detections, dtype=np.float32) if detections else np.zeros((0, 2), dtype=np.float32)
        if dets.shape[0] == 0:
            # Increment disappeared for all
            for tid in track_ids:
                self.tracks[tid]["disappeared"] += 1
                self.tracks[tid]["last_frame"] = frame_num
            # Remove disappeared beyond max
            for tid in list(track_ids):
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    # move to retired for potential re-identification
                    self.retired[tid] = {
                        "centroid": self.tracks[tid]["centroid"],
                        "last_frame": self.tracks[tid]["last_frame"]
                    }
                    del self.tracks[tid]
            return []

        # Compute distance matrix
        dist = np.linalg.norm(dets[:, None, :] - track_centroids[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(dist)

        used_dets = set()
        used_trks = set()
        assignments = []
        for r, c in zip(rows, cols):
            if dist[r, c] <= self.max_distance:
                tid = track_ids[c]
                x, y = float(dets[r, 0]), float(dets[r, 1])
                self.tracks[tid]["centroid"] = (x, y)
                self.tracks[tid]["disappeared"] = 0
                self.tracks[tid]["last_frame"] = frame_num
                self.tracks[tid]["history"].append((frame_num, timestamp, x, y))
                assignments.append([x, y, tid])
                used_dets.add(r)
                used_trks.add(c)

        # Unmatched tracks: increment disappeared, remove if needed
        for idx, tid in enumerate(track_ids):
            if idx not in used_trks:
                self.tracks[tid]["disappeared"] += 1
                self.tracks[tid]["last_frame"] = frame_num
        for tid in list(track_ids):
            if self.tracks[tid]["disappeared"] > self.max_disappeared:
                self.retired[tid] = {
                    "centroid": self.tracks[tid]["centroid"],
                    "last_frame": self.tracks[tid]["last_frame"]
                }
                del self.tracks[tid]

        # Unmatched detections: create new tracks (with RE-ID check)
        for i in range(dets.shape[0]):
            if i in used_dets:
                continue
            x, y = float(dets[i, 0]), float(dets[i, 1])
            revived_id = self._try_reidentify(x, y, frame_num)
            tid_use = self._create_track_with_id(revived_id, x, y, frame_num, timestamp)
            assignments.append([x, y, tid_use])

        return assignments

    def _create_track_with_id(self, specific_id: Optional[int], x: float, y: float, frame_num: int, timestamp: float) -> int:
        if specific_id is not None:
            tid = int(specific_id)
            # If somehow still present in active, reuse; else create
            self.tracks[tid] = {
                "centroid": (x, y),
                "disappeared": 0,
                "last_frame": frame_num,
                "history": [(frame_num, timestamp, x, y)]
            }
            if tid in self.retired:
                del self.retired[tid]
            return tid
        tid = self.next_id
        self.tracks[tid] = {
            "centroid": (x, y),
            "disappeared": 0,
            "last_frame": frame_num,
            "history": [(frame_num, timestamp, x, y)]
        }
        self.next_id += 1
        return tid

    def _try_reidentify(self, x: float, y: float, frame_num: int) -> Optional[int]:
        if not self.retired:
            return None
        best_id = None
        best_dist = float('inf')
        for tid, info in list(self.retired.items()):
            gap = frame_num - info["last_frame"]
            if gap < 0 or gap > self.reid_max_gap_frames:
                # too old, discard
                del self.retired[tid]
                continue
            dx = x - info["centroid"][0]
            dy = y - info["centroid"][1]
            d = (dx*dx + dy*dy) ** 0.5
            if d <= self.reid_max_distance and d < best_dist:
                best_dist = d
                best_id = tid
        if best_id is not None:
            # reuse this id
            del self.retired[best_id]
            return best_id
        return None


class CellTracker:
    def __init__(self, settings_file):
        """Initialize cell tracker with settings from file."""
        self.settings_file = Path(settings_file)
        self.settings = {}
        
        # Output paths (same directory as settings file)
        self.output_dir = self.settings_file.parent
        self.base_name = self.settings_file.stem.replace('_setting', '')
        
        # Results storage
        self.frame_data = []
        self.cell_trajectories = {}
        
        # Initialize multi-object tracker
        self.tracker = MultiObjectTracker(max_disappeared=10, max_distance=75)
        
        # Load settings after initializing base attributes
        self.load_settings()
        
        # ROI support
        self.rois: Optional[Dict[str, Tuple[int, int, int, int]]] = self._load_rois_from_settings()
        # Per-ROI trackers and trajectories
        self.trackers_per_roi: Dict[str, MultiObjectTracker] = {}
        self.simple_trackers_per_roi: Dict[str, SimpleTracker] = {}
        self.kf_trajectories: Dict[str, Dict[int, List[dict]]] = {k: {} for k in (self.rois.keys() if self.rois else ['left', 'middle', 'right'])}
        self.raw_trajectories: Dict[str, Dict[int, List[dict]]] = {k: {} for k in (self.rois.keys() if self.rois else ['left', 'middle', 'right'])}
        if self.rois:
            for roi_name in self.rois.keys():
                self.trackers_per_roi[roi_name] = MultiObjectTracker(
                    max_disappeared=int(self.settings.get('max_disappeared', 10)),
                    max_distance=float(self.settings.get('max_distance', 75)),
                    process_noise_scale=float(self.settings.get('kf_process_noise_scale', 1.0)),
                    measurement_noise_scale=float(self.settings.get('kf_measurement_noise_scale', 1.0))
                )
                self.simple_trackers_per_roi[roi_name] = SimpleTracker(
                    max_disappeared=int(self.settings.get('max_disappeared', 10)),
                    max_distance=float(self.settings.get('max_distance', 75)),
                    reid_max_gap_frames=int(self.settings.get('reid_max_gap_frames', 60)),
                    reid_max_distance=float(self.settings.get('reid_max_distance', 60.0))
                )

    def _load_rois_from_settings(self) -> Optional[Dict[str, Tuple[int, int, int, int]]]:
        """Load ROI rectangles from settings if present.
        Returns dict: {'left': (x1,y1,x2,y2), 'middle': (...), 'right': (...)} or None
        """
        labels = ['left', 'middle', 'right']
        rois = {}
        for label in labels:
            x1 = self.settings.get(f'{label}_roi_x1')
            y1 = self.settings.get(f'{label}_roi_y1')
            x2 = self.settings.get(f'{label}_roi_x2')
            y2 = self.settings.get(f'{label}_roi_y2')
            if all(v is not None for v in [x1, y1, x2, y2]):
                rois[label] = (int(x1), int(y1), int(x2), int(y2))
        return rois if len(rois) == 3 else None

    @staticmethod
    def _point_in_rect(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = rect
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return x1 <= x <= x2 and y1 <= y <= y2
        
    def load_settings(self):
        """Load settings from the settings file."""
        print(f"Loading settings from: {self.settings_file}")
        
        if not self.settings_file.exists():
            raise FileNotFoundError(f"Settings file not found: {self.settings_file}")
            
        with open(self.settings_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Convert to appropriate type
                        try:
                            if '.' in value:
                                self.settings[key] = float(value)
                            else:
                                self.settings[key] = int(value)
                        except ValueError:
                            self.settings[key] = value
        
        print(f"Loaded {len(self.settings)} settings")
        
        # Find video file (assume it's in the same directory)
        video_name = self.base_name + '.mp4'
        video_path = self.output_dir / video_name
        if not video_path.exists():
            # Try common video extensions
            for ext in ['.avi', '.mov', '.mkv']:
                alt_path = self.output_dir / (self.base_name + ext)
                if alt_path.exists():
                    video_path = alt_path
                    break
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        self.video_path = video_path
        print(f"Using video: {self.video_path}")
        
    def detect_cells_in_frame(self, frame):
        """Detect cells in a single frame using the loaded parameters."""
        # Get parameters from settings
        blur_kernel = self.settings.get('blur_kernel_size', 5)
        threshold_val = self.settings.get('threshold_value', 120)
        erode_iter = self.settings.get('erode_iterations', 2)
        min_area = self.settings.get('min_cell_area', self.settings.get('min_blob_area', 25))
        max_area = self.settings.get('max_cell_area', self.settings.get('max_blob_area', 500))
        min_circularity = self.settings.get('min_circularity', 0.6)
        min_inertia = self.settings.get('min_inertia_ratio', 0.01)
        max_inertia = self.settings.get('max_inertia_ratio', 1.0)
        roi_crop_detection = bool(self.settings.get('roi_crop_detection', 1))
        
        def detect_on_gray(gray_img):
            blurred = cv2.GaussianBlur(gray_img, (blur_kernel, blur_kernel), 0)
            _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            processed_local = cv2.erode(thresh, kernel, iterations=erode_iter)
            processed_local = cv2.bitwise_not(processed_local)
            params = cv2.SimpleBlobDetector_Params()
            params.minThreshold = 5
            params.maxThreshold = 255
            params.blobColor = 255
            params.filterByArea = True
            params.minArea = min_area
            params.maxArea = max_area
            params.filterByCircularity = True
            params.minCircularity = min_circularity
            params.filterByConvexity = False
            params.filterByInertia = True
            params.minInertiaRatio = min_inertia
            params.maxInertiaRatio = max_inertia
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(processed_local)
            centers_local = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
            return centers_local, processed_local
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If ROIs exist and cropping is enabled, run detection per ROI crop for higher specificity
        if self.rois and roi_crop_detection:
            centers = []
            processed_vis = np.zeros_like(gray)
            for rect in self.rois.values():
                x1, y1, x2, y2 = rect
                x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
                y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
                x_min = max(0, x_min); y_min = max(0, y_min)
                x_max = min(gray.shape[1]-1, x_max); y_max = min(gray.shape[0]-1, y_max)
                crop = gray[y_min:y_max+1, x_min:x_max+1]
                roi_centers, processed_local = detect_on_gray(crop)
                # Shift back to full-image coords
                for cx, cy in roi_centers:
                    centers.append((cx + x_min, cy + y_min))
                # For visualization aggregation
                processed_vis[y_min:y_max+1, x_min:x_max+1] = np.maximum(
                    processed_vis[y_min:y_max+1, x_min:x_max+1], processed_local
                )
            centers = sorted(centers, key=lambda c: c[0])
            return centers, processed_vis
        
        # Fallback: global detection then filter by ROI if defined
        centers, processed = detect_on_gray(gray)
        if self.rois:
            filtered = []
            for x, y in centers:
                for rect in self.rois.values():
                    if self._point_in_rect(x, y, rect):
                        filtered.append((x, y))
                        break
            centers = filtered
        return sorted(centers, key=lambda c: c[0]), processed
    
    def categorize_cells(self, centers):
        """Categorize cells into left, middle, right columns or ROIs."""
        if not centers:
            return {'left': [], 'middle': [], 'right': []}
        
        # If ROI rectangles are present, use them
        if self.rois:
            categorized = {k: [] for k in self.rois.keys()}
            for c in centers:
                x, y = c
                for label, rect in self.rois.items():
                    if self._point_in_rect(x, y, rect):
                        categorized[label].append(c)
                        break
            # Ensure all labels exist
            for k in ['left', 'middle', 'right']:
                if k not in categorized:
                    categorized[k] = []
            return categorized
        
        # Fallback: Get position boundaries from settings (P1/P2)
        left_p1_x = self.settings.get('left_p1_x', 0)
        left_p2_x = self.settings.get('left_p2_x', 0)
        right_p1_x = self.settings.get('right_p1_x', 0)
        right_p2_x = self.settings.get('right_p2_x', 0)
        tolerance = self.settings.get('line_x_tolerance_pixels', 1)
        
        # Define column boundaries (with some margin)
        left_boundary = max(left_p1_x, left_p2_x) + tolerance
        right_boundary = min(right_p1_x, right_p2_x) - tolerance
        
        categorized = {'left': [], 'middle': [], 'right': []}
        
        for center in centers:
            x, y = center
            if x < left_boundary:
                categorized['left'].append(center)
            elif x > right_boundary:
                categorized['right'].append(center)
            else:
                categorized['middle'].append(center)
        
        return categorized
    
    def analyze_video(self):
        """Analyze the entire video and track cell movements."""
        print(f"Analyzing video: {self.video_path}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
        
        frame_num = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect cells in current frame
            centers, processed_frame = self.detect_cells_in_frame(frame)
            
            # Categorize cells
            categorized_cells = self.categorize_cells(centers)
            
            # Calculate timestamp
            timestamp = frame_num / fps
            
            # Store frame data (global)
            frame_data = {
                'frame_number': frame_num,
                'timestamp': timestamp,
                'all_positions': centers,
                'categorized_positions': categorized_cells
            }
            
            self.frame_data.append(frame_data)
            
            # Track per ROI
            self.track_cells_by_roi(categorized_cells, frame_num, timestamp)
            
            # Progress update
            if frame_num % 100 == 0:
                elapsed = time.time() - start_time
                progress = (frame_num + 1) / total_frames
                eta = elapsed / progress - elapsed if progress > 0 else 0
                print(f"Processing frame {frame_num + 1}/{total_frames} "
                      f"({progress*100:.1f}%) - ETA: {eta:.1f}s")
            
            frame_num += 1
        
        cap.release()
        print(f"Analysis complete! Processed {frame_num} frames")
    
    def track_cells_by_roi(self, categorized_centers: Dict[str, List[Tuple[int, int]]], frame_num: int, timestamp: float):
        """Track cells separately for each ROI using both simple and KF trackers."""
        roi_labels = list(categorized_centers.keys())
        for roi in roi_labels:
            detections = [[x, y] for (x, y) in categorized_centers.get(roi, [])]
            # Initialize per-ROI trackers lazily if ROIs not defined in settings
            if roi not in self.trackers_per_roi:
                self.trackers_per_roi[roi] = MultiObjectTracker(
                    max_disappeared=int(self.settings.get('max_disappeared', 10)),
                    max_distance=float(self.settings.get('max_distance', 75)),
                    process_noise_scale=float(self.settings.get('kf_process_noise_scale', 1.0)),
                    measurement_noise_scale=float(self.settings.get('kf_measurement_noise_scale', 1.0))
                )
            if roi not in self.simple_trackers_per_roi:
                self.simple_trackers_per_roi[roi] = SimpleTracker(
                    max_disappeared=int(self.settings.get('max_disappeared', 10)),
                    max_distance=float(self.settings.get('max_distance', 75)),
                    reid_max_gap_frames=int(self.settings.get('reid_max_gap_frames', 60)),
                    reid_max_distance=float(self.settings.get('reid_max_distance', 60.0))
                )

            # Simple tracker (raw)
            raw_objs = self.simple_trackers_per_roi[roi].update(detections, frame_num, timestamp)
            for x, y, track_id in raw_objs:
                tid = int(track_id)
                if tid not in self.raw_trajectories[roi]:
                    self.raw_trajectories[roi][tid] = []
                self.raw_trajectories[roi][tid].append({
                    'frame': frame_num,
                    'timestamp': timestamp,
                    'x': x,
                    'y': y
                })

            # Kalman tracker (KF)
            kf_objs = self.trackers_per_roi[roi].update(detections)
            for x, y, track_id in kf_objs:
                tid = int(track_id)
                if tid not in self.kf_trajectories[roi]:
                    self.kf_trajectories[roi][tid] = []
                self.kf_trajectories[roi][tid].append({
                    'frame': frame_num,
                    'timestamp': timestamp,
                    'x': x,
                    'y': y
                })
    
    def save_raw_csv_results(self) -> Path:
        """Save raw (simple tracker) trajectories to CSV with ROI column."""
        print("Saving RAW CSV results...")
        rows = []
        for roi, tracks in self.raw_trajectories.items():
            for tid, traj in tracks.items():
                for p in traj:
                    rows.append({
                        'roi': roi,
                        'frame_number': p['frame'],
                        'timestamp': p['timestamp'],
                        'cell_id': tid,
                        'cell_x': p['x'],
                        'cell_y': p['y']
                    })
        rows.sort(key=lambda r: (r['frame_number'], r['roi'], r['cell_id']))
        df = pd.DataFrame(rows)
        csv_path = self.output_dir / f"{self.base_name}_raw.csv"
        df.to_csv(csv_path, index=False)
        print(f"RAW CSV saved to: {csv_path}")
        return csv_path

    def save_kf_csv_results(self) -> Path:
        """Save Kalman-filtered trajectories to CSV with ROI column."""
        print("Saving KF CSV results...")
        rows = []
        for roi, tracks in self.kf_trajectories.items():
            for tid, traj in tracks.items():
                for p in traj:
                    rows.append({
                        'roi': roi,
                        'frame_number': p['frame'],
                        'timestamp': p['timestamp'],
                        'cell_id': tid,
                        'cell_x': p['x'],
                        'cell_y': p['y']
                    })
        rows.sort(key=lambda r: (r['frame_number'], r['roi'], r['cell_id']))
        df = pd.DataFrame(rows)
        csv_path = self.output_dir / f"{self.base_name}_kf.csv"
        df.to_csv(csv_path, index=False)
        print(f"KF CSV saved to: {csv_path}")
        return csv_path
    
    def save_csv_results(self):
        """Save detailed frame-by-frame results to CSV."""
        print("Saving CSV results...")
        
        # Prepare data for CSV
        csv_data = []
        
        for frame_data in self.frame_data:
            # Handle different frame_data formats (full analysis vs limited analysis)
            if 'total_cells' in frame_data:
                # Full analysis format
                base_row = {
                    'frame_number': frame_data['frame_number'],
                    'timestamp': frame_data['timestamp'],
                    'total_cells': frame_data['total_cells'],
                    'left_cells': frame_data['left_cells'],
                    'middle_cells': frame_data['middle_cells'],
                    'right_cells': frame_data['right_cells']
                }
                all_positions = frame_data['all_positions']
                categorized = frame_data['categorized_positions']
            else:
                # Limited analysis format - single cell entry
                base_row = {
                    'frame_number': frame_data['frame_number'],
                    'timestamp': frame_data['timestamp'],
                    'total_cells': 1,  # Single cell entry
                    'left_cells': 1 if frame_data['column'] == 'left' else 0,
                    'middle_cells': 1 if frame_data['column'] == 'middle' else 0,
                    'right_cells': 1 if frame_data['column'] == 'right' else 0
                }
                # Create compatible structure
                all_positions = [(frame_data['cell_x'], frame_data['cell_y'])]
                categorized = {(frame_data['cell_x'], frame_data['cell_y']): frame_data['column']}
            
            # If no cells, add one row with the base data
            if not all_positions:
                csv_data.append(base_row)
            else:
                # Add each cell as a separate row
                for i, (x, y) in enumerate(all_positions):
                    row = base_row.copy()
                    row.update({
                        'cell_id': i,
                        'cell_x': x,
                        'cell_y': y,
                        'column': self.get_cell_column(x, y, categorized)
                    })
                    csv_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / f"{self.base_name}_analysis.csv"
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")
        
        return csv_path
    
    def get_cell_column(self, x, y, categorized):
        """Determine which column a cell belongs to."""
        for column, positions in categorized.items():
            if (x, y) in positions:
                return column
        return 'unknown'
    
    def save_csv_results_from_trajectories(self):
        """Save CSV results directly from cell trajectories (for limited analysis)."""
        print("Saving CSV results from trajectories...")
        
        csv_data = []
        
        # Add individual cell data from trajectories
        for track_id, trajectory in self.cell_trajectories.items():
            for point in trajectory:
                csv_data.append({
                    'frame_number': point['frame'],
                    'timestamp': point['timestamp'],
                    'cell_id': track_id,
                    'cell_x': point['x'],
                    'cell_y': point['y'],
                    'column': self.classify_cell_position(point['x'])
                })
        
        # Sort by frame number, then by cell_id
        csv_data.sort(key=lambda x: (x['frame_number'], x['cell_id']))
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / f"{self.base_name}_analysis.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"CSV saved: {csv_path}")
        print(f"Total rows: {len(csv_data)}")
        return csv_path
    
    def classify_cell_position(self, x):
        """Classify cell position into left, middle, right based on x coordinate."""
        # Use P1/P2 boundaries from settings
        left_p1_x = self.settings.get('left_p1_x', 0)
        left_p2_x = self.settings.get('left_p2_x', 0)
        right_p1_x = self.settings.get('right_p1_x', 0)
        right_p2_x = self.settings.get('right_p2_x', 0)
        tolerance = self.settings.get('line_x_tolerance_pixels', 1)
        
        left_boundary = max(left_p1_x, left_p2_x) + tolerance
        right_boundary = min(right_p1_x, right_p2_x) - tolerance
        
        if x < left_boundary:
            return 'left'
        elif x > right_boundary:
            return 'right'
        else:
            return 'middle'
    
    def create_visualization(self):
        """Create visualization showing cell paths over time with P1/P2 lines."""
        print("Creating trajectory visualization...")
        
        # Get P1/P2 positions
        left_p1_x = self.settings.get('left_p1_x', 0)
        left_p2_x = self.settings.get('left_p2_x', 0)
        right_p1_x = self.settings.get('right_p1_x', 0)
        right_p2_x = self.settings.get('right_p2_x', 0)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot cell trajectories
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.cell_trajectories)))
        
        for traj_id, (traj, color) in enumerate(zip(self.cell_trajectories.values(), colors)):
            if len(traj) < 5:  # Skip very short trajectories
                continue
                
            timestamps = [point['timestamp'] for point in traj]
            x_positions = [point['x'] for point in traj]
            
            plt.plot(timestamps, x_positions, color=color, alpha=0.7, linewidth=1,
                    label=f'Cell {traj_id}' if traj_id < 10 else None)
        
        # Add P1 and P2 reference lines
        max_time = max([frame['timestamp'] for frame in self.frame_data]) if self.frame_data else 1
        
        plt.axhline(y=left_p1_x, color='red', linestyle='--', linewidth=2, label='Left P1')
        plt.axhline(y=left_p2_x, color='red', linestyle=':', linewidth=2, label='Left P2')
        plt.axhline(y=right_p1_x, color='blue', linestyle='--', linewidth=2, label='Right P1')
        plt.axhline(y=right_p2_x, color='blue', linestyle=':', linewidth=2, label='Right P2')
        
        # Formatting
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('X Position (pixels)', fontsize=12)
        plt.title(f'Cell Movement Analysis - {self.base_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save plot
        plot_path = self.output_dir / f"{self.base_name}_analysis_visualization.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {plot_path}")
        return plot_path
    
    def create_annotated_video(self):
        """Create annotated video showing detected cells."""
        print("Creating annotated video...")
        
        # Open input video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        output_path = self.output_dir / f"{self.base_name}_analysis_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Get P1/P2 positions for reference lines
        left_p1_x = self.settings.get('left_p1_x', 0)
        left_p2_x = self.settings.get('left_p2_x', 0)
        right_p1_x = self.settings.get('right_p1_x', 0)
        right_p2_x = self.settings.get('right_p2_x', 0)
        
        frame_num = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get frame data
            if frame_num < len(self.frame_data):
                frame_data = self.frame_data[frame_num]
                centers = frame_data['all_positions']
                categorized = frame_data['categorized_positions']
                
                # Draw detected cells
                for center in centers:
                    cv2.circle(frame, center, 8, (0, 255, 0), 2)
                    coord_text = f"({center[0]},{center[1]})"
                    cv2.putText(frame, coord_text, (center[0] + 15, center[1] + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Draw P1/P2 reference lines
                cv2.line(frame, (left_p1_x, 0), (left_p1_x, height), (0, 0, 255), 2)
                cv2.line(frame, (left_p2_x, 0), (left_p2_x, height), (0, 0, 255), 1)
                cv2.line(frame, (right_p1_x, 0), (right_p1_x, height), (255, 0, 0), 2)
                cv2.line(frame, (right_p2_x, 0), (right_p2_x, height), (255, 0, 0), 1)
                
                # Add labels for reference lines
                cv2.putText(frame, "Left P1", (left_p1_x + 5, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "Left P2", (left_p2_x + 5, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.putText(frame, "Right P1", (right_p1_x + 5, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, "Right P2", (right_p2_x + 5, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                
                # Add frame info
                timestamp = frame_data['timestamp']
                info_text = f"Frame: {frame_num} | Time: {timestamp:.2f}s | Cells: {len(centers)}"
                cv2.putText(frame, info_text, (10, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add cell count by column
                count_info = f"Left: {len(categorized['left'])} | Middle: {len(categorized['middle'])} | Right: {len(categorized['right'])}"
                cv2.putText(frame, count_info, (10, height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out.write(frame)
            
            # Progress update
            if frame_num % 100 == 0:
                elapsed = time.time() - start_time
                progress = (frame_num + 1) / total_frames
                eta = elapsed / progress - elapsed if progress > 0 else 0
                print(f"Processing frame {frame_num + 1}/{total_frames} "
                      f"({progress*100:.1f}%) - ETA: {eta:.1f}s")
            
            frame_num += 1
        
        cap.release()
        out.release()
        
        print(f"Annotated video saved to: {output_path}")
        return output_path
    
    def run_analysis(self, target_fps=None):
        """Run complete analysis pipeline."""
        print(f"Starting cell tracking analysis for: {self.base_name}")
        print(f"Output directory: {self.output_dir}")
        if target_fps:
            print(f"Target analysis FPS: {target_fps}")
        
        # Step 1: Analyze video
        if target_fps:
            self.analyze_video_with_fps(target_fps)
        else:
            self.analyze_video()
        
        # Step 2a: Save RAW CSV results (per ROI, simple tracker)
        raw_csv_path = self.save_raw_csv_results()
        
        # Step 2b: Save KF CSV results (per ROI, KF tracker)
        kf_csv_path = self.save_kf_csv_results()
        
        # Optional legacy CSV and visualization
        # csv_path = self.save_csv_results()
        # plot_path = self.create_visualization()
        
        # Step 3: Create annotated video (optional)
        # video_path = self.create_annotated_video()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print(f"Input settings: {self.settings_file}")
        print(f"Input video: {self.video_path}")
        print(f"RAW CSV results: {raw_csv_path}")
        print(f"KF CSV results: {kf_csv_path}")
        print("="*50)
    
    def run_analysis_limited(self, max_frames, target_fps=None):
        """Run limited analysis for testing (first N frames only)."""
        print("="*50)
        print(f"CELL TRACKING TEST (FIRST {max_frames} FRAMES)")
        print("="*50)
        
        try:
            # Run limited video analysis
            if target_fps:
                self.analyze_video_limited_with_fps(max_frames, target_fps)
            else:
                self.analyze_video_limited(max_frames)
            
            # Save RAW and KF CSV results
            raw_csv_path = self.save_raw_csv_results()
            kf_csv_path = self.save_kf_csv_results()
            
            print(f"\n✅ Test analysis completed! Processed {max_frames} frames")
            print(f"Created RAW trajectories (ROIs): {[len(v) for v in self.raw_trajectories.values()]}")
            print(f"RAW CSV: {raw_csv_path}")
            print(f"KF CSV: {kf_csv_path}")
            
            # Show trajectory summary (KF)
            total_kf_tracks = sum(len(tracks) for tracks in self.kf_trajectories.values())
            print(f"KF tracks: {total_kf_tracks}")
            for roi_label, tracks in self.kf_trajectories.items():
                for track_id, trajectory in tracks.items():
                    if len(trajectory) >= 5:  # Show only significant trajectories
                        start = trajectory[0]
                        end = trajectory[-1]
                        print(f"[{roi_label}] Track {track_id}: {len(trajectory)} points, ({start['x']:.0f},{start['y']:.0f}) → ({end['x']:.0f},{end['y']:.0f})")
            
        except Exception as e:
            print(f"\n❌ Test analysis failed: {e}")
            raise
    
    def analyze_video_limited(self, max_frames):
        """Analyze limited number of frames from video."""
        print(f"Analyzing first {max_frames} frames: {self.video_path}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {total_frames} total frames, {fps:.2f} FPS")
        print(f"Processing: {min(max_frames, total_frames)} frames")
        
        frame_num = 0
        start_time = time.time()
        
        while frame_num < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate timestamp
            timestamp = frame_num / fps
            
            # Detect cells in current frame
            centers, processed_frame = self.detect_cells_in_frame(frame)
            
            # Categorize cells by column/ROI
            categorized_cells = self.categorize_cells(centers)
            
            # Track cells per ROI
            self.track_cells_by_roi(categorized_cells, frame_num, timestamp)
            
            # Store frame data (raw positions)
            self.frame_data.append({
                'frame_number': frame_num,
                'timestamp': timestamp,
                'all_positions': centers,
                'categorized_positions': categorized_cells
            })
            
            # Progress update
            if (frame_num + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (frame_num + 1) / elapsed
                print(f"Processed {frame_num + 1}/{max_frames} frames ({rate:.1f} fps) - {len(centers)} cells detected")
            
            frame_num += 1
        
        cap.release()
        print(f"Limited analysis complete! Processed {frame_num} frames")
    
    def analyze_video_with_fps(self, target_fps):
        """Analyze video with frame rate sampling for faster processing."""
        print(f"Analyzing video with target FPS: {target_fps}")
        print(f"Video path: {self.video_path}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps
        
        # Calculate frame skip interval
        frame_skip = max(1, int(original_fps / target_fps))
        estimated_frames = total_frames // frame_skip
        
        print(f"Original video: {total_frames} frames at {original_fps:.2f} FPS ({duration:.2f}s)")
        print(f"Analysis: Every {frame_skip} frames → ~{estimated_frames} frames at {target_fps} FPS")
        
        frame_num = 0
        processed_frames = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process frames at the target interval
            if frame_num % frame_skip == 0:
                # Calculate timestamp based on original video timing
                timestamp = frame_num / original_fps
                
                # Detect cells in current frame
                centers, processed_frame = self.detect_cells_in_frame(frame)
                
                # Categorize cells by column/ROI
                categorized_cells = self.categorize_cells(centers)
                
                # Track cells per ROI
                self.track_cells_by_roi(categorized_cells, processed_frames, timestamp)
                
                # Store frame data with original structure for compatibility
                self.frame_data.append({
                    'frame_number': frame_num,  # Original frame number
                    'timestamp': timestamp,
                    'all_positions': centers,
                    'categorized_positions': categorized_cells
                })
                
                processed_frames += 1
                
                # Progress update
                if processed_frames % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_frames / elapsed
                    percent = (frame_num / total_frames) * 100
                    print(f"Processed {processed_frames} frames ({rate:.1f} fps) - {percent:.1f}% complete - {len(centers)} cells detected")
            
            frame_num += 1
        
        cap.release()
        print(f"FPS-limited analysis complete! Processed {processed_frames} frames from {frame_num} total frames")
    
    def analyze_video_limited_with_fps(self, max_frames, target_fps):
        """Analyze limited frames with FPS sampling."""
        print(f"Analyzing first {max_frames} frames with target FPS: {target_fps}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame skip interval
        frame_skip = max(1, int(original_fps / target_fps))
        max_processed = max_frames // frame_skip
        
        print(f"Video: {total_frames} total frames, {original_fps:.2f} FPS")
        print(f"Sampling: Every {frame_skip} frames → ~{max_processed} processed frames")
        
        frame_num = 0
        processed_frames = 0
        start_time = time.time()
        
        while frame_num < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process frames at the target interval
            if frame_num % frame_skip == 0:
                # Calculate timestamp based on original video timing
                timestamp = frame_num / original_fps
                
                # Detect cells in current frame
                centers, processed_frame = self.detect_cells_in_frame(frame)
                
                # Categorize cells by column/ROI
                categorized_cells = self.categorize_cells(centers)
                
                # Track cells per ROI
                self.track_cells_by_roi(categorized_cells, processed_frames, timestamp)
                
                processed_frames += 1
                
                # Progress update
                if processed_frames % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_frames / elapsed
                    print(f"Processed {processed_frames} frames ({rate:.1f} fps) - {len(centers)} cells detected")
            
            frame_num += 1
        
        cap.release()
        print(f"Limited FPS analysis complete! Processed {processed_frames} frames from {frame_num} total frames")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Cell Tracking Analysis Tool with Improved Tracking and ROI support')
    parser.add_argument('settings_file', nargs='?', help='Path to settings.txt file')
    parser.add_argument('--test', type=int, metavar='N', help='Test mode: process only first N frames')
    parser.add_argument('--fps', type=float, metavar='FPS', help='Target analysis FPS (skips frames to achieve speed)')
    parser.add_argument('--max-distance', type=float, default=75, help='Max distance for cell tracking (default: 75)')
    parser.add_argument('--max-disappeared', type=int, default=10, help='Max frames cell can disappear (default: 10)')
    parser.add_argument('--kf-q', type=float, default=1.0, help='Kalman process noise scale (default: 1.0)')
    parser.add_argument('--kf-r', type=float, default=1.0, help='Kalman measurement noise scale (default: 1.0)')
    parser.add_argument('--reid-gap', type=int, default=60, help='Max gap (frames) to re-identify a disappeared track (default: 60)')
    parser.add_argument('--reid-dist', type=float, default=60.0, help='Max distance (px) to re-identify a disappeared track (default: 60)')
    
    if len(sys.argv) == 1:
        # Interactive mode if no arguments provided
        settings_file = input("Enter path to settings file: ").strip()
        if not settings_file:
            print("No settings file provided. Exiting.")
            return
        test_frames = None
    else:
        args = parser.parse_args()
        settings_file = args.settings_file
        test_frames = args.test
        target_fps = args.fps
        max_distance = args.max_distance  
        max_disappeared = args.max_disappeared
        kf_q = args.kf_q
        kf_r = args.kf_r
        reid_gap = args.reid_gap
        reid_dist = args.reid_dist
    
    try:
        # Create tracker and run analysis
        tracker = CellTracker(settings_file)
        
        # Configure tracker if arguments provided
        if 'max_distance' in locals():
            # Default global tracker
            tracker.tracker.max_distance = max_distance
            tracker.tracker.max_disappeared = max_disappeared
            # Per-ROI trackers (if already created)
            for tr in tracker.trackers_per_roi.values():
                tr.max_distance = max_distance
                tr.max_disappeared = max_disappeared
                tr.process_noise_scale = kf_q if 'kf_q' in locals() else tr.process_noise_scale
                tr.measurement_noise_scale = kf_r if 'kf_r' in locals() else tr.measurement_noise_scale
            for sr in tracker.simple_trackers_per_roi.values():
                sr.max_distance = max_distance
                sr.max_disappeared = max_disappeared
                sr.reid_max_gap_frames = reid_gap
                sr.reid_max_distance = reid_dist
            print(f"Tracker configured: max_distance={max_distance}, max_disappeared={max_disappeared}, kf_q={locals().get('kf_q', 'N/A')}, kf_r={locals().get('kf_r', 'N/A')}, reid_gap={locals().get('reid_gap','N/A')}, reid_dist={locals().get('reid_dist','N/A')}")
        
        if test_frames:
            print(f"🧪 Test mode: processing first {test_frames} frames")
            if 'target_fps' in locals() and target_fps:
                tracker.run_analysis_limited(test_frames, target_fps)
            else:
                tracker.run_analysis_limited(test_frames)
        else:
            if 'target_fps' in locals() and target_fps:
                tracker.run_analysis(target_fps)
            else:
                tracker.run_analysis()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())