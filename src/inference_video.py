"""
Video Inference Script with Bag Counting
Real-time bag detection and counting for Fillpac conveyor monitoring
"""

import cv2
import argparse
import numpy as np 
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import supervision as sv
import threading
import time
import os
import logging
import logging.handlers
import sys
from typing import Optional, Tuple
import yaml

# Ultra-low latency FFMPEG options for RTSP streaming
# Aggressive settings to minimize buffering and delay
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"           # TCP is more stable than UDP for industrial RTSP
    "fflags;nobuffer|"               # Disable buffering
    "flags;low_delay|"               # Low delay mode
    "framedrop;1|"                   # Drop frames if needed to reduce latency
    "max_delay;0|"                   # No delay tolerance
    "reorder_queue_size;0|"          # Disable frame reordering
    "buffer_size;0"                  # Minimal buffer size
)


def setup_logging(log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_file: Path to log file (None for console only)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("BagCounter")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


from collections import deque

def letterbox(frame, target_size=(478, 850)):
    """Preserve aspect ratio while resizing (letterboxing)"""
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create black canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return canvas

class ThreadedCamera:
    """Helper class for threaded frame reading with automatic reconnection"""
    
    def __init__(self, source, target_size=None, max_reconnect_attempts=5, reconnect_delay=2.0, rotation=0):
        self.source = source
        self.target_size = target_size
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.rotation = rotation # 0, 90, 180, 270
        self.logger = logging.getLogger("BagCounter.Camera")
        
        # Initialize camera with error handling
        self.cap = None
        self._connect_camera()
        
        # Frame buffer and state
        self.frame_buffer = deque(maxlen=1)
        self.frame_id = 0
        self.started = False
        self.thread = None
        self.last_frame_time = time.time()
        
        # Pull first frame
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and self._validate_frame(frame):
                frame = self._apply_rotation(frame)
                if self.target_size:
                    frame = letterbox(frame, self.target_size)
                self.frame_buffer.append(frame)
                self.frame_id = 1
                self.logger.info(f"Camera initialized successfully: {self.source} (Rotation: {self.rotation})")
            else:
                self.logger.warning("Failed to read initial frame")
    
    def _connect_camera(self):
        """Connect to camera with error handling"""
        try:
            # Clean the source string
            if isinstance(self.source, str):
                self.source = self.source.strip()
            
            # Use FFMPEG backend explicitly for RTSP to avoid fallback issues
            if isinstance(self.source, str) and self.source.startswith("rtsp"):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera: {self.source}. Check your network connection and URL.")
            
            # Ultra-low latency settings
            if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.logger.info(f"Connected to camera: {self.source}")
            
        except Exception as e:
            error_msg = str(e)
            if "CAP_IMAGES" in error_msg:
                error_msg = ("OpenCV tried to read the URL as an image sequence. "
                             "This usually happens when the RTSP stream is unreachable or the backend failed.")
            self.logger.error(f"Camera connection error: {error_msg}")
            self.cap = None
            raise
    
    def _reconnect_camera(self):
        """Attempt to reconnect to camera with exponential backoff"""
        for attempt in range(self.max_reconnect_attempts):
            try:
                self.logger.warning(f"Reconnection attempt {attempt + 1}/{self.max_reconnect_attempts}")
                
                if self.cap:
                    self.cap.release()
                
                time.sleep(self.reconnect_delay * (attempt + 1))  # Exponential backoff
                self._connect_camera()
                
                if self.cap and self.cap.isOpened():
                    self.logger.info("Camera reconnected successfully")
                    return True
                    
            except Exception as e:
                self.logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        self.logger.error("All reconnection attempts failed")
        return False
    
    def _apply_rotation(self, frame):
        """Rotate frame based on configuration"""
        if self.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    @staticmethod
    def _validate_frame(frame) -> bool:
        """Validate that frame is not corrupted"""
        if frame is None:
            return False
        if frame.size == 0:
            return False
        if len(frame.shape) != 3:
            return False
        return True

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        """Background thread that reads frames with automatic reconnection"""
        target_interval = 1.0 / 30.0
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.started:
            start_time = time.time()
            
            try:
                # Check if camera is open
                if not self.cap or not self.cap.isOpened():
                    self.logger.warning("Camera not open, attempting reconnection")
                    if not self._reconnect_camera():
                        self.logger.error("Failed to reconnect, stopping camera thread")
                        break
                    consecutive_failures = 0
                
                # Read frame
                grabbed, frame = self.cap.read()
                
                if grabbed and self._validate_frame(frame):
                    # Rotate frame
                    frame = self._apply_rotation(frame)
                    
                    # Resize if needed
                    if self.target_size:
                        # Use standard resize for industrial aspect ratio forcing if requested
                        frame = cv2.resize(frame, self.target_size)
                    # Update buffer
                    self.frame_buffer.append(frame)
                    self.frame_id += 1
                    self.last_frame_time = time.time()
                    consecutive_failures = 0
                    
                    # No artificial frame rate limiting - use full sensor speed 
                    # deque(maxlen=1) handles latency by dropping stale frames
                else:
                    # Frame read failed
                    consecutive_failures += 1
                    self.logger.debug(f"Frame read failed (attempt {consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.error(f"Too many consecutive failures ({consecutive_failures}), attempting reconnection")
                        if not self._reconnect_camera():
                            self.logger.error("Reconnection failed, stopping camera thread")
                            break
                        consecutive_failures = 0
                    
                    time.sleep(0.01)
                    
            except Exception as e:
                self.logger.error(f"Error in camera update loop: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    if not self._reconnect_camera():
                        break
                    consecutive_failures = 0
                time.sleep(0.1)

    def read(self):
        if len(self.frame_buffer) > 0:
            return True, self.frame_buffer[0], self.frame_id
        return False, None, self.frame_id

    def release(self):
        self.started = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.cap.release()

    def get(self, prop):
        return self.cap.get(prop)

    def isOpened(self):
        return self.cap.isOpened()


class BagCounterVideo:
    """Real-time bag counter for video streams with configuration support"""
    
    def __init__(
        self, 
        weights_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        counting_mode: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the Bag Counter.

        This system uses a configuration-first approach, where values are loaded from a 
        YAML file and can be overridden by specific parameters provided during construction.

        Args:
            weights_path (str, optional): Overrides model path in config.
            conf_threshold (float, optional): Overrides confidence threshold.
            counting_mode (str, optional): 'line' or 'zone'. Overrides config.
            config_path (str, optional): Path to the system configuration YAML.
        
        Raises:
            RuntimeError: If the YOLO model cannot be loaded.
            ValueError: If configuration values are semantically invalid.
        """
        self.logger = logging.getLogger("BagCounter.Detector")
        
        # Load and validate configuration
        self.config = self._load_config(config_path)
        
        # Override with constructor parameters & validate types
        if weights_path:
            if not os.path.exists(weights_path):
                raise ValueError(f"Weights file not found: {weights_path}")
            self.config['model']['weights'] = weights_path
            
        if conf_threshold is not None:
            if not (0.0 <= conf_threshold <= 1.0):
                raise ValueError("Confidence threshold must be between 0.0 and 1.0")
            self.config['model']['confidence'] = conf_threshold
            
        if counting_mode:
            if counting_mode not in ['line', 'zone']:
                raise ValueError("counting_mode must be either 'line' or 'zone'")
            self.config['counting']['mode'] = counting_mode
        
        # Initialize YOLO model with error handling
        try:
            model_path = self.config['model']['weights']
            self.logger.info(f"Loading YOLO model from: {model_path}")
            self.model = YOLO(model_path)
            self.logger.info("YOLO model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Could not load model from {model_path}: {e}")
        
        self.conf_threshold = self.config['model']['confidence']
        self.counting_mode = self.config['counting']['mode']
        
        # Check if GPU is available and set half precision if so
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half = self.config['model'].get('half', False) and (self.device == 'cuda')
        
        try:
            self.model.to(self.device)
            if self.half:
                self.logger.info("GPU detected, using half-precision inference")
        except Exception as e:
            self.logger.warning(f"Failed to move model to {self.device}: {e}")
        
        # Tracker for unique bag identification with config
        track_config = self.config.get('tracking', {})
        # Use same activation threshold as confidence to avoid double-filtering
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.conf_threshold,
            lost_track_buffer=self.config['tracking'].get('lost_track_buffer', 30),
            minimum_matching_threshold=self.config['tracking'].get('minimum_matching_threshold', 0.6),
            frame_rate=30,
            minimum_consecutive_frames=1
        )
        
        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)
        
        # Region of Interest (ROI) - from config
        roi_config = self.config.get('roi', {})
        if roi_config.get('enabled', True):
            self.roi_polygon = np.array(roi_config.get('polygon', [[0, 450], [0, 50], [478, 50], [478, 450]]))
            self.roi_zone = sv.PolygonZone(polygon=self.roi_polygon)
            self.roi_annotator = sv.PolygonZoneAnnotator(zone=self.roi_zone, thickness=2)
            
            # ROI Bounding Box for auto-cropping
            x_min, y_min = np.min(self.roi_polygon, axis=0)
            x_max, y_max = np.max(self.roi_polygon, axis=0)
            self.roi_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
        else:
            self.roi_polygon = None
            self.roi_zone = None
            self.roi_annotator = None
            self.roi_bbox = None

        # Shared state for async/sync inference
        self.last_processed_frame = None
        self.last_tracked_detections = sv.Detections.empty()
        self.current_count = 0
        self.inference_thread = None
        self.inference_active = False
        self.inference_lock = threading.Lock()
        self.count_log = []
        self.last_trigger_time = 0
        self.last_printed_count = -1
        self.last_printed_frame = 0

    def _draw_calibration_overlays(self, frame: np.ndarray) -> np.ndarray:
        """
        Overlay industrial calibration guides to help site teams position cameras.
        Shows the "Sweet Spot" (478x850 aspect ratio) used during training.
        """
        cal_cfg = self.config.get('calibration', {})
        if not cal_cfg.get('show_training_zone', False) and not cal_cfg.get('show_guideline_boxes', False):
            return frame

        H, W = frame.shape[:2]
        TRAIN_W, TRAIN_H = 478, 850
        TRAIN_ASPECT = TRAIN_W / TRAIN_H
        alpha = cal_cfg.get('overlay_alpha', 0.3)

        # Calculate Training Zone coordinates
        target_h_in_source = H
        target_w_in_source = int(H * TRAIN_ASPECT)
        start_x = (W - target_w_in_source) // 2
        end_x = start_x + target_w_in_source

        # 1. Training Zone Mask
        if cal_cfg.get('show_training_zone', True):
            mask = np.zeros_like(frame)
            cv2.rectangle(mask, (start_x, 0), (end_x, H), (255, 255, 255), -1)
            frame = cv2.addWeighted(frame, 1.0 - alpha, cv2.bitwise_and(frame, mask), alpha, 0)
            cv2.rectangle(frame, (start_x, 0), (end_x, H), (0, 255, 255), 2) # Yellow border

        # 2. Guideline Boxes (Typical bag scale)
        if cal_cfg.get('show_guideline_boxes', True):
            # Target bag size should be roughly 40-60% of vertical training box width
            bw, bh = target_w_in_source * 0.5, target_h_in_source * 0.2
            cx, cy = W // 2, H // 2
            cv2.rectangle(frame, (int(cx-bw/2), int(cy-bh/2)), (int(cx+bw/2), int(cy+bh/2)), (0, 255, 0), 1)
            cv2.putText(frame, "IDEAL BAG SIZE", (int(cx-bw/2), int(cy-bh/2)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        return frame

    def _load_config(self, config_path: Optional[str]) -> dict:
        """
        Load system configuration with default fallback values.

        Merging strategy: Default values are provided for all critical parameters. 
        If a config file exists, its values will override the defaults. 
        Deep merging is performed only on top-level sections.

        Args:
            config_path (str, optional): Path to the YAML configuration file.

        Returns:
            dict: The resolved configuration dictionary.
        """
        defaults = {
            'camera': {'source': None, 'buffer_size': 0, 'fps_limit': 30, 'reconnect_attempts': 5, 'reconnect_delay': 2.0},
            'video': {'maintain_original_speed': True, 'speed_multiplier': 1.0, 'frame_skip': 1},
            'model': {'weights': 'models/weights/best.pt', 'confidence': 0.25, 'imgsz': 416, 'half': True},
            'roi': {'enabled': False, 'polygon': [[0, 850], [0, 0], [478, 0], [478, 850]]},
            'counting': {
                'mode': 'line', 
                'min_area': 500,
                'direction': 'both',
                'trigger_anchor': 'center',
                'line': {'start': [0, 425], 'end': [478, 425]}
            },
            'display': {'window_name': 'Fillpac Bag Counter', 'show_fps': True, 'show_count': True, 'show_tracking_ids': True, 'diagnostic_mode': True},
            'logging': {'level': 'INFO', 'file': 'logs/inference.log', 'max_bytes': 10485760, 'backup_count': 5},
            'tracking': {'track_activation_threshold': 0.25, 'lost_track_buffer': 90, 'minimum_matching_threshold': 0.7, 'minimum_consecutive_frames': 2}
        }
        
        if config_path:
            if not os.path.exists(config_path):
                self.logger.warning(f"Config file not found at {config_path}. Using defaults.")
            else:
                try:
                    with open(config_path, 'r') as f:
                        user_config = yaml.safe_load(f)
                        if not isinstance(user_config, dict):
                            raise ValueError("Config file must be a valid YAML dictionary")
                        
                        # Merge user config with defaults deeply
                        for section, values in user_config.items():
                            if section in defaults:
                                if isinstance(values, dict):
                                    defaults[section].update(values)
                                else:
                                    defaults[section] = values
                            else:
                                defaults[section] = values
                    self.logger.info(f"Loaded configuration from {config_path}")
                except Exception as e:
                    self.logger.error(f"Error parsing config file: {e}. Using defaults.")
        
        return defaults
        
    def _apply_auto_crop(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame to the ROI bounding box if enabled"""
        if not self.config.get('roi', {}).get('auto_crop', False) or self.roi_bbox is None:
            return frame
        
        x1, y1, x2, y2 = self.roi_bbox
        H, W = frame.shape[:2]
        
        # Clamp to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2]
        return frame

    def _draw_styled_label(self, frame: np.ndarray, text: str, position: tuple, 
                         bg_color: tuple = (255, 255, 255), text_color: tuple = (0, 0, 0),
                         scale: float = 1.2, thickness: int = 3) -> np.ndarray:
        """Draw text with a background box (Matches supervision LineZoneAnnotator style)"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        x, y = position
        padding_x = 20
        padding_y = 15
        
        # Draw background box (white)
        cv2.rectangle(frame, 
                      (x, y - text_height - padding_y), 
                      (x + text_width + padding_x * 2, y + padding_y), 
                      bg_color, -1)
        
        # Draw text (black)
        cv2.putText(frame, text, (x + padding_x, y), font, scale, text_color, thickness, cv2.LINE_AA)
        return frame

    def _inference_loop(self, cap):
        print("✓ Inference thread started (Asynchronous)")
        last_processed_id = -1
        while self.inference_active:
            ret, frame, frame_id = cap.read()
            if not ret or frame is None or frame_id == last_processed_id:
                time.sleep(0.001)
                continue
            
            last_processed_id = frame_id
                
            # Run YOLO detection with high speed settings
            results = self.model(
                frame, 
                conf=self.conf_threshold, 
                classes=[0], 
                verbose=False, 
                half=self.half, 
                imgsz=416 # Lower internal resolution for raw speed
            )[0]
            
            detections = sv.Detections.from_ultralytics(results)
            
            # Filter by ROI
            if self.roi_zone:
                mask = self.roi_zone.trigger(detections)
                detections = detections[mask]
                
            # Filter by area if needed
            min_area = self.config['counting'].get('min_area', 500)
            valid_mask = detections.area > min_area
            self.valid_detections = detections[valid_mask]
            self.filtered_detections = detections[~valid_mask]
            
            # Track and Update state
            tracked = self.tracker.update_with_detections(self.valid_detections)
            
            with self.inference_lock:
                self.last_processed_frame = frame.copy()
                self.last_tracked_detections = tracked
                self.filtered_detections = detections[~valid_mask] # Corrected: use localized detections
                
                # Check for triggers
                prev_in = self.line_zone.in_count
                prev_out = self.line_zone.out_count
                
                self.line_zone.trigger(tracked)
                
                # Update current count based on direction mode
                direction = self.config['counting'].get('direction', 'both')
                if direction == 'in':
                    self.current_count = self.line_zone.in_count
                elif direction == 'out':
                    self.current_count = self.line_zone.out_count
                else: # both
                    self.current_count = self.line_zone.in_count + self.line_zone.out_count
                
                # Trigger "pulse" effect if count increased
                if self.line_zone.in_count > prev_in or self.line_zone.out_count > prev_out:
                    self.last_trigger_time = time.time()

    def setup_counting_zone(self, frame_shape: tuple):
        """
        Configure the counting boundary (line or polygon zone).
        
        This method initializes the line or zone based on the loaded configuration
        and the dimensions of the input video stream.

        Args:
            frame_shape (tuple): The (height, width) or (height, width, channels) of the frame.
        
        Raises:
            ValueError: If frame_shape is invalid.
        """
        if not isinstance(frame_shape, (tuple, list)) or len(frame_shape) < 2:
            raise ValueError(f"Invalid frame_shape provided: {frame_shape}")
            
        height, width = frame_shape[:2]
        config_counting = self.config.get('counting', {})
        
        if self.counting_mode == 'line':
            line_cfg = config_counting.get('line', {})
            # Prioritize line_y if available, otherwise use line.start/end
            line_y = config_counting.get('line_y')
            
            if line_y is not None:
                # If line_y is < 1.0, treat as percentage
                if line_y <= 1.0:
                    y_coord = int(height * line_y)
                else:
                    y_coord = int(line_y)
                start_cfg = [0, y_coord]
                end_cfg = [width, y_coord]
            else:
                start_cfg = line_cfg.get('start', [0, int(height * 0.5)])
                end_cfg = line_cfg.get('end', [width, int(height * 0.5)])
            
            start_point = sv.Point(int(start_cfg[0]), int(start_cfg[1]))
            end_point = sv.Point(int(end_cfg[0]), int(end_cfg[1]))
            
            anchor_map = {
                'center': sv.Position.CENTER,
                'bottom_center': sv.Position.BOTTOM_CENTER,
                'top_center': sv.Position.TOP_CENTER
            }
            anchor = anchor_map.get(config_counting.get('trigger_anchor', 'center'), sv.Position.CENTER)
            
            self.line_zone = sv.LineZone(
                start=start_point, 
                end=end_point,
                triggering_anchors=[anchor]
            )
            self.zone_annotator = sv.LineZoneAnnotator(
                thickness=4,
                text_thickness=4,
                text_scale=2.0, 
                display_in_count=False, # Disable default labels to use custom styled ones
                display_out_count=False
            )
        else:
            # Zone mode
            zone_cfg = config_counting.get('zone', {})
            poly_cfg = zone_cfg.get('polygon', [
                [0, int(height * 0.6)],
                [width, int(height * 0.6)],
                [width, height],
                [0, height]
            ])
            polygon = np.array(poly_cfg)
            self.line_zone = sv.PolygonZone(polygon=polygon)
            self.zone_annotator = sv.PolygonZoneAnnotator(
                zone=self.line_zone,
                color=sv.Color.red(),
                thickness=2,
                text_thickness=2,
                text_scale=1
            )
        
        self.logger.info(f"Counting zone configured: {self.counting_mode} mode")
    
    def process_video(
        self, 
        video_source: str, 
        output_path: Optional[str] = None,
        display: bool = True,
        log_file: Optional[str] = None,
        sync_mode: bool = False
    ):
        """
        The main processing entry point for a video stream.
        
        This method orchestrates the threaded reading, async inference (if applicable), 
        logic for line crossing/zone triggers, and visual display/saving.

        Args:
            video_source (str): File path, RTSP/HTTP URL, or camera index.
            output_path (str, optional): Destination for saved annotated video (.mp4).
            display (bool): Whether to show the OpenCV window. Defaults to True.
            log_file (str, optional): Destination for the frame-by-frame counting log.
            sync_mode (bool): If True, processes synchronously (waits for each frame). 
                            If False (default), stays in zero-latency/async mode.
        """
        # Type validation for critical parameters
        if not video_source:
            raise ValueError("video_source cannot be empty")
            
        # 1. Open video source with ThreadedCamera
        is_live = str(video_source).isdigit() or str(video_source).startswith(('rtsp://', 'http://', 'https://'))
        
        # Target dimensions for YOLO processing (internal imgsz usually 416-640)
        # We handle resizing per frame to preserve source aspect ratio for display.
        yolo_width, yolo_height = 416, 416 
        
        if is_live:
            rotation = self.config.get('camera', {}).get('rotation', 0)
            target_res = self.config.get('display', {}).get('target_resolution')
            if target_res:
                target_res = tuple(target_res) # (width, height)
            
            cap = ThreadedCamera(
                int(video_source) if video_source.isdigit() else video_source,
                target_size=target_res,
                rotation=rotation
            )
            cap.start()
            print(f"✓ Started zero-latency camera: {video_source} (Rotation: {rotation} deg)")
            if target_res:
                print(f"✓ Forced Resolution: {target_res[0]}x{target_res[1]}")
        else:
            cap = cv2.VideoCapture(video_source)
            print(f"✓ Opened video file: {video_source}")
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_source}")
            return
        
        # Get video properties
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps_raw) if 0 < fps_raw < 150 else 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Setup counting zone
        self.setup_counting_zone((height, width, 3))
        
        # Start Inference Thread for live streams
        if is_live:
            self.inference_active = True
            self.inference_thread = threading.Thread(target=self._inference_loop, args=(cap,))
            self.inference_thread.daemon = True
            self.inference_thread.start()
        
        # Video writer setup
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Use current frame dimensions for the writer
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                print(f"⚠ Warning: Could not initialize video writer for {output_path}")
                writer = None
        
        # Processing loop
        frame_count = 0
        start_time = datetime.now()
        last_fps_time = time.time()
        actual_fps = 0
        tracked_detections = sv.Detections.empty()
        current_count = 0
        
        # Frame rate control for video files (not live streams)
        video_config = self.config.get('video', {})
        maintain_speed = video_config.get('maintain_original_speed', True)
        speed_multiplier = video_config.get('speed_multiplier', 1.0)
        frame_skip = video_config.get('frame_skip', 1)  # Process every Nth frame
        target_frame_time = (1.0 / fps / speed_multiplier) if (not is_live and maintain_speed) else 0
        last_frame_time = time.time()
        
        # Frame skipping state for video files
        last_detections = sv.Detections.empty()
        last_tracked = sv.Detections.empty()
        
        print(f"\n{'='*60}")
        print(f"Processing... {video_source}")
        if not is_live and maintain_speed:
            print(f"Playback speed: {speed_multiplier}x (Frame time: {target_frame_time*1000:.1f}ms)")
            if frame_skip > 1:
                print(f"Frame skip: Processing every {frame_skip} frame(s) for real-time playback")
        print(f"Press 'q' to quit")
        print(f"{'='*60}\n")
        
        try:
            while cap.isOpened() if not is_live else cap.started:
                # Frame timing - start of iteration for accurate frame rate control
                frame_start_time = time.time()
                
                # Handle both ThreadedCamera (3 returns) and VideoCapture (2 returns)
                if is_live:
                    ret, frame, fid = cap.read()
                else:
                    ret, frame = cap.read()
                    fid = frame_count
                
                if not ret or frame is None:
                    if is_live:
                        continue
                    else:
                        break
            
                # Console Monitoring - Print only when count increases or periodically (every 100 frames)
                if current_count > self.last_printed_count:
                    elapsed = time.time() - last_fps_time
                    actual_fps = (frame_count - self.last_printed_frame) / elapsed if elapsed > 0 else 0
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] BAG DETECTED! | Total: {current_count} | FPS: {actual_fps:.1f}")
                    sys.stdout.flush()
                    self.last_printed_count = current_count
                    self.last_printed_frame = frame_count
                    last_fps_time = time.time()
                elif frame_count % 300 == 0:
                    elapsed = time.time() - last_fps_time
                    actual_fps = 300 / elapsed if elapsed > 0 else 0
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring... FPS: {actual_fps:.1f} | Bags: {current_count}")
                    sys.stdout.flush()
                    last_fps_time = time.time()
                    self.last_printed_frame = frame_count
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    now = time.time()
                    elapsed = now - last_fps_time
                    actual_fps = 30 / elapsed if elapsed > 0 else 0
                    last_fps_time = now

                if is_live:
                    with self.inference_lock:
                        # Synchronized mode vs Zero-Latency mode
                        if sync_mode and self.last_processed_frame is not None:
                            # Use the frame that was actually processed by YOLO
                            frame = self.last_processed_frame
                        # Otherwise 'frame' is already the latest from cap.read()
                        
                        tracked_detections = self.last_tracked_detections
                        current_count = self.current_count
                else:
                    # Processing for video files with optional frame skipping
                    # Process every Nth frame with YOLO, but display all frames
                    should_process = (frame_count % frame_skip == 0)
                    
                    if should_process:
                        # Preserve original frame for display
                        inf_frame = letterbox(frame, (yolo_width, yolo_height))
                        results = self.model(inf_frame, conf=self.conf_threshold, classes=[0], verbose=False, half=self.half, imgsz=416)[0]
                        detections = sv.Detections.from_ultralytics(results)
                        
                        # Map detections back to original frame size for display/counting
                        scale_w = frame.shape[1] / yolo_width
                        scale_h = frame.shape[0] / yolo_height
                        detections.xyxy[:, 0::2] *= scale_w
                        detections.xyxy[:, 1::2] *= scale_h
                        
                        if self.roi_zone:
                            mask = self.roi_zone.trigger(detections)
                            detections = detections[mask]
                            
                        # Filter by area
                        min_area = self.config['counting'].get('min_area', 500)
                        valid_mask = detections.area > min_area
                        self.valid_detections = detections[valid_mask]
                        self.filtered_detections = detections[~valid_mask]
                        
                        tracked_detections = self.tracker.update_with_detections(self.valid_detections)
                        
                        # Store for reuse on skipped frames
                        last_detections = self.valid_detections
                        last_tracked = tracked_detections
                    else:
                        # Reuse previous detections for skipped frames
                        self.valid_detections = last_detections
                        tracked_detections = last_tracked
                    
                    # Check for triggers (only on processed frames)
                    if should_process:
                        prev_in = self.line_zone.in_count
                        prev_out = self.line_zone.out_count
                        self.line_zone.trigger(tracked_detections)
                        
                        # Update current count based on direction mode
                        direction = self.config['counting'].get('direction', 'both')
                        if direction == 'in':
                            current_count = self.line_zone.in_count
                        elif direction == 'out':
                            current_count = self.line_zone.out_count
                        else: # both
                            current_count = self.line_zone.in_count + self.line_zone.out_count
                            
                        # Trigger "pulse" effect
                        if self.line_zone.in_count > prev_in or self.line_zone.out_count > prev_out:
                            self.last_trigger_time = time.time()
                            
                        self.current_count = current_count

                if display or writer:
                    scene = frame.copy()
                    
                    # current_count is already set above from self.current_count or local counting logic
                    
                    # Annotate detections
                    if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > 0:
                        scene = self.box_annotator.annotate(scene=scene, detections=tracked_detections)
                        labels = [f"#{tid}" for tid in tracked_detections.tracker_id]
                        scene = self.label_annotator.annotate(scene=scene, detections=tracked_detections, labels=labels)
                    
                    # Show raw detections in light-blue if tracker hasn't caught them yet
                    # This helps debug if detection is happening at all
                    if hasattr(self, 'valid_detections') and len(self.valid_detections) > 0:
                        raw_color = (255, 255, 0) # Cyan
                        for bbox in self.valid_detections.xyxy:
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(scene, (x1, y1), (x2, y2), raw_color, 1, lineType=cv2.LINE_AA)
                            cv2.putText(scene, "detecting...", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, raw_color, 1)

                    # Diagnostic View: Show filtered (small) detections
                    if self.config['display'].get('diagnostic_mode', True) and hasattr(self, 'filtered_detections'):
                        for bbox in self.filtered_detections.xyxy:
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(scene, (x1, y1), (x2, y2), (100, 100, 100), 1)
                            cv2.putText(scene, "small", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                    
                    # Annotate counting zone with pulse effect
                    if self.counting_mode == 'line' and self.line_zone:
                        annotated_frame = self.zone_annotator.annotate(frame=scene, line_counter=self.line_zone)
                        
                        # Fallback explicit line drawing for clarity if annotator is too thin
                        line_color = (0, 0, 255) # Red default
                        if hasattr(self, 'last_trigger_time') and (time.time() - self.last_trigger_time < 0.3):
                            line_color = (0, 255, 0) # Green pulse
                            
                        # Use sv.Point coordinates for compatibility
                        s = self.line_zone.vector.start if hasattr(self.line_zone, 'vector') else self.line_zone.start
                        e = self.line_zone.vector.end if hasattr(self.line_zone, 'vector') else self.line_zone.end
                        
                        # Explicit Red line for production visibility
                        cv2.line(annotated_frame, 
                                 (int(s.x), int(s.y)),
                                 (int(e.x), int(e.y)),
                                 line_color, 6)
                    else:
                        annotated_frame = self.zone_annotator.annotate(scene=scene)
                    
                    # Apply Calibration Overlays
                    annotated_frame = self._draw_calibration_overlays(annotated_frame)
                    
                    # Auto-crop the output if enabled
                    final_frame = self._apply_auto_crop(annotated_frame)
                    
                    # Draw Production-Ready UI ON THE FINAL FRAME
                    # Only showing IN bags as requested
                    in_count = self.line_zone.in_count if hasattr(self.line_zone, 'in_count') else current_count
                    
                    # Draw IN box
                    self._draw_styled_label(final_frame, f"IN BAGS: {in_count}", (20, 60), scale=1.2, thickness=3)
                    
                    # Smaller status labels for FPS/Confidence
                    if self.config['display'].get('show_fps', True):
                        status_text = f"LIVE | FPS: {actual_fps:.1f} | CONF: {self.conf_threshold:.2f}"
                        cv2.putText(final_frame, status_text, (20, 105), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    if writer and frame_count == 1:
                        # Re-initialize writer if frame size changed due to cropping
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        orig_writer = writer
                        h_final, w_final = final_frame.shape[:2]
                        writer = cv2.VideoWriter(output_path, fourcc, fps, (w_final, h_final))
                        orig_writer.release()

                    if writer: writer.write(final_frame)
                    if display:
                        window_name = self.config.get('display', {}).get('window_name', 'Fillpac Bag Counter')
                        # Ensure window snaps to video size and reset it if already open to clear gray area
                        if frame_count == 1:
                            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                            h_disp, w_disp = final_frame.shape[:2]
                            cv2.resizeWindow(window_name, w_disp, h_disp)
                            
                        cv2.imshow(window_name, final_frame)
                        # Minimal delay for event processing only
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'): 
                            self.logger.info("User quit with 'q'")
                            break
                
                if frame_count % fps == 0:
                    self.count_log.append({'timestamp': (datetime.now() - start_time).total_seconds(), 'frame': frame_count, 'count': current_count})
                
                # Frame rate limiting for video files to maintain original playback speed
                if not is_live and target_frame_time > 0:
                    elapsed_frame_time = time.time() - frame_start_time
                    sleep_time = target_frame_time - elapsed_frame_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        finally:
            self.inference_active = False
            if self.inference_thread: self.inference_thread.join(timeout=1.0)
            cap.release()
            if writer: writer.release()
            cv2.destroyAllWindows()
            
            duration = (datetime.now() - start_time).total_seconds()
            print(f"\n{'='*60}\nProcessing Complete\n{'='*60}")
            print(f"Total bags counted: {current_count}")
            print(f"Average Rate: {frame_count / duration:.2f} FPS\n{'='*60}\n")
            
            if log_file:
                self.save_log(log_file, current_count, frame_count, duration)
        
        return current_count

    def save_log(self, log_file: str, total_count: int, frames: int, duration: float):
        """Save counting log to file"""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w') as f:
            f.write(f"Fillpac Bag Counting Log\n")
            f.write(f"{'='*60}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Bags Counted: {total_count}\n")
            f.write(f"Total Frames: {frames}\n")
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write(f"{'='*60}\n\n")
            
            f.write("Frame-by-Frame Log:\n")
            for entry in self.count_log:
                f.write(f"Time: {entry['timestamp']:.2f}s, Frame: {entry['frame']}, Count: {entry['count']}\n")
        
        print(f"✓ Log saved to: {log_path}")



def main():
    """Main video inference script"""
    parser = argparse.ArgumentParser(description='Count bags in video using YOLO')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--weights', type=str, help='Path to trained weights (overrides config)')
    parser.add_argument('--source', type=str,
                       help='Path to video file or camera index (0 for webcam). Overrides config source.')
    parser.add_argument('--conf', type=float, help='Confidence threshold (overrides config)')
    parser.add_argument('--mode', type=str, choices=['line', 'zone'],
                       help='Counting mode (overrides config)')
    parser.add_argument('--output', type=str, help='Path to save annotated video')
    parser.add_argument('--log', type=str, help='Path to save count log')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    parser.add_argument('--sync', action='store_true', help='Sync video with detections')
    
    args = parser.parse_args()
    
    # 1. Initialize logging first (temporary counter to get config for logging)
    temp_counter = BagCounterVideo(config_path=args.config)
    log_cfg = temp_counter.config.get('logging', {})
    setup_logging(
        log_file=args.log or log_cfg.get('file'),
        level=log_cfg.get('level', 'INFO')
    )
    
    logger = logging.getLogger("BagCounter")
    logger.info("Initializing Fillpac Bag Counter...")
    
    # 2. Re-initialize counter with proper overrides for model settings
    counter = BagCounterVideo(
        weights_path=args.weights,
        conf_threshold=args.conf,
        counting_mode=args.mode,
        config_path=args.config
    )
    
    # Determine source (CLI overrides config)
    source = args.source or counter.config.get('camera', {}).get('source')
    
    if source is None:
        logger.error("No video source provided! Specify --source or set 'camera.source' in config.")
        sys.exit(1)
        
    # Process video
    counter.process_video(
        source,
        output_path=args.output,
        display=not args.no_display,
        log_file=args.log,
        sync_mode=args.sync
    )


if __name__ == '__main__':
    main()