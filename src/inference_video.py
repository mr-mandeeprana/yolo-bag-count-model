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
    "rtsp_transport;udp|"           # UDP is faster than TCP (less overhead)
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
    
    def __init__(self, source, target_size=None, max_reconnect_attempts=5, reconnect_delay=2.0):
        self.source = source
        self.target_size = target_size
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
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
                if self.target_size:
                    frame = cv2.resize(frame, self.target_size)
                self.frame_buffer.append(frame)
                self.frame_id = 1
                self.logger.info(f"Camera initialized successfully: {self.source}")
            else:
                self.logger.warning("Failed to read initial frame")
    
    def _connect_camera(self):
        """Connect to camera with error handling"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera: {self.source}")
            
            # Ultra-low latency settings
            if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.logger.info(f"Connected to camera: {self.source}")
            
        except Exception as e:
            self.logger.error(f"Camera connection error: {e}")
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
                    # Resize if needed
                    if self.target_size:
                        frame = letterbox(frame, self.target_size)
                    # Update buffer
                    self.frame_buffer.append(frame)
                    self.frame_id += 1
                    self.last_frame_time = time.time()
                    consecutive_failures = 0
                    
                    # Frame rate limiting
                    elapsed = time.time() - start_time
                    sleep_time = target_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
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
        else:
            self.roi_polygon = None
            self.roi_zone = None
        
        # Shared state for async/sync inference
        self.last_processed_frame = None
        self.last_tracked_detections = sv.Detections.empty()
        self.current_count = 0
        self.inference_thread = None
        self.inference_active = False
        self.inference_lock = threading.Lock()
        
        # Logging
        self.count_log = []

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
            'camera': {'buffer_size': 0, 'fps_limit': 30, 'reconnect_attempts': 5, 'reconnect_delay': 2.0},
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
        
    def _inference_loop(self, cap):
        """Background thread for continuous YOLO inference"""
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
                self.filtered_detections = filtered_detections # Store for diagnostic display
                
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
                text_thickness=2,
                text_scale=1.5,
                display_in_count=True,
                display_out_count=True
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
        
        # Target dimensions for training consistency
        target_width, target_height = 478, 850
        
        if is_live:
            cap = ThreadedCamera(
                int(video_source) if video_source.isdigit() else video_source,
                target_size=(target_width, target_height)
            )
            cap.start()
            print(f"✓ Started zero-latency camera: {video_source}")
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
            writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
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
        
        print(f"\n{'='*60}")
        print(f"Processing... {video_source}")
        print(f"Press 'q' to quit")
        print(f"{'='*60}\n")
        
        try:
            while cap.isOpened():
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
                    # Processing for video files
                    proc_frame = letterbox(frame, (target_width, target_height))
                    results = self.model(proc_frame, conf=self.conf_threshold, classes=[0], verbose=False, half=self.half, imgsz=416)[0]
                    detections = sv.Detections.from_ultralytics(results)
                    
                    if self.roi_zone:
                        mask = self.roi_zone.trigger(detections)
                        detections = detections[mask]
                        
                    # Filter by area
                    min_area = self.config['counting'].get('min_area', 500)
                    valid_mask = detections.area > min_area
                    self.valid_detections = detections[valid_mask]
                    self.filtered_detections = detections[~valid_mask]
                    
                    tracked_detections = self.tracker.update_with_detections(self.valid_detections)
                    
                    # Check for triggers
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
                        
                    frame = proc_frame

                if display or writer:
                    scene = frame.copy()
                    
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
                    if self.counting_mode == 'line':
                        annotated_frame = self.zone_annotator.annotate(frame=scene, line_counter=self.line_zone)
                        
                        # Dynamic line color pulse
                        line_color = (0, 0, 255) # Red default
                        if hasattr(self, 'last_trigger_time') and (time.time() - self.last_trigger_time < 0.3):
                            line_color = (0, 255, 0) # Green pulse
                            
                        cv2.line(annotated_frame, 
                                 (int(self.line_zone.vector.start.x), int(self.line_zone.vector.start.y)),
                                 (int(self.line_zone.vector.end.x), int(self.line_zone.vector.end.y)),
                                 line_color, 4)
                    else:
                        annotated_frame = self.zone_annotator.annotate(scene=scene)
                    
                    # Draw diagnostic info
                    info_text = [
                        f"Bags Counted: {current_count}",
                        f"Display FPS: {actual_fps:.1f}",
                        f"Confidence: {self.conf_threshold:.2f}",
                        f"Direction: {self.config['counting'].get('direction', 'both').upper()}"
                    ]
                    y = 30
                    for text in info_text:
                        cv2.putText(annotated_frame, text, (15, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        y += 30

                    if writer: writer.write(annotated_frame)
                    if display:
                        window_name = self.config.get('display', {}).get('window_name', 'Fillpac Bag Counter')
                        cv2.imshow(window_name, annotated_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'): 
                            self.logger.info("User quit with 'q'")
                            break
                
                if frame_count % fps == 0:
                    self.count_log.append({'timestamp': (datetime.now() - start_time).total_seconds(), 'frame': frame_count, 'count': current_count})

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
    parser.add_argument('--source', type=str, required=True,
                       help='Path to video file or camera index (0 for webcam)')
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
    
    # Process video
    counter.process_video(
        args.source,
        output_path=args.output,
        display=not args.no_display,
        log_file=args.log,
        sync_mode=args.sync
    )


if __name__ == '__main__':
    main()