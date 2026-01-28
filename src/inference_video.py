"""
Video Inference Script with Bag Counting
Real-time bag detection and counting for Fillpac conveyor monitoring
"""

import cv2
import argparse #Allows running the script with options (e.g., --source, --conf)
import numpy as np 
from pathlib import Path # For handling file paths
from datetime import datetime# For timestamping
from ultralytics import YOLO # YOLOv8 framework
import supervision as sv # For visualization and tracking
import threading # For threaded frame reading
import time #for thread management
import os #for os operations

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


from collections import deque

class ThreadedCamera:
    """Helper class for threaded frame reading with absolute latest frame delivery"""
    def __init__(self, source, target_size=None):
        self.cap = cv2.VideoCapture(source)
        
        # Ultra-low latency settings
        if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # Zero buffer for minimum latency
        
        # Limit FPS to prevent overwhelming the system
        # Camera reported 120 FPS which is too high
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Cap at 30 FPS for smooth operation
        
        self.target_size = target_size
        # Deque with maxlen=1 ensures we always have exactly the latest frame
        self.frame_buffer = deque(maxlen=1)
        self.frame_id = 0
        self.started = False
        self.thread = None
        
        # Pull first frame
        ret, frame = self.cap.read()
        if ret:
            if self.target_size:
                frame = cv2.resize(frame, self.target_size)
            self.frame_buffer.append(frame)
            self.frame_id = 1

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        target_interval = 1.0 / 30.0  # Target 30 FPS
        while self.started:
            start_time = time.time()
            
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue
            
            # Simple approach: just read frames as fast as possible
            # The deque(maxlen=1) automatically keeps only the latest
            grabbed, frame = self.cap.read()
            
            if grabbed and frame is not None:
                if self.target_size:
                    frame = cv2.resize(frame, self.target_size)
                self.frame_buffer.append(frame)
                self.frame_id += 1
                
                # Frame rate limiting for smooth delivery
                elapsed = time.time() - start_time
                sleep_time = target_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                time.sleep(0.001)

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
    """Real-time bag counter for video streams"""
    
    def __init__(
        self, 
        weights_path: str, 
        conf_threshold: float = 0.5,
        counting_mode: str = 'line'  # 'line' or 'zone'
    ):
        """
        Initialize video bag counter
        
        Args:
            weights_path: Path to trained YOLO weights
            conf_threshold: Confidence threshold for detections
            counting_mode: 'line' for line crossing or 'zone' for zone entry/exit
        """
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.counting_mode = counting_mode
        
        # Check if GPU is available and set half precision if so
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half = (self.device == 'cuda')
        self.model.to(self.device)
        if self.half:
            print("✓ GPU detected, using half-precision inference")
        
        # Tracker for unique bag identification
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=90,  # Increased memory for stability
            minimum_matching_threshold=0.7,
            minimum_consecutive_frames=2
        )
        
        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)
        
        # Region of Interest (ROI) - covering only the conveyor belt
        # Points based on TRAINING DATA: 478x850 (WxH)
        # Training annotations show bags in center region (normalized x: 0.2-0.8, y: 0.2-0.6)
        self.roi_polygon = np.array([
            [50, 850],   # Bottom Left
            [428, 850],  # Bottom Right (478-50 to stay within frame width)
            [350, 150],  # Top Right
            [80, 150]    # Top Left
        ])
        
        # Shared state for async/sync inference
        self.last_processed_frame = None
        self.last_tracked_detections = sv.Detections.empty()
        self.current_count = 0
        self.inference_thread = None
        self.inference_active = False
        self.inference_lock = threading.Lock()
        
        # Logging
        self.count_log = []
        
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
            
            # Filter by ROI and Area
            mask = sv.PolygonZone(polygon=self.roi_polygon).trigger(detections)
            detections = detections[mask]
            detections = detections[detections.area > 2000]
            
            # Track and Update state
            tracked = self.tracker.update_with_detections(detections)
            
            with self.inference_lock:
                self.last_processed_frame = frame.copy()
                self.last_tracked_detections = tracked
                if self.counting_mode == 'line':
                    self.line_zone.trigger(tracked)
                    self.current_count = self.line_zone.in_count  # Only IN count
                else:
                    self.line_zone.trigger(tracked)
                    self.current_count = self.line_zone.count

    def setup_counting_zone(self, frame_shape: tuple, zone_config: dict = None):
        """
        Setup counting line or zone
        
        Args:
            frame_shape: (height, width, channels) of video frame
            zone_config: Custom zone configuration (optional)
        """
        height, width = frame_shape[:2]
        
        if zone_config is None:
            # Default: horizontal line at 60% of frame height (for conveyor exit)
            if self.counting_mode == 'line':
                # Use BOTTOM_CENTER for more stable crossing detection on conveyors
                start_point = sv.Point(0, int(height * 0.5))
                end_point = sv.Point(width, int(height * 0.5))
                self.line_zone = sv.LineZone(
                    start=start_point, 
                    end=end_point,
                    triggering_anchors=[sv.Position.BOTTOM_CENTER]
                )
                self.zone_annotator = sv.LineZoneAnnotator(
                    thickness=4,
                    text_thickness=2,
                    text_scale=1.5,
                    display_in_count=True,
                    display_out_count=True  # Show both IN and OUT
                )
            else:
                # Zone mode: bottom 40% of frame
                polygon = np.array([
                    [0, int(height * 0.6)],
                    [width, int(height * 0.6)],
                    [width, height],
                    [0, height]
                ])
                self.line_zone = sv.PolygonZone(polygon=polygon)
                self.zone_annotator = sv.PolygonZoneAnnotator(
                    zone=self.line_zone,
                    color=sv.Color.red(),
                    thickness=2,
                    text_thickness=2,
                    text_scale=1
                )
        
        print(f"✓ Counting zone configured: {self.counting_mode} mode")
    
    def process_video(
        self, 
        video_source: str, 
        output_path: str = None,
        display: bool = True,
        log_file: str = None,
        sync_mode: bool = False
    ):
        """
        Process video and count bags with zero-latency streaming support
        """
        # Open video
        is_live = isinstance(video_source, int) or video_source.isdigit() or video_source.startswith(('rtsp://', 'http://', 'https://'))
        
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
                    if frame_count % 2 == 0: 
                        proc_frame = cv2.resize(frame, (target_width, target_height))
                        results = self.model(proc_frame, conf=self.conf_threshold, classes=[0], verbose=False, half=self.half, imgsz=416)[0]
                        detections = sv.Detections.from_ultralytics(results)
                        mask = sv.PolygonZone(polygon=self.roi_polygon).trigger(detections)
                        detections = detections[mask]
                        detections = detections[detections.area > 2000]
                        tracked_detections = self.tracker.update_with_detections(detections)
                        if self.counting_mode == 'line':
                            self.line_zone.trigger(tracked_detections)
                            current_count = self.line_zone.in_count  # Only IN count
                        else:
                            self.line_zone.trigger(tracked_detections)
                            current_count = self.line_zone.count
                        frame = proc_frame

                if display or writer:
                    scene = frame.copy()
                    cv2.polylines(scene, [self.roi_polygon], True, (255, 255, 0), 2)
                    if tracked_detections.tracker_id is not None:
                        # Removed trace_annotator for stable FPS during motion
                        scene = self.box_annotator.annotate(scene=scene, detections=tracked_detections)
                        labels = [f"#{tid}" for tid in tracked_detections.tracker_id]
                        scene = self.label_annotator.annotate(scene=scene, detections=tracked_detections, labels=labels)
                    else:
                        scene = self.box_annotator.annotate(scene=scene, detections=tracked_detections)
                    
                    if self.counting_mode == 'line':
                        annotated_frame = self.zone_annotator.annotate(frame=scene, line_counter=self.line_zone)
                    else:
                        annotated_frame = self.zone_annotator.annotate(scene=scene)
                    
                    info_text = [
                        f"Bags Counted: {current_count}",
                        f"Display FPS: {actual_fps:.1f}",
                        f"Sync: {'Synchronized (Delayed)' if sync_mode else 'Zero-Latency (Real-Time)'}"
                    ]
                    y = 30
                    for text in info_text:
                        cv2.putText(annotated_frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y += 30

                    if writer: writer.write(annotated_frame)
                    if display:
                        cv2.imshow('Fillpac Zero-Latency Bag Counter', annotated_frame)
                        # Non-blocking waitKey for maximum responsiveness
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'): break
                
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
    parser.add_argument('--weights', type=str, default='models/weights/best.pt',
                       help='Path to trained weights')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to video file or camera index (0 for webcam)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--mode', type=str, default='line', choices=['line', 'zone'],
                       help='Counting mode: line crossing or zone entry')
    parser.add_argument('--output', type=str, help='Path to save annotated video')
    parser.add_argument('--log', type=str, help='Path to save count log')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    parser.add_argument('--sync', action='store_true', help='Sync video with detections (adds slight delay)')
    
    args = parser.parse_args()
    
    # Initialize counter
    counter = BagCounterVideo(args.weights, args.conf, args.mode)
    
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