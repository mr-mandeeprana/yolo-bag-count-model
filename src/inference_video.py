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


class ThreadedCamera:
    """Helper class for threaded frame reading to minimize lag on live streams"""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        self.started = False
        if self.thread:
            self.thread.join()
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
        # Points based on 478x850 frame
        self.roi_polygon = np.array([
            [50, 850],   # Bottom Left
            [450, 850],  # Bottom Right
            [300, 150],  # Top Right
            [100, 150]   # Top Left
        ])
        
        # Counting zones (will be set based on video dimensions)
        self.line_zone = None
        self.zone_annotator = None
        
        # Logging
        self.count_log = []
        
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
                    text_scale=1.5
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
        log_file: str = None
    ):
        """
        Process video and count bags
        
        Args:
            video_source: Path to video file or camera index (0 for webcam)
            output_path: Path to save annotated video (optional)
            display: Show real-time display (default: True)
            log_file: Path to save count log (optional)
        """
        # Open video
        is_live = isinstance(video_source, int) or video_source.isdigit() or video_source.startswith(('rtsp://', 'http://', 'https://'))
        
        if is_live:
            cap = ThreadedCamera(int(video_source) if video_source.isdigit() else video_source)
            cap.start()
            print(f"✓ Started threaded camera: {video_source}")
        else:
            cap = cv2.VideoCapture(video_source)
            print(f"✓ Opened video: {video_source}")
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Fallback for streams
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else 0
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Setup counting zone
        self.setup_counting_zone((height, width, 3))
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"✓ Saving output to: {output_path}")
        
        # Processing loop
        frame_count = 0
        start_time = datetime.now()
        
        # Custom resizing for performance (e.g., if source has 4K or 1080p, resize to 640 for speed)
        target_width = 850  # Matches ROI coordinate system
        resize_factor = target_width / width if width > 0 else 1.0
        
        print(f"\n{'='*60}")
        print(f"Processing video... Press 'q' to quit")
        if resize_factor != 1.0:
            print(f"Applying resize factor: {resize_factor:.2f}")
        print(f"{'='*60}\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Performance optimization: Resize frame if it's too large
            if resize_factor != 1.0:
                frame = cv2.resize(frame, (int(width * resize_factor), int(height * resize_factor)))
            
            # Optional: Skip frames if falling behind (only for live streams)
            # This is a simple implementation; more robust would use threading
            if frame_count % 2 != 0 and (isinstance(video_source, int) or video_source.isdigit() or video_source.startswith(('rtsp://', 'http://', 'https://'))):
                # We could potentially skip every other frame to keep up with real-time
                # For now, let's just resize and see if it's enough.
                pass

            # Run YOLO detection
            results = self.model(frame, conf=self.conf_threshold, classes=[0], verbose=False)[0]
            
            # Convert to Supervision detections
            detections = sv.Detections.from_ultralytics(results)
            
            # 1. Filter by ROI
            mask = sv.PolygonZone(polygon=self.roi_polygon).trigger(detections)
            detections = detections[mask]
            
            # 2. Filter by minimum area (ignore far-away noise or glitch boxes)
            # Area in pixels
            min_area = 2000 
            detections = detections[detections.area > min_area]
            
            # Track objects
            tracked_detections = self.tracker.update_with_detections(detections)
            
            # Update counting zone
            if self.counting_mode == 'line':
                self.line_zone.trigger(tracked_detections)
                # Combine both to be safe, as bags might cross in either direction depending on camera orientation
                # This ensures we don't miss a bag just because it crossed 'in' instead of 'out'
                current_count = self.line_zone.in_count + self.line_zone.out_count
            else:
                self.line_zone.trigger(tracked_detections)
                current_count = self.line_zone.count  # Bags currently in zone
            
            # Annotate frame
            scene = frame.copy()
            
            # Draw ROI
            cv2.polylines(scene, [self.roi_polygon], True, (255, 255, 0), 2)
            
            # Draw traces
            scene = self.trace_annotator.annotate(
                scene=scene,
                detections=tracked_detections
            )
            
            # Draw boxes
            scene = self.box_annotator.annotate(
                scene=scene,
                detections=tracked_detections
            )
            
            # Draw labels with tracker IDs
            labels = [
                f"#{tracker_id} {confidence:.2f}"
                for confidence, tracker_id in zip(tracked_detections.confidence, tracked_detections.tracker_id)
            ]
            scene = self.label_annotator.annotate(
                scene=scene,
                detections=tracked_detections,
                labels=labels
            )
            
            # Draw counting zone
            if self.counting_mode == 'line':
                annotated_frame = self.zone_annotator.annotate(
                    frame=scene,
                    line_counter=self.line_zone
                )
            else:
                annotated_frame = self.zone_annotator.annotate(
                    scene=scene
                )
            
            # Add info overlay
            info_text = [
                f"Frame: {frame_count}/{total_frames}",
                f"Bags Counted: {current_count}",
                f"Detections: {len(tracked_detections)}",
                f"FPS: {fps}"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(
                    annotated_frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                y_offset += 30
            
            # Save frame
            if writer:
                writer.write(annotated_frame)
            
            # Display
            if display:
                cv2.imshow('Fillpac Bag Counter', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break
            
            # Log count periodically (every second)
            if frame_count % fps == 0:
                self.count_log.append({
                    'timestamp': (datetime.now() - start_time).total_seconds(),
                    'frame': frame_count,
                    'count': current_count
                })
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print(f"Processing Complete")
        print(f"{'='*60}")
        print(f"Total frames processed: {frame_count}")
        print(f"Total bags counted: {current_count}")
        print(f"Processing time: {duration:.2f} seconds")
        print(f"Average FPS: {frame_count / duration:.2f}")
        print(f"{'='*60}\n")
        
        # Save log
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
    
    args = parser.parse_args()
    
    # Initialize counter
    counter = BagCounterVideo(args.weights, args.conf, args.mode)
    
    # Process video
    counter.process_video(
        args.source,
        output_path=args.output,
        display=not args.no_display,
        log_file=args.log
    )


if __name__ == '__main__':
    main()
