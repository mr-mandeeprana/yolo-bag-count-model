"""
Diagnostic script to test bag detection and counting
This will help identify where the counting is failing
"""

from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np

# Load model
model = YOLO('models/weights/best.pt')

# Open video
video_path = "WhatsApp Video 2026-01-20 at 12.38.18.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {width}x{height} @ {fps} FPS")
print(f"Testing first 100 frames...\n")

# Setup tracker
tracker = sv.ByteTrack()

# Setup counting line at different positions
line_positions = [0.3, 0.4, 0.5, 0.6, 0.7]

for line_pos in line_positions:
    print(f"\n{'='*60}")
    print(f"Testing line at {int(line_pos*100)}% of frame height")
    print(f"{'='*60}")
    
    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Create line zone
    start_point = sv.Point(0, int(height * line_pos))
    end_point = sv.Point(width, int(height * line_pos))
    line_zone = sv.LineZone(start=start_point, end=end_point)
    
    # Reset tracker
    tracker = sv.ByteTrack()
    
    total_detections = 0
    frames_with_detections = 0
    
    # Process 100 frames
    for frame_num in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        results = model(frame, conf=0.25, classes=[0], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        if len(detections) > 0:
            total_detections += len(detections)
            frames_with_detections += 1
        
        # Track
        tracked = tracker.update_with_detections(detections)
        
        # Update line
        line_zone.trigger(tracked)
    
    print(f"  Frames with detections: {frames_with_detections}/100")
    print(f"  Total detections: {total_detections}")
    print(f"  Bags crossed line: {line_zone.out_count}")
    print(f"  Bags in count: {line_zone.in_count}")

cap.release()

print(f"\n{'='*60}")
print("Diagnostic Complete")
print(f"{'='*60}")
print("\nConclusion:")
print("- If 'Frames with detections' is 0: Model isn't detecting bags")
print("- If 'Total detections' > 0 but 'Bags crossed' = 0: Line position issue")
print("- Check which line position gives the best count")
