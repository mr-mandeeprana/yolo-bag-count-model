# Example: Real-time Bag Counting from Video

This notebook demonstrates real-time bag counting from Fillpac video feeds.

## Setup

```python
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from datetime import datetime

# Load trained model
model = YOLO('../models/weights/best.pt')
print("Model loaded successfully")
```

## Setup Video Source

```python
# Option 1: Video file
video_path = 'path/to/fillpac_video.mp4'

# Option 2: RTSP camera stream
# video_path = 'rtsp://camera-ip/stream'

# Option 3: USB camera
# video_path = 0

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {width}x{height} @ {fps} FPS")
```

## Setup Tracking and Counting

```python
# Initialize ByteTrack tracker
tracker = sv.ByteTrack()

# Setup counting line (horizontal line at 60% of frame height)
line_start = sv.Point(0, int(height * 0.6))
line_end = sv.Point(width, int(height * 0.6))
line_zone = sv.LineZone(start=line_start, end=line_end)

# Annotators
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)
box_annotator = sv.BoxAnnotator(thickness=2)

print("Tracking and counting setup complete")
```

## Process Video

```python
frame_count = 0
start_time = datetime.now()

# For notebook display
from IPython.display import clear_output
import matplotlib.pyplot as plt

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Run detection
    results = model(frame, conf=0.5, classes=[0], verbose=False)[0]
    
    # Convert to Supervision detections
    detections = sv.Detections.from_ultralytics(results)
    
    # Track objects
    tracked_detections = tracker.update_with_detections(detections)
    
    # Update counting line
    line_zone.trigger(tracked_detections)
    
    # Annotate frame
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=tracked_detections
    )
    annotated_frame = line_annotator.annotate(
        frame=annotated_frame,
        line_counter=line_zone
    )
    
    # Add info overlay
    cv2.putText(
        annotated_frame, 
        f"Frame: {frame_count} | Bags: {line_zone.out_count}", 
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )
    
    # Display every 30 frames (for notebook)
    if frame_count % 30 == 0:
        clear_output(wait=True)
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        plt.title(f"Bag Count: {line_zone.out_count}")
        plt.axis('off')
        plt.show()
    
    # Break after 300 frames for demo
    if frame_count >= 300:
        break

cap.release()

# Final statistics
end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

print(f"\n{'='*60}")
print(f"Processing Complete")
print(f"{'='*60}")
print(f"Total frames: {frame_count}")
print(f"Total bags counted: {line_zone.out_count}")
print(f"Processing time: {duration:.2f} seconds")
print(f"Average FPS: {frame_count / duration:.2f}")
print(f"{'='*60}")
```

## Save Results to Video

```python
# Reprocess and save to file
cap = cv2.VideoCapture(video_path)
output_path = 'fillpac_counted.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Reset tracker and counter
tracker = sv.ByteTrack()
line_zone = sv.LineZone(start=line_start, end=line_end)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Detection and tracking
    results = model(frame, conf=0.5, classes=[0], verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    tracked_detections = tracker.update_with_detections(detections)
    line_zone.trigger(tracked_detections)
    
    # Annotate
    annotated_frame = box_annotator.annotate(frame.copy(), tracked_detections)
    annotated_frame = line_annotator.annotate(annotated_frame, line_zone)
    
    # Write frame
    writer.write(annotated_frame)
    
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
writer.release()

print(f"\n✓ Output saved to: {output_path}")
print(f"Final bag count: {line_zone.out_count}")
```

## Analyze Counting Performance

```python
# Compare with ground truth (if available)
ground_truth_count = 45  # Replace with actual count

predicted_count = line_zone.out_count
error = predicted_count - ground_truth_count
accuracy = (1 - abs(error) / ground_truth_count) * 100

print(f"\nCounting Accuracy:")
print(f"  Ground Truth: {ground_truth_count}")
print(f"  Predicted: {predicted_count}")
print(f"  Error: {error}")
print(f"  Accuracy: {accuracy:.2f}%")
```

## Production Deployment Tips

```python
# For production, consider:

# 1. Error handling
try:
    results = model(frame, conf=0.5, classes=[0], verbose=False)[0]
except Exception as e:
    print(f"Detection error: {e}")
    continue

# 2. Logging
import logging
logging.basicConfig(filename='bag_counter.log', level=logging.INFO)
logging.info(f"Bag count: {line_zone.out_count} at {datetime.now()}")

# 3. Alert system
if abs(predicted_count - expected_count) > threshold:
    send_alert(f"Count discrepancy: {predicted_count} vs {expected_count}")

# 4. Performance monitoring
if fps < 15:
    print("Warning: Low FPS, consider optimization")
```

## Next Steps

1. Test with live Fillpac camera feed
2. Adjust counting line position based on conveyor layout
3. Integrate with production monitoring system
4. Set up automated alerts for anomalies
