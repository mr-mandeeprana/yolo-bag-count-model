"""
Diagnostic script to verify detection quality.
Saves crops of detected bags to a directory so we can see what the model is actually seeing.
"""

from ultralytics import YOLO
import cv2
import supervision as sv
import os
from pathlib import Path

# Setup
weights = 'models/weights/best.pt'
video_path = "WhatsApp Video 2026-01-20 at 12.38.18.mp4"
output_dir = "debug_detections"
os.makedirs(output_dir, exist_ok=True)

model = YOLO(weights)
cap = cv2.VideoCapture(video_path)

# Extract 10 random detections with confidence > 0.3
count = 0
frame_idx = 0

print("Extracting detection samples...")

while cap.isOpened() and count < 20:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    if frame_idx % 50 != 0: # Check every 50 frames to get variety
        continue
        
    results = model(frame, conf=0.3, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    for i, (xyxy, mask, confidence, class_id, tracker_id, data) in enumerate(detections):
        x1, y1, x2, y2 = map(int, xyxy)
        # Crop with some padding
        h, w = frame.shape[:2]
        pad = 20
        crop = frame[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
        
        if crop.size > 0:
            filename = f"{output_dir}/det_{count}_frame_{frame_idx}_conf_{confidence:.2f}.jpg"
            cv2.imwrite(filename, crop)
            print(f"Saved: {filename}")
            count += 1
            if count >= 20:
                break

cap.release()
print(f"\nExtracted {count} samples for verification.")
