#!/usr/bin/env python3
"""
Diagnostic script to analyze what's being detected and filtered.
Helps identify why bags aren't being counted.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
import supervision as sv
import yaml

def diagnose_video(video_path: str, config_path: str = "config/video_config.yaml", max_frames: int = 100):
    """Analyze detections on a video file"""
    
    print(f"📊 Analyzing video: {video_path}")
    print("=" * 70)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model - use best.pt if openvino model fails
    weights = config['model'].get('weights', 'models/weights/best.pt')
    if 'openvino' in weights.lower():
        weights = 'models/weights/best.pt'  # Fallback to regular PyTorch
    confidence = config['model'].get('confidence', 0.5)
    imgsz = config['model'].get('imgsz', 416)
    
    print(f"\n🔧 Configuration:")
    print(f"   - Weights: {weights}")
    print(f"   - Confidence threshold: {confidence}")
    print(f"   - Image size: {imgsz}")
    
    # Load YOLO model
    try:
        model = YOLO(weights)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Get class info
    names = getattr(model, 'names', {})
    print(f"   - Classes: {names}")
    
    # Counting config
    min_area = config['counting'].get('min_area', 500)
    max_area = config['counting'].get('max_area', None)
    min_ratio = config['counting'].get('min_aspect_ratio')
    max_ratio = config['counting'].get('max_aspect_ratio')
    
    print(f"\n📏 Filtering thresholds:")
    print(f"   - Min area: {min_area} pixels")
    print(f"   - Max area: {max_area if max_area else 'unlimited'} pixels")
    print(f"   - Aspect ratio: {min_ratio} - {max_ratio}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n📹 Video properties:")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - FPS: {fps}")
    
    # Analyze frames
    print(f"\n🔍 Analyzing {max_frames} frames...")
    print("-" * 70)
    
    stats = {
        'total_raw_detections': 0,
        'low_conf': 0,
        'passed_conf': 0,
        'failed_area_small': 0,
        'failed_area_large': 0,
        'failed_aspect': 0,
        'final_valid': 0,
        'sample_areas': [],
        'sample_confidences': []
    }
    
    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=0.10, verbose=False, imgsz=imgsz)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        if len(detections) == 0:
            frame_idx += 1
            continue
        
        stats['total_raw_detections'] += len(detections)
        
        # Filter by confidence
        conf_mask = detections.confidence >= confidence
        stats['passed_conf'] += sum(conf_mask)
        stats['low_conf'] += sum(~conf_mask)
        
        valid_conf = detections[conf_mask]
        if len(valid_conf) == 0:
            frame_idx += 1
            continue
        
        # Filter by area
        areas = valid_conf.area
        area_mask = areas > min_area
        stats['failed_area_small'] += sum(~area_mask)
        
        if max_area:
            area_mask = area_mask & (areas < max_area)
            stats['failed_area_large'] += sum((areas >= max_area))
        
        valid_area = valid_conf[area_mask]
        if len(valid_area) == 0:
            frame_idx += 1
            continue
        
        # Filter by aspect ratio
        if min_ratio or max_ratio:
            xyxy = valid_area.xyxy
            widths = np.maximum(xyxy[:, 2] - xyxy[:, 0], 1.0)
            heights = np.maximum(xyxy[:, 3] - xyxy[:, 1], 1.0)
            aspect = widths / heights
            
            aspect_mask = np.ones(len(aspect), dtype=bool)
            if min_ratio:
                aspect_mask = aspect_mask & (aspect >= min_ratio)
            if max_ratio:
                aspect_mask = aspect_mask & (aspect <= max_ratio)
            
            stats['failed_aspect'] += sum(~aspect_mask)
            valid_area = valid_area[aspect_mask]
        
        stats['final_valid'] += len(valid_area)
        
        # Sample areas and confidences
        if len(valid_area) > 0:
            stats['sample_areas'].extend(valid_area.area.tolist())
            stats['sample_confidences'].extend(valid_area.confidence.tolist())
        
        frame_idx += 1
    
    cap.release()
    
    # Print results
    print(f"\n📊 DETECTION STATISTICS (over {frame_idx} frames):")
    print("-" * 70)
    print(f"Total raw detections:     {stats['total_raw_detections']}")
    print(f"Passed confidence filter: {stats['passed_conf']}")
    print(f"Failed confidence filter: {stats['low_conf']}")
    print(f"Failed min area filter:   {stats['failed_area_small']}")
    if max_area:
        print(f"Failed max area filter:   {stats['failed_area_large']}")
    print(f"Failed aspect ratio:      {stats['failed_aspect']}")
    print(f"Final valid detections:   {stats['final_valid']}")
    
    if stats['final_valid'] == 0:
        print("\n⚠️  NO BAGS BEING DETECTED!")
        print("\nPossible causes:")
        if stats['low_conf'] > stats['passed_conf']:
            print("  1. ❌ Confidence threshold ({}) is TOO HIGH".format(confidence))
            if stats['sample_confidences']:
                avg_conf = sum(stats['sample_confidences']) / len(stats['sample_confidences'])
                print(f"     → Average detected confidence: {avg_conf:.3f}")
        if stats['failed_area_small'] > 0:
            print("  2. ❌ Min area threshold ({}) is TOO HIGH".format(min_area))
            if stats['sample_areas']:
                avg_area = sum(stats['sample_areas']) / len(stats['sample_areas'])
                print(f"     → Average detected area: {avg_area:.0f} pixels")
        if max_area and stats['failed_area_large'] > 0:
            print(f"  3. ❌ Max area threshold ({max_area}) is TOO LOW or bags are large")
        if stats['failed_aspect'] > 0:
            print(f"  4. ❌ Aspect ratio limits ({min_ratio}-{max_ratio}) are TOO STRICT")
    else:
        print(f"\n✅ {stats['final_valid']} bags detected! Counting should work.")
        if stats['sample_areas']:
            avg_area = sum(stats['sample_areas']) / len(stats['sample_areas'])
            print(f"   Average bag area: {avg_area:.0f} pixels")
        if stats['sample_confidences']:
            avg_conf = sum(stats['sample_confidences']) / len(stats['sample_confidences'])
            print(f"   Average confidence: {avg_conf:.3f}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    # Try to find a video file
    video_files = list(Path(".").glob("*.mp4")) + list(Path(".").glob("Recording*.mp4"))
    
    if not video_files:
        print("❌ No .mp4 files found in current directory")
        print("Usage: python diagnose_detection.py <video_path>")
        sys.exit(1)
    
    video_path = str(video_files[0])
    print(f"Using video: {video_path}\n")
    diagnose_video(video_path, max_frames=100)
