"""
Simple Frame Extractor for Fillpac Videos
Extracts frames from videos for training data preparation
"""

import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, frame_interval=30):
    """
    Extract frames from video
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_interval: Extract every Nth frame (30 = 1 per second at 30fps)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo: {Path(video_path).name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")
    print(f"  Extracting every {frame_interval} frames...")
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    video_name = Path(video_path).stem
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = output_path / f"{video_name}_frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"  Extracted {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    print(f"✓ Extracted {saved_count} frames to {output_dir}\n")
    return saved_count


if __name__ == '__main__':
    print("="*60)
    print("Fillpac Video Frame Extractor")
    print("="*60)
    
    # Video files
    video1 = "WhatsApp Video 2026-01-20 at 12.38.18.mp4"
    video2 = "WhatsApp Video 2026-01-20 at 14.12.53.mp4"
    
    # Output directory
    output_dir = "data/raw/extracted_frames"
    
    # Extract frames from both videos
    total_frames = 0
    
    if os.path.exists(video1):
        frames1 = extract_frames(video1, output_dir, frame_interval=30)
        total_frames += frames1
    else:
        print(f"Warning: {video1} not found")
    
    if os.path.exists(video2):
        frames2 = extract_frames(video2, output_dir, frame_interval=30)
        total_frames += frames2
    else:
        print(f"Warning: {video2} not found")
    
    print("="*60)
    print(f"Total frames extracted: {total_frames}")
    print("="*60)
    print("\nNext steps:")
    print("1. Go to https://roboflow.com and create a free account")
    print("2. Create a new project for 'Fillpac Bag Detection'")
    print(f"3. Upload images from: {output_dir}")
    print("4. Draw bounding boxes around bags (label as 'bag')")
    print("5. Export in YOLOv8 format")
    print("6. Download and extract to data/processed/")
    print("\nThen you can train: python src/train.py --data config/data.yaml")
