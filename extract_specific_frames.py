import cv2
import os
import re
from pathlib import Path

def extract_missing_frames():
    root = Path(r"c:\Users\mrman\OneDrive\Desktop\Beumer work\Beumer Data\Yolo_bag_count_model")
    video_path = str(root / "Recording 2026-03-11 151245.mp4")
    raw_dir = root / "data" / "raw" / "extracted_frames"
    labels_dir = root / "data" / "processed" / "labels"
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    # Find all frame indices from label names
    frame_indices = set()
    label_files = list(labels_dir.rglob("*.txt"))
    for lbl in label_files:
        # Match f000000 etc
        match = re.search(r'_f(\d{6})', lbl.name)
        if match:
            frame_indices.add(int(match.group(1)))
    
    print(f"Total unique frames requested by labels: {len(frame_indices)}")
    
    # Check which ones are missing
    missing_indices = []
    for idx in sorted(frame_indices):
        frame_name = f"Recording_2026-03-11_105738_f{idx:06d}.jpg"
        if not (raw_dir / frame_name).exists():
            missing_indices.append(idx)
            
    if not missing_indices:
        print("No missing frames to extract.")
        return
        
    print(f"Extracting {len(missing_indices)} missing frames...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video.")
        return
        
    for idx in missing_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_name = f"Recording_2026-03-11_105738_f{idx:06d}.jpg"
            out_path = str(raw_dir / frame_name)
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  Extracted {frame_name}")
        else:
            print(f"  [Error] Failed to read frame {idx}")
            
    cap.release()
    print("Extraction complete.")

if __name__ == "__main__":
    extract_missing_frames()
