"""
extract_frames.py  –  Extract frames from a video for YOLO training data.

Usage:
    python scripts/extract_frames.py --video "Recording 2026-03-11 105738.mp4"
    python scripts/extract_frames.py --video "Recording 2026-03-11 105738.mp4" --fps 1 --out data/frames
"""

import argparse
import cv2
import os
import sys

def extract_frames(video_path: str, out_dir: str, target_fps: float, max_frames: int):
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        sys.exit(1)

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration  = total_frm / src_fps

    print(f"\n{'='*55}")
    print(f"  Video      : {os.path.basename(video_path)}")
    print(f"  Resolution : {width}x{height}  |  FPS: {src_fps:.1f}")
    print(f"  Duration   : {duration:.1f}s  |  Total frames: {total_frm}")
    print(f"  Extract at : {target_fps} fps  ->  ~{int(duration * target_fps)} frames")
    print(f"  Output dir : {out_dir}")
    print(f"{'='*55}\n")

    os.makedirs(out_dir, exist_ok=True)

    interval   = max(1, int(src_fps / target_fps))
    frame_idx  = 0
    saved      = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0].replace(" ", "_")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            fname = os.path.join(out_dir, f"{video_name}_f{frame_idx:06d}.jpg")
            cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            if saved % 20 == 0:
                print(f"  Saved {saved} frames...", end="\r")
            if max_frames and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"\n[DONE] Extracted {saved} frames -> {out_dir}")
    print(f"       Next step: annotate with LabelImg or Roboflow, then train.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video for YOLO training.")
    parser.add_argument("--video",      required=True,  help="Path to input video file")
    parser.add_argument("--out",        default="data/training_frames", help="Output directory (default: data/training_frames)")
    parser.add_argument("--fps",        type=float, default=2.0, help="Frames to extract per second (default: 2)")
    parser.add_argument("--max-frames", type=int,   default=0,   help="Max frames to extract (0 = all)")
    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        out_dir=args.out,
        target_fps=args.fps,
        max_frames=args.max_frames,
    )
