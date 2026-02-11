import cv2
import os
import sys
import time
import argparse
import yaml
from pathlib import Path

def test_rtsp_connection(url):
    """
    Diagnostic tool to test RTSP stream connectivity and performance.
    """
    print(f"\n{'='*60}")
    print(f"RTSP Diagnostic Tool")
    print(f"{'='*60}")
    print(f"Target URL: {url}")
    
    # Clean the URL
    url = url.strip()
    
    # 1. Connectivity Check
    print("\n1. Testing Backend Connectivity...")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    
    start_time = time.time()
    # Explicitly use FFMPEG for RTSP
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        duration = time.time() - start_time
        print(f"FAILED: Could not open stream after {duration:.2f} seconds.")
        print("\nPossible causes:")
        print("- Network: Is the camera IP (192.168.1.5) reachable from this PC?")
        print("- Credentials: Is the username/password 'admin:Admin@123' correct?")
        print("- Firewall: Is port 554 open on the camera and network?")
        return False

    print(f"SUCCESS: Connected in {time.time() - start_time:.2f} seconds.")
    
    # 2. Stream Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n2. Stream Properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   Reported FPS: {fps}")
    
    # 3. Read Test
    print("\n3. Testing Frame Read...")
    ret, frame = cap.read()
    if ret:
        print("   ✓ Successfully read a frame.")
        # Save sample frame
        cv2.imwrite("rtsp_sample.jpg", frame)
        print("   ✓ Saved sample frame to 'rtsp_sample.jpg'")
    else:
        print("   ✗ Connected, but failed to read frames (timeout or codec issue).")
        
    cap.release()
    print(f"\n{'='*60}")
    print("Diagnostic Complete.")
    print(f"{'='*60}\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RTSP Camera Diagnostic')
    parser.add_argument('--url', type=str, help='RTSP URL to test')
    parser.add_argument('--config', type=str, default='config/video_config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    target_url = args.url
    if not target_url and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            target_url = cfg.get('camera', {}).get('source')
            
    if not target_url:
        print("Error: Provide --url or ensure camera.source is in config/video_config.yaml")
        sys.exit(1)
        
    test_rtsp_connection(target_url)
