import cv2
import os
import sys
import time
import argparse
import yaml
from pathlib import Path
from urllib.parse import urlparse


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

    # Parse URL components for dynamic error messages
    parsed = urlparse(url)

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
        print(f"- Network: Is the camera IP ({parsed.hostname}) reachable from this PC?")
        if parsed.username:
            print(f"- Credentials: Are credentials for user '{parsed.username}' correct?")
        print(f"- Firewall: Is port {parsed.port or 554} open on the camera and network?")
        return False

    print(f"SUCCESS: Connected in {time.time() - start_time:.2f} seconds.")

    # 2. Stream Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_str = f"{fps:.2f}" if fps > 0 else "Unknown (camera may not report FPS)"
    print(f"\n2. Stream Properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   Reported FPS: {fps_str}")

    # 3. Read Test (with retry)
    print("\n3. Testing Frame Read...")
    output_path = Path(__file__).parent.parent / "outputs" / "rtsp_sample.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    MAX_RETRIES = 5
    ret, frame = False, None
    for attempt in range(MAX_RETRIES):
        ret, frame = cap.read()
        if ret:
            break
        print(f"   Retry {attempt + 1}/{MAX_RETRIES}...")
        time.sleep(0.5)

    if ret:
        print("   ✓ Successfully read a frame.")
        cv2.imwrite(str(output_path), frame)
        print(f"   ✓ Saved sample frame to '{output_path}'")
    else:
        print("   ✗ Connected, but failed to read frames after retries (timeout or codec issue).")

    cap.release()
    print(f"\n{'='*60}")
    print("Diagnostic Complete.")
    print(f"{'='*60}\n")
    return ret


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

    success = test_rtsp_connection(target_url)
    sys.exit(0 if success else 1)
