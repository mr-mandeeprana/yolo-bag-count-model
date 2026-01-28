import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from src.inference_video import ThreadedCamera

def main():
    parser = argparse.ArgumentParser(description='Industrial Camera Calibration Tool')
    parser.add_argument('--source', type=str, required=True, help='Video source (RTSP/File)')
    args = parser.parse_args()

    # Target training dimensions
    TRAIN_W, TRAIN_H = 478, 850
    TRAIN_ASPECT = TRAIN_W / TRAIN_H

    # Initialize camera
    cap = ThreadedCamera(args.source)
    cap.start()
    
    # Wait for first frame to get dimensions
    while cap.frame_id == 0:
        time.sleep(0.1)
    
    ret, frame, _ = cap.read()
    if not ret:
        print("Failed to read from source.")
        return

    H, W = frame.shape[:2]
    
    # Calculate the "Sweet Spot" (where the 478x850 vertical slice would be in the source)
    # If source is 640x480 (Wide), the vertical slice will be scaled down
    target_h_in_source = H
    target_w_in_source = int(H * TRAIN_ASPECT)
    
    # Center the slice
    start_x = (W - target_w_in_source) // 2
    end_x = start_x + target_w_in_source

    print(f"Source: {W}x{H} | Calibration Zone: {target_w_in_source}x{target_h_in_source}")

    window_name = "Industrial Calibration Guide"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame, _ = cap.read()
        if not ret:
            break

        # Create Overlay
        overlay = frame.copy()
        
        # 1. Darken areas outside the training zone
        mask = np.zeros_like(frame)
        cv2.rectangle(mask, (start_x, 0), (end_x, H), (255, 255, 255), -1)
        frame = cv2.addWeighted(frame, 0.3, cv2.bitwise_and(frame, mask), 0.7, 0)

        # 2. Draw Training Zone Border
        cv2.rectangle(frame, (start_x, 0), (end_x, H), (0, 255, 255), 2)
        
        # 3. Draw Guideline Boxes (Typical bag sizes from training)
        # Large bag (close)
        cw, ch = target_w_in_source * 0.5, target_h_in_source * 0.2
        cx, cy = W // 2, H // 2
        cv2.rectangle(frame, (int(cx-cw/2), int(cy-ch/2)), (int(cx+cw/2), int(cy+ch/2)), (0, 255, 0), 1)
        cv2.putText(frame, "IDEAL BAG SIZE", (int(cx-cw/2), int(cy-ch/2)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 4. Instructions
        y_off = 30
        texts = [
            "CALIBRATION MODE",
            f"1. Position conveyor within YELLOW box",
            f"2. Align bags to match GREEN box size",
            f"3. Ensure the counting line (middle) is clear",
            "Press 'q' to exit"
        ]
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (20, y_off + i*25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
