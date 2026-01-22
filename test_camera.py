import cv2
import argparse
import time

def test_camera(source):
    print(f"Connecting to: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("\n[ERROR] Could not connect to camera.")
        print("1. Check if the IP address is correct.")
        print("2. Verify username and password.")
        print("3. Ensure the camera and PC are on the same network.")
        return

    print("\n[SUCCESS] Connected! Opening live preview...")
    print("Press 'q' to close the window.")
    
    # Calculate FPS for display
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Lost connection to camera.")
            break
            
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Add FPS text
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Camera Connection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Test finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="RTSP URL or Camera Index")
    args = parser.parse_args()
    test_camera(args.source)
