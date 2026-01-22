"""
Offline Inference with Roboflow-Trained Model
Uses your trained model locally without internet connection
"""

from ultralytics import YOLO
import cv2
import supervision as sv
from pathlib import Path

class OfflineBagCounter:
    """Offline bag counter using locally stored model"""
    
    def __init__(self, model_path='models/weights/best.pt'):
        """
        Initialize with local model file
        
        Args:
            model_path: Path to your downloaded Roboflow model (.pt file)
        """
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        print("✓ Model loaded successfully (offline)")
    
    def detect_image(self, image_path, save_path=None, show=False):
        """
        Detect bags in a single image (offline)
        
        Args:
            image_path: Path to image
            save_path: Where to save result
            show: Display result
        """
        # Run detection locally
        results = self.model(image_path, conf=0.5, verbose=False)[0]
        
        # Count bags
        bag_count = len(results.boxes)
        
        print(f"\n{'='*60}")
        print(f"Image: {Path(image_path).name}")
        print(f"Bags detected: {bag_count}")
        print(f"{'='*60}")
        
        # Show detections
        for i, box in enumerate(results.boxes):
            conf = float(box.conf)
            bbox = box.xyxy[0].cpu().numpy()
            print(f"  Bag {i+1}: Confidence={conf:.3f}")
        
        # Save/show result
        annotated = results.plot()
        
        if save_path:
            cv2.imwrite(save_path, annotated)
            print(f"✓ Saved to: {save_path}")
        
        if show:
            cv2.imshow('Bag Detection (Offline)', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return bag_count, results
    
    def count_video(self, video_source, output_path=None, log_path=None):
        """
        Count bags in video with tracking (offline)
        
        Args:
            video_source: Video file path or camera index
            output_path: Save annotated video
            log_path: Save count log
        """
        # Open video
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open {video_source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nProcessing video: {width}x{height} @ {fps} FPS")
        print("Running completely offline...\n")
        
        # Setup tracking and counting
        tracker = sv.ByteTrack()
        line_zone = sv.LineZone(
            start=sv.Point(0, int(height * 0.6)),
            end=sv.Point(width, int(height * 0.6))
        )
        
        # Annotators
        box_annotator = sv.BoxAnnotator()
        line_annotator = sv.LineZoneAnnotator()
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Log file
        log_file = None
        if log_path:
            log_file = open(log_path, 'w')
            log_file.write("Timestamp,Frame,Count\n")
        
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect bags (offline)
                results = self.model(frame, conf=0.5, classes=[0], verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                
                # Track
                tracked = tracker.update_with_detections(detections)
                
                # Count
                line_zone.trigger(tracked)
                
                # Annotate
                annotated = box_annotator.annotate(frame.copy(), tracked)
                annotated = line_annotator.annotate(annotated, line_zone)
                
                # Add info
                cv2.putText(
                    annotated,
                    f"Bags: {line_zone.out_count} | Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Save
                if writer:
                    writer.write(annotated)
                
                # Log
                if log_file and frame_count % fps == 0:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"{timestamp},{frame_count},{line_zone.out_count}\n")
                
                # Display
                cv2.imshow('Offline Bag Counter', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if log_file:
                log_file.close()
            cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"Processing Complete (Offline)")
        print(f"{'='*60}")
        print(f"Total frames: {frame_count}")
        print(f"Total bags counted: {line_zone.out_count}")
        print(f"{'='*60}\n")
        
        return line_zone.out_count


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Offline bag detection and counting')
    parser.add_argument('--model', type=str, default='models/weights/best.pt',
                       help='Path to your Roboflow model (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                       help='Image/video path or camera index')
    parser.add_argument('--output', type=str, help='Save output video')
    parser.add_argument('--log', type=str, help='Save count log')
    parser.add_argument('--show', action='store_true', help='Display results')
    
    args = parser.parse_args()
    
    # Initialize offline counter
    counter = OfflineBagCounter(args.model)
    
    # Check if source is image or video
    if args.source.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Image detection
        counter.detect_image(args.source, args.output, args.show)
    else:
        # Video counting
        counter.count_video(args.source, args.output, args.log)


if __name__ == '__main__':
    main()
