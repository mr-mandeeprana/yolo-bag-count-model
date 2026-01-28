"""
Image Inference Script for YOLO Bag Detection
Detect and count bags in static images
"""

import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml

# OpenCV for image I/O and display
# Command-line argument parsing
# Modern file path handling
# YOLOv8 framework
# YAML file parsing (not used in current version)

class BagDetectorImage:
    """Bag detector for static images"""
    
    def __init__(self, weights_path: str, conf_threshold: float = 0.5):
        """
        Initialize detector
        
        Args:
            weights_path: Path to trained YOLO weights
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        
    def detect(self, image_path: str, save_path: str = None, show: bool = False):
        """
        Detect bags in an image
        
        Args:
            image_path: Path to input image
            save_path: Path to save annotated image (optional)
            show: Display image (default: False)
            
        Returns:
            Number of detected bags and detection results
        """
        # Run inference
        results = self.model(image_path, conf=self.conf_threshold)[0]
        
        # Count bags (class 0)
        bag_count = sum(1 for det in results.boxes if int(det.cls) == 0)
        
        # Get annotated image
        annotated_img = results.plot()
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, annotated_img)
            print(f"✓ Saved annotated image to: {save_path}")
        
        # Display if requested
        if show:
            cv2.imshow('Bag Detection', annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Print detection summary
        print(f"\n{'='*60}")
        print(f"Detection Results for: {Path(image_path).name}")
        print(f"{'='*60}")
        print(f"Total bags detected: {bag_count}")
        print(f"Confidence threshold: {self.conf_threshold}")
        
        # Print individual detections
        for i, det in enumerate(results.boxes):
            if int(det.cls) == 0:  # bag class
                conf = float(det.conf)
                bbox = det.xyxy[0].cpu().numpy()
                print(f"  Bag {i+1}: Confidence={conf:.3f}, BBox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        
        print(f"{'='*60}\n")
        
        return bag_count, results
    
    def batch_detect(self, images_dir: str, output_dir: str = None):
        """
        Detect bags in multiple images
        
        Args:
            images_dir: Directory containing images
            output_dir: Directory to save results (optional)
            
        Returns:
            Dictionary of image names to bag counts
        """
        images_path = Path(images_dir)
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        results_dict = {}
        
        print(f"\nProcessing {len(image_files)} images...")
        
        for img_file in image_files:
            save_path = None
            if output_dir:
                save_path = str(Path(output_dir) / f"detected_{img_file.name}")
            
            count, _ = self.detect(str(img_file), save_path=save_path, show=False)
            results_dict[img_file.name] = count
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Batch Detection Summary")
        print(f"{'='*60}")
        print(f"Total images processed: {len(results_dict)}")
        print(f"Total bags detected: {sum(results_dict.values())}")
        print(f"Average bags per image: {sum(results_dict.values()) / len(results_dict):.2f}")
        print(f"{'='*60}\n")
        
        return results_dict


def main():
    """Main inference script"""
    parser = argparse.ArgumentParser(description='Detect bags in images using YOLO')
    parser.add_argument('--weights', type=str, default='models/weights/best.pt',
                       help='Path to trained weights')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image or directory of images')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--save', type=str, help='Path to save annotated image(s)')
    parser.add_argument('--show', action='store_true', help='Display results')
    parser.add_argument('--batch', action='store_true', help='Process directory of images')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = BagDetectorImage(args.weights, args.conf)
    
    # Single image or batch
    if args.batch or Path(args.source).is_dir():
        detector.batch_detect(args.source, args.save)
    else:
        detector.detect(args.source, args.save, args.show)


if __name__ == '__main__':
    main()
