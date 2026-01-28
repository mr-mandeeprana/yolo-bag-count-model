"""
Image Inference Script for YOLO Bag Detection
Detect and count bags in static images
"""

import cv2
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import yaml
from ultralytics import YOLO
import supervision as sv
import torch
import os


class BagDetectorImage:
    """
    Bag detector for static images with production-aligned filtering.
    Supports ROI zones and Area filtering to match deployment logic.
    """
    
    def __init__(self, weights_path: str, config_path: Optional[str] = None):
        """
        Initialize detector with production configuration.
        """
        self.config = self._load_config(config_path)
        
        # Override weights if provided
        final_weights = weights_path or self.config['model']['weights']
        
        if not os.path.exists(final_weights):
            raise FileNotFoundError(f"Model weights not found at: {final_weights}")
            
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(final_weights).to(self.device)
        self.conf_threshold = self.config['model']['confidence']
        
        # Setup ROI if enabled
        self.roi_zone = None
        if self.config['roi']['enabled']:
            self.roi_polygon = np.array(self.config['roi']['polygon'])
            self.roi_zone = sv.PolygonZone(polygon=self.roi_polygon)
            
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load settings from production config to align filtering logic"""
        defaults = {
            'model': {'weights': 'models/weights/best.pt', 'confidence': 0.5, 'imgsz': 416},
            'roi': {'enabled': True, 'polygon': [[0, 450], [0, 50], [478, 50], [478, 450]]},
            'counting': {'min_area': 2000}
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                for section in ['model', 'roi', 'counting']:
                    if section in user_config:
                        defaults[section].update(user_config[section])
            print(f"✓ Loaded production config from: {config_path}")
        
        return defaults
        
    def detect(self, image_path: str, save_path: str = None, show: bool = False) -> Tuple[int, sv.Detections]:
        """
        Detect bags in an image using production filters.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
            
        # Run inference
        results = self.model(image, conf=self.conf_threshold, verbose=False, imgsz=self.config['model']['imgsz'])[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Apply ROI filtering
        if self.roi_zone:
            mask = self.roi_zone.trigger(detections)
            detections = detections[mask]
            
        # Apply Area filtering
        min_area = self.config['counting'].get('min_area', 2000)
        detections = detections[detections.area > min_area]
        
        bag_count = len(detections)
        
        # Visualization
        annotated_img = image.copy()
        
        # Draw ROI if enabled
        if self.roi_zone:
            annotated_img = sv.draw_polygon(
                scene=annotated_img,
                polygon=self.roi_polygon,
                color=sv.Color.RED,
                thickness=2
            )
            
        # Annotate boxes and labels
        labels = [f"bag {conf:.2f}" for conf in detections.confidence]
        annotated_img = self.box_annotator.annotate(scene=annotated_img, detections=detections)
        annotated_img = self.label_annotator.annotate(scene=annotated_img, detections=detections, labels=labels)
        
        # Add summary text to image
        cv2.putText(annotated_img, f"Count: {bag_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, annotated_img)
            print(f"✓ Saved annotated image to: {save_path}")
        
        # Display if requested
        if show:
            window_name = f"Inference: {Path(image_path).name}"
            cv2.imshow(window_name, annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print(f"\nDetection Summary for: {Path(image_path).name}")
        print(f"  Count: {bag_count} bags")
        print(f"  Confidence Threshold: {self.conf_threshold}")
        print(f"  Min Area Filter: {min_area} pixels")
        print(f"  ROI Filter: {'Enabled' if self.roi_zone else 'Disabled'}")
        
        return bag_count, detections
    
    def batch_detect(self, images_dir: str, output_dir: Optional[str] = None) -> Dict[str, int]:
        """
        Detect bags in multiple images with summary reporting.
        """
        images_path = Path(images_dir)
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        
        if not image_files:
            print(f"Warning: No images found in {images_dir}")
            return {}
            
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        results_dict = {}
        total_bags = 0
        
        print(f"\nBatch processing {len(image_files)} images...")
        
        for img_file in tqdm(image_files, desc="Batch Inference"):
            save_path = None
            if output_dir:
                save_path = str(Path(output_dir) / f"detected_{img_file.name}")
            
            try:
                count, _ = self.detect(str(img_file), save_path=save_path, show=False)
                results_dict[img_file.name] = count
                total_bags += count
            except Exception as e:
                print(f"  Error processing {img_file.name}: {e}")
        
        # Final Summary
        print(f"\n{'='*60}")
        print(f"Batch Detection Summary")
        print(f"{'='*60}")
        print(f"  Total images processed: {len(results_dict)}")
        print(f"  Total bags detected: {total_bags}")
        if len(results_dict) > 0:
            print(f"  Average bags per image: {total_bags / len(results_dict):.2f}")
        print(f"{'='*60}\n")
        
        return results_dict


def main():
    """Main inference script"""
    parser = argparse.ArgumentParser(description='Detect bags in images using YOLO')
    parser.add_argument('--weights', type=str, help='Path to trained weights (overrides config)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image or directory of images')
    parser.add_argument('--config', type=str, default='config/video_config.yaml',
                       help='Path to production config')
    parser.add_argument('--save', type=str, help='Path to save annotated image(s)')
    parser.add_argument('--show', action='store_true', help='Display results')
    parser.add_argument('--batch', action='store_true', help='Process directory of images')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = BagDetectorImage(args.weights, args.config)
    
    # Single image or batch
    if args.batch or Path(args.source).is_dir():
        detector.batch_detect(args.source, args.save)
    else:
        detector.detect(args.source, args.save, args.show)


if __name__ == '__main__':
    main()
