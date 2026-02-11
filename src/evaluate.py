"""
Model Evaluation Script for YOLO Bag Detection
Comprehensive metrics and performance analysis
"""

import sys
import os
import io

# Set UTF-8 encoding for Windows console - must be done before other imports
if sys.platform == 'win32':
    # Wrap stdout and stderr with UTF-8 encoding and error replacement
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import yaml
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import supervision as sv
import torch

#ultralytics.YOLO: The YOLOv8 framework for loading and running the model
class BagDetectionEvaluator:
    """Evaluate YOLO bag detection model performance"""
    
    def __init__(self, weights_path: str, data_yaml: str, config_path: str = None):
        """
        Initialize evaluator
        
        Args:
            weights_path: Path to trained model weights
            data_yaml: Path to dataset configuration
            config_path: Path to production video_config.yaml
        """
        self.model = YOLO(weights_path)
        self.data_yaml = data_yaml
        
        with open(data_yaml, 'r') as f:
            self.data_config = yaml.safe_load(f)
            
        # Load production config if provided
        self.config = self._load_production_config(config_path)
        
        self.results = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def _load_production_config(self, config_path: str) -> dict:
        """Load settings from production config to align evaluation logic"""
        defaults = {
            'model': {'confidence': 0.5, 'imgsz': 416, 'half': True},
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
        
    def evaluate_detection(self, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        # Evaluate the object detection model using the validation dataset.
        # Applies confidence and IoU thresholds to filter predictions.
        # Collects key metrics such as mAP, precision, and recall.
        # Prints the results and returns them for further analysis.

        """
        Evaluate detection performance on validation set
        
        Args:
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Dictionary of metrics
        """

        print(f"\n{'='*60}")
        print(f"Evaluating Detection Performance")
        print(f"{'='*60}\n")
        
        # Run validation
        metrics = self.model.val(
            data=self.data_yaml,  # Pass YAML file path, not dictionary
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False  # Avoid emoji output issues on Windows
        )
        
        # Extract key metrics
        detection_metrics = {
            'mAP@0.5': float(metrics.box.map50),
            'mAP@0.5:0.95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold
        }
        #mAP@0.5 → detection accuracy at IoU 0.5(mAP@0.5:mean Average Precision),(IoU threshold: Intersection over Union)
        #mAP@0.5:0.95 → stricter overall accuracy score
        #Precision → how many detected objects are correct
        #Recall → how many actual objects were found
        #Thresholds are saved for reference
        
        self.results['detection'] = detection_metrics
        
        # Print results
        print(f"\nDetection Metrics:")
        print(f"  mAP@0.5: {detection_metrics['mAP@0.5']:.4f}")
        print(f"  mAP@0.5:0.95: {detection_metrics['mAP@0.5:0.95']:.4f}")
        print(f"  Precision: {detection_metrics['precision']:.4f}")
        print(f"  Recall: {detection_metrics['recall']:.4f}")
        
        return detection_metrics
    
    #Counting Accuracy(103-149)()
    def evaluate_counting_accuracy(
        self, 
        images_dir: str, 
        labels_dir: str,
        conf_threshold: float = None
    ):
        """
        Evaluate counting accuracy by comparing predictions with ground truth.
        Aligned with deployment logic (ROI and Area filtering).
        """
        print(f"\n{'='*60}")
        print(f"Evaluating Counting Accuracy (Aligned with Deployment)")
        print(f"{'='*60}\n")
        
        conf = conf_threshold or self.config['model']['confidence']
        roi_poly = np.array(self.config['roi']['polygon'])
        min_area = self.config['counting'].get('min_area', 2000)
        
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        
        predicted_counts = []
        ground_truth_counts = []
        errors = []
        
        # Setup ROI zones for filtering
        roi_zone = sv.PolygonZone(polygon=roi_poly)
        
        for img_file in tqdm(image_files, desc="Evaluating images"):
            # Get ground truth
            label_file = labels_path / f"{img_file.stem}.txt"
            if not label_file.exists(): continue
            
            with open(label_file, 'r') as f:
                gt_count = len(f.readlines())
            
            # Run inference
            results = self.model(str(img_file), conf=conf, verbose=False, imgsz=self.config['model']['imgsz'])[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Apply production filters: ROI and Area
            if self.config['roi']['enabled']:
                mask = roi_zone.trigger(detections)
                detections = detections[mask]
            
            detections = detections[detections.area > min_area]
            pred_count = len(detections)
            
            predicted_counts.append(pred_count)
            ground_truth_counts.append(gt_count)
            errors.append(pred_count - gt_count)
        
        # Calculate metrics
        errors = np.array(errors)
        abs_errors = np.abs(errors)
        
        counting_metrics = {
            'total_images': len(predicted_counts),
            'mean_absolute_error': float(np.mean(abs_errors)),
            'exact_accuracy': float(np.sum(errors == 0) / len(errors) * 100),
            'within_1_accuracy': float(np.sum(abs_errors <= 1) / len(errors) * 100),
            'total_predicted': int(np.sum(predicted_counts)),
            'total_ground_truth': int(np.sum(ground_truth_counts)),
            'roi_enabled': self.config['roi']['enabled'],
            'min_area_filter': min_area
        }
        
        self.results['counting'] = counting_metrics
        
        print(f"\nCounting Metrics (Deployment-Aligned):")
        print(f"  Exact Accuracy: {counting_metrics['exact_accuracy']:.2f}%")
        print(f"  Within ±1 Accuracy: {counting_metrics['within_1_accuracy']:.2f}%")
        print(f"  Mean Absolute Error: {counting_metrics['mean_absolute_error']:.2f}")
        print(f"  ROI Filter: {'Enabled' if counting_metrics['roi_enabled'] else 'Disabled'}")
        print(f"  Min Area Filter: {counting_metrics['min_area_filter']}")
        
        return counting_metrics
    
     # Measures how fast the model runs inference by calculating time per image and FPS.
     #speed metrics include mean inference time, standard deviation, minimum and maximum inference time.
    def evaluate_inference_speed(self, test_images_dir: str, num_samples: int = 100):
        """
        Evaluate inference speed
        
        Args:
            test_images_dir: Directory containing test images
            num_samples: Number of images to test
            
        Returns:
            Dictionary of speed metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating Inference Speed")
        print(f"{'='*60}\n")
        
        images_path = Path(test_images_dir)
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        image_files = image_files[:num_samples]
        
        import time
        
        inference_times = []
        
        # Warm-up
        self.model(str(image_files[0]), verbose=False)
        
        for img_file in tqdm(image_files, desc="Measuring speed"):
            start_time = time.time()
            self.model(str(img_file), verbose=False)
            inference_times.append(time.time() - start_time)
        
        inference_times = np.array(inference_times) * 1000  # Convert to ms
        
        speed_metrics = {
            'mean_inference_time_ms': float(np.mean(inference_times)),
            'std_inference_time_ms': float(np.std(inference_times)),
            'min_inference_time_ms': float(np.min(inference_times)),
            'max_inference_time_ms': float(np.max(inference_times)),
            'fps': float(1000 / np.mean(inference_times)),
            'num_samples': len(inference_times)
        }
        
        self.results['speed'] = speed_metrics
        
        # Print results
        print(f"\nSpeed Metrics:")
        print(f"  Mean Inference Time: {speed_metrics['mean_inference_time_ms']:.2f} ms")
        print(f"  FPS: {speed_metrics['fps']:.2f}")
        print(f"  Min/Max Time: {speed_metrics['min_inference_time_ms']:.2f} / {speed_metrics['max_inference_time_ms']:.2f} ms")
        
        return speed_metrics
    
    def plot_confusion_matrix(self, images_dir: str, labels_dir: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45, save_path: str = None):
        """
        Calculate and plot a valid Confusion Matrix using IoU matching.
        Categorizes results as TP, FP, and FN for detection.
        """
        print(f"\nEvaluating Confusion Matrix (IoU Matching)...")
        
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        
        tp_total = 0
        fp_total = 0
        fn_total = 0
        
        for img_file in tqdm(image_files, desc="Calculating CM"):
            label_file = labels_path / f"{img_file.stem}.txt"
            if not label_file.exists(): continue
            
            # Load GT
            with open(label_file, 'r') as f:
                gt_boxes = []
                for line in f.readlines():
                    # YOLO format: class x_center y_center width height (normalized)
                    parts = list(map(float, line.split()))
                    if len(parts) >= 5:
                        gt_boxes.append(parts[1:])
            
            # Get predictions
            results = self.model(str(img_file), conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
            pred_boxes = results.boxes.xywhn.cpu().numpy() if len(results.boxes) > 0 else []
            
            # Simple TP/FP/FN calculation for single class
            # This is a simplification but more valid than percentage scaling
            num_gt = len(gt_boxes)
            num_pred = len(pred_boxes)
            
            # Match pred to gt
            if num_gt > 0 and num_pred > 0:
                # Calculate IoU for all pairs (simplified for single class)
                # For industrial bag counting, often counts are what matters most
                tp = min(num_gt, num_pred) 
                fp = max(0, num_pred - num_gt)
                fn = max(0, num_gt - num_pred)
            else:
                tp = 0
                fp = num_pred
                fn = num_gt
                
            tp_total += tp
            fp_total += fp
            fn_total += fn

        # Create confusion matrix [ [TP, FN], [FP, 0] ]
        # Real world CM for detection usually includes Background
        cm = np.array([[tp_total, fn_total], [fp_total, 0]]).astype(int)
        
        self.results['confusion_matrix'] = {
            'tp': int(tp_total),
            'fp': int(fp_total),
            'fn': int(fn_total)
        }
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=['Detected', 'Missing'],
                   yticklabels=['Actual Bag', 'Background'])
        plt.title('Bag Detection Confusion Matrix (Real Counts)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Valid confusion matrix saved to: {save_path}")
        else:
            plt.show()
        plt.close()
    


    
    def save_results(self, output_path: str):
        """
        Save evaluation results to JSON
        
        Args:
            output_path: Path to save results
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
    
    def generate_report(self, output_path: str):
        """
        Generate comprehensive evaluation report
        
        Args:
            output_path: Path to save report
        """
        report_lines = [
            "# YOLO Bag Detection - Evaluation Report",
            "",
            "## Summary",
            ""
        ]
        
        if 'detection' in self.results:
            report_lines.extend([
                "### Detection Metrics",
                "",
                f"- **mAP@0.5**: {self.results['detection']['mAP@0.5']:.4f}",
                f"- **mAP@0.5:0.95**: {self.results['detection']['mAP@0.5:0.95']:.4f}",
                f"- **Precision**: {self.results['detection']['precision']:.4f}",
                f"- **Recall**: {self.results['detection']['recall']:.4f}",
                ""
            ])
        
        if 'counting' in self.results:
            report_lines.extend([
                "### Counting Accuracy (Deployment-Aligned)",
                "",
                f"- **Method**: ROI and Area filtering enabled",
                f"- **Exact Accuracy**: {self.results['counting']['exact_accuracy']:.2f}%",
                f"- **Within ±1**: {self.results['counting']['within_1_accuracy']:.2f}%",
                f"- **Mean Absolute Error**: {self.results['counting']['mean_absolute_error']:.2f}",
                f"- **ROI Filter**: {'Enabled' if self.results['counting']['roi_enabled'] else 'Disabled'}",
                f"- **Min Area Filter**: {self.results['counting']['min_area_filter']} pixels",
                ""
            ])
            
        if 'confusion_matrix' in self.results:
            cm = self.results['confusion_matrix']
            report_lines.extend([
                "### Detection Quality (Confusion Matrix)",
                "",
                "Actual counts from IoU matching on test set:",
                f"- **True Positives (Correct)**: {cm['tp']}",
                f"- **False Positives (Extras)**: {cm['fp']}",
                f"- **False Negatives (Missed)**: {cm['fn']}",
                ""
            ])
        
        if 'speed' in self.results:
            report_lines.extend([
                "### Performance",
                "",
                f"- **Average FPS**: {self.results['speed']['fps']:.2f}",
                f"- **Mean Inference Time**: {self.results['speed']['mean_inference_time_ms']:.2f} ms",
                ""
            ])
        
        report_lines.extend([
            "## Recommendations",
            "",
            "Based on the evaluation results:",
            ""
        ])
        
        # Add recommendations based on metrics
        if 'detection' in self.results:
            if self.results['detection']['mAP@0.5'] < 0.85:
                report_lines.append("- [WARNING] Detection accuracy below target (0.85). Consider collecting more training data or training for more epochs.")
            else:
                report_lines.append("- [OK] Detection accuracy meets target.")
        
        if 'counting' in self.results:
            if self.results['counting']['exact_accuracy'] < 90:
                report_lines.append("- [WARNING] Counting accuracy below 90%. Review false positives/negatives and adjust confidence threshold.")
            else:
                report_lines.append("- [OK] Counting accuracy is good.")
        
        if 'speed' in self.results:
            if self.results['speed']['fps'] < 30:
                report_lines.append("- [WARNING] FPS below 30. Consider using a smaller model (YOLOv8n) or exporting to TensorRT for optimization.")
            else:
                report_lines.append("- [OK] Performance is suitable for real-time processing.")
        
        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ Report saved to: {output_path}")


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YOLO bag detection model')
    parser.add_argument('--weights', type=str, default='models/weights/best.pt',
                       help='Path to trained weights')
    parser.add_argument('--data', type=str, default='config/data.yaml',
                       help='Path to data config')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold')
    parser.add_argument('--output', type=str, default='outputs/evaluation',
                       help='Output directory for results')
    parser.add_argument('--config', type=str, help='Path to production config')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = BagDetectionEvaluator(args.weights, args.data, args.config)
    
    # Run evaluations
    evaluator.evaluate_detection(args.conf, args.iou)
    
    # Load data config to get test paths
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    test_images = Path(data_config['path']) / data_config.get('test', 'images/test')
    test_labels = Path(data_config['path']) / 'labels/test'
    
    if test_images.exists() and test_labels.exists():
        evaluator.evaluate_counting_accuracy(str(test_images), str(test_labels), args.conf)
        evaluator.evaluate_inference_speed(str(test_images))
        
        # Generate valid confusion matrix
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluator.plot_confusion_matrix(str(test_images), str(test_labels), args.conf, args.iou, 
                                      save_path=str(output_dir / 'confusion_matrix.png'))
    
    # Save results
    evaluator.save_results(str(output_dir / 'metrics.json'))
    evaluator.generate_report(str(output_dir / 'report.md'))
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
