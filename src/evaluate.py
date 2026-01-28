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

#ultralytics.YOLO: The YOLOv8 framework for loading and running the model
class BagDetectionEvaluator:
    """Evaluate YOLO bag detection model performance"""
    
    def __init__(self, weights_path: str, data_yaml: str):
        """
        Initialize evaluator
        
        Args:
            weights_path: Path to trained model weights
            data_yaml: Path to dataset configuration
        """
        self.model = YOLO(weights_path)
        self.data_yaml = data_yaml  # Store the YAML path
        
        with open(data_yaml, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.results = {}
        
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
        conf_threshold: float = 0.5
    ):
        """
        Evaluate counting accuracy by comparing predictions with ground truth
        
        Args:
            images_dir: Directory containing test images
            labels_dir: Directory containing ground truth labels
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary of counting metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating Counting Accuracy")
        print(f"{'='*60}\n")
        
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        
        predicted_counts = []
        ground_truth_counts = []
        errors = []
        
        for img_file in tqdm(image_files, desc="Processing images"):
            # Get ground truth count
            label_file = labels_path / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue
            
            with open(label_file, 'r') as f:
                gt_count = len(f.readlines())
            
            # Get predicted count
            results = self.model(str(img_file), conf=conf_threshold, verbose=False)[0]
            pred_count = sum(1 for det in results.boxes if int(det.cls) == 0)
            
            predicted_counts.append(pred_count)
            ground_truth_counts.append(gt_count)
            errors.append(pred_count - gt_count)
        

        # Calculate metrics
        errors = np.array(errors)
        abs_errors = np.abs(errors)
        
        counting_metrics = {
            'total_images': len(predicted_counts), #Number of images evaluated
            'mean_error': float(np.mean(errors)), #Average counting error
            'mean_absolute_error': float(np.mean(abs_errors)),#Average absolute 
            'std_error': float(np.std(errors)),#Variation in error
            'max_error': float(np.max(abs_errors)),#Maximum counting error
            'exact_accuracy': float(np.sum(errors == 0) / len(errors) * 100),#Exact accuracy
            'within_1_accuracy': float(np.sum(abs_errors <= 1) / len(errors) * 100),#Within 1 accuracy
            'within_2_accuracy': float(np.sum(abs_errors <= 2) / len(errors) * 100),#Within 2 accuracy
            'total_predicted': int(np.sum(predicted_counts)),
            'total_ground_truth': int(np.sum(ground_truth_counts)),
        }
        
        
        self.results['counting'] = counting_metrics
        
        # Print results
        print(f"\nCounting Metrics:")
        print(f"  Total images: {counting_metrics['total_images']}")
        print(f"  Mean Absolute Error: {counting_metrics['mean_absolute_error']:.2f}")
        print(f"  Exact Accuracy: {counting_metrics['exact_accuracy']:.2f}%")
        print(f"  Within ±1 Accuracy: {counting_metrics['within_1_accuracy']:.2f}%")
        print(f"  Within ±2 Accuracy: {counting_metrics['within_2_accuracy']:.2f}%")
        print(f"  Total Predicted: {counting_metrics['total_predicted']}")
        print(f"  Total Ground Truth: {counting_metrics['total_ground_truth']}")
        
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
    
    #Confusion Matrix
    def plot_confusion_matrix(self, save_path: str = None):
        # Correct detections (True Positives)
        # Wrong detections (False Positives)
        # Missed objects (False Negatives)
        """
        Plot confusion matrix (for detection: TP, FP, FN)
        
        Args:
            save_path: Path to save plot
        """
        if 'detection' not in self.results:
            print("Run evaluate_detection() first")
            return
        
        metrics = self.results['detection']
        precision = metrics['precision']
        recall = metrics['recall']
        
        # Calculate TP, FP, FN (simplified)
        # Assuming 100 ground truth objects
        tp = recall * 100
        fn = 100 - tp
        fp = (tp / precision) - tp if precision > 0 else 0
        
        # Create confusion matrix
        cm = np.array([[tp, fn], [fp, 0]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues', 
                   xticklabels=['Positive', 'Negative'],
                   yticklabels=['True', 'False'])
        plt.title('Detection Confusion Matrix (Simplified)')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    

    def plot_error_distribution(self, save_path: str = None):
       # This function is intended to plot the distribution of counting errors.
       # It checks whether counting evaluation has been done.
       # In this simplified version, individual error values are not stored,
       # so the actual error distribution plot cannot be generated.

        if 'counting' not in self.results:
            print("Run evaluate_counting_accuracy() first")
            return
        
        # This requires storing individual errors - simplified version
        print("Error distribution plot requires individual error data")
        print("Run evaluate_counting_accuracy() and store errors for detailed plot")
    
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
                "### Counting Accuracy",
                "",
                f"- **Exact Accuracy**: {self.results['counting']['exact_accuracy']:.2f}%",
                f"- **Within ±1**: {self.results['counting']['within_1_accuracy']:.2f}%",
                f"- **Mean Absolute Error**: {self.results['counting']['mean_absolute_error']:.2f}",
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
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = BagDetectionEvaluator(args.weights, args.data)
    
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
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator.save_results(str(output_dir / 'metrics.json'))
    evaluator.generate_report(str(output_dir / 'report.md'))
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
