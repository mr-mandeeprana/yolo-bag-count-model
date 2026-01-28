"""
Utility Functions for YOLO Bag Counting System
"""

import cv2
import numpy as np
import os
import random
import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from tqdm import tqdm


def visualize_dataset_sample(
    images_dir: str, 
    labels_dir: str, 
    num_samples: int = 5,
    save_path: str = None,
    shuffle: bool = True
):
    """
    Visualize annotated samples. Added random shuffling and robust path handling.
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
    if not image_files:
        print(f"⚠ No images found in {images_dir}")
        return
        
    if shuffle:
        random.shuffle(image_files)
        
    image_files = image_files[:num_samples]
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        h, w = img.shape[:2]
        
        # Read corresponding label
        label_file = labels_path / f"{img_file.stem}.txt"
        if not label_file.exists():
            continue
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Draw bounding boxes
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            
            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, 'bag', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display or save
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_dir / f"viz_{img_file.name}"), img)
        else:
            cv2.imshow(f'Sample: {img_file.name}', img)
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print(f"✓ Visualized {len(image_files)} samples")


def calculate_dataset_statistics(labels_dir: str) -> Dict:
    """
    Enhanced dataset statistics with bounding box area distributions.
    """
    labels_path = Path(labels_dir)
    label_files = list(labels_path.glob('*.txt'))
    
    if not label_files:
        print(f"⚠ No label files found in {labels_dir}")
        return {}
        
    total_images = len(label_files)
    total_bags = 0
    bags_per_image = []
    bbox_areas = []
    
    for label_file in tqdm(label_files, desc="Analyzing labels"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        num_bags = len(lines)
        total_bags += num_bags
        bags_per_image.append(num_bags)
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO: cls x y w h
                w, h = float(parts[3]), float(parts[4])
                bbox_areas.append(w * h)
    
    stats = {
        'total_images': total_images,
        'total_bags': total_bags,
        'avg_bags_per_image': float(np.mean(bags_per_image)),
        'max_bags_per_image': int(np.max(bags_per_image)),
        'avg_bbox_area_normalized': float(np.mean(bbox_areas)) if bbox_areas else 0,
        'q1_area': float(np.percentile(bbox_areas, 25)) if bbox_areas else 0,
        'q3_area': float(np.percentile(bbox_areas, 75)) if bbox_areas else 0,
    }
    
    print(f"\n{'='*60}")
    print(f"Industrial Dataset Statistics")
    print(f"{'='*60}")
    print(f"  Total Images: {stats['total_images']}")
    print(f"  Total Bags: {stats['total_bags']}")
    print(f"  Avg Bags/Img: {stats['avg_bags_per_image']:.2f}")
    print(f"  Avg Norm Area: {stats['avg_bbox_area_normalized']:.6f} (Helpful for min_area tuning)")
    print(f"{'='*60}\n")
    
    return stats


class DatasetQA:
    """
    Dataset Quality Assurance tool for industrial YOLO datasets.
    Finds corruptions, outliers, and invalid coordinates.
    """
    
    def __init__(self, images_dir: str, labels_dir: str):
        self.images_path = Path(images_dir)
        self.labels_path = Path(labels_dir)
        
    def run_integrity_check(self) -> Dict:
        """Scan for missing files and integrity issues"""
        print(f"\nRunning Integrity Check...")
        
        images = {f.stem for f in list(self.images_path.glob('*.jpg')) + list(self.images_path.glob('*.png'))}
        labels = {f.stem for f in self.labels_path.glob('*.txt')}
        
        missing_labels = images - labels
        orphaned_labels = labels - images
        
        # Check for empty files
        empty_labels = []
        for lf in self.labels_path.glob('*.txt'):
            if lf.stat().st_size == 0:
                empty_labels.append(lf.name)
        
        total_issues = len(missing_labels) + len(orphaned_labels) + len(empty_labels)
        
        report = {
            'images_count': len(images),
            'labels_count': len(labels),
            'missing_labels': list(missing_labels),
            'orphaned_labels': list(orphaned_labels),
            'empty_label_files': empty_labels,
            'status': 'OK' if total_issues == 0 else 'WARNING'
        }
        
        print(f"  Status: {report['status']}")
        print(f"  Images without labels: {len(missing_labels)}")
        print(f"  Labels without images: {len(orphaned_labels)}")
        print(f"  Empty labels: {len(empty_labels)}")
        
        return report

    def validate_coordinates(self) -> List[str]:
        """Scan for invalid YOLO coordinates (outside [0, 1])"""
        print(f"\nValidating YOLO Coordinates...")
        invalid_files = []
        
        for label_file in tqdm(list(self.labels_path.glob('*.txt')), desc="Validating boxes"):
            with open(label_file, 'r') as f:
                for ln, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    
                    coords = [float(x) for x in parts[1:]]
                    if any(c < 0 or c > 1 for c in coords):
                        invalid_files.append(f"{label_file.name} (Line {ln}: {coords})")
                        break
        
        print(f"  Found {len(invalid_files)} files with invalid coordinates.")
        return invalid_files

    def report_health(self):
        """Generate a full health report for the dataset"""
        print(f"\n{'#'*60}")
        print(f"DATASET HEALTH REPORT: {self.images_path.parent.name}")
        print(f"{'#'*60}")
        
        integrity = self.run_integrity_check()
        invalids = self.validate_coordinates()
        
        if integrity['status'] == 'OK' and not invalids:
            print(f"\n{Colors.OKGREEN}✓ Dataset is CLEAN and ready for training.{Colors.ENDC}")
        else:
            print(f"\n{Colors.FAIL}✖ Dataset has ISSUES that may cause training to fail.{Colors.ENDC}")
            print(f"  Tip: Run with --action cleanup to fix orphaned labels.")
            if invalids:
                print(f"  Critical: {len(invalids)} files have invalid coordinates.")

    def cleanup(self, backup_dir: Optional[str] = None):
        """Move orphaned labels to a backup directory to synchronize dataset"""
        print(f"\nStarting Dataset Cleanup...")
        
        # Identify orphans
        images = {f.stem for f in list(self.images_path.glob('*.jpg')) + list(self.images_path.glob('*.png'))}
        labels = {f.stem for f in self.labels_path.glob('*.txt')}
        
        orphaned_label_stems = labels - images
        
        if not orphaned_label_stems:
            print(f"  {Colors.OKGREEN}✓ No orphaned labels found. Cleanup skipped.{Colors.ENDC}")
            return
            
        # Setup backup
        if backup_dir is None:
            backup_dir = self.labels_path.parent / 'orphaned_backup'
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        print(f"  Moving {len(orphaned_label_stems)} orphaned labels to: {backup_path}")
        
        moved_count = 0
        for stem in orphaned_label_stems:
            label_file = self.labels_path / f"{stem}.txt"
            if label_file.exists():
                try:
                    shutil.move(str(label_file), str(backup_path / label_file.name))
                    moved_count += 1
                except Exception as e:
                    print(f"  Error moving {label_file.name}: {e}")
        
        print(f"\n{Colors.OKGREEN}✓ Successfully moved {moved_count} files.{Colors.ENDC}")
        print(f"  Dataset is now synchronized.")


class Colors:
    """Helper for console colors"""
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def create_video_from_images(images_dir: str, output_video: str, fps: int = 30):
    """
    Create video from sequence of images
    
    Args:
        images_dir: Directory containing images
        output_video: Path to output video file
        fps: Frames per second
    """
    images_path = Path(images_dir)
    image_files = sorted(list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')))
    
    if not image_files:
        print("No images found")
        return
    
    # Get dimensions from first image
    first_img = cv2.imread(str(image_files[0]))
    height, width = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        writer.write(img)
    
    writer.release()
    print(f"✓ Created video: {output_video} ({len(image_files)} frames @ {fps} FPS)")


def compare_counts(
    predicted_counts: dict, 
    ground_truth_counts: dict,
    save_report: str = None
):
    """
    Compare predicted counts with ground truth
    
    Args:
        predicted_counts: Dictionary of {image_name: predicted_count}
        ground_truth_counts: Dictionary of {image_name: true_count}
        save_report: Path to save comparison report (optional)
        
    Returns:
        Accuracy metrics
    """
    common_keys = set(predicted_counts.keys()) & set(ground_truth_counts.keys())
    
    if not common_keys:
        print("No common images found")
        return None
    
    errors = []
    absolute_errors = []
    
    for key in common_keys:
        pred = predicted_counts[key]
        true = ground_truth_counts[key]
        error = pred - true
        errors.append(error)
        absolute_errors.append(abs(error))
    
    metrics = {
        'total_images': len(common_keys),
        'mean_error': np.mean(errors),
        'mean_absolute_error': np.mean(absolute_errors),
        'max_error': max(absolute_errors),
        'accuracy': sum(1 for e in errors if e == 0) / len(errors) * 100,
        'within_1': sum(1 for e in absolute_errors if e <= 1) / len(errors) * 100,
        'within_2': sum(1 for e in absolute_errors if e <= 2) / len(errors) * 100,
    }
    
    print(f"\n{'='*60}")
    print(f"Counting Accuracy Metrics")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if 'accuracy' in key or 'within' in key:
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"{'='*60}\n")
    
    if save_report:
        with open(save_report, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Report saved to: {save_report}")
    
    return metrics


def draw_counting_zone_preview(
    video_path: str, 
    zone_type: str = 'line',
    save_path: str = None
):
    """
    Preview counting zone on first frame
    
    Args:
        video_path: Path to video file
        zone_type: 'line' or 'zone'
        save_path: Path to save preview image
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read video")
        return
    
    height, width = frame.shape[:2]
    
    # Draw zone
    if zone_type == 'line':
        y = int(height * 0.6)
        cv2.line(frame, (0, y), (width, y), (0, 0, 255), 3)
        cv2.putText(frame, 'Counting Line', (10, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        y_start = int(height * 0.6)
        cv2.rectangle(frame, (0, y_start), (width, height), (0, 0, 255), 3)
        cv2.putText(frame, 'Counting Zone', (10, y_start - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, frame)
        print(f"✓ Zone preview saved to: {save_path}")
    else:
        cv2.imshow('Counting Zone Preview', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """CLI for Dataset QA and Utilities"""
    import argparse
    parser = argparse.ArgumentParser(description='Industrial Dataset QA & Utilities')
    parser.add_argument('--action', type=str, required=True, 
                       choices=['qa', 'stats', 'viz', 'video', 'cleanup'],
                       help='Action to perform')
    parser.add_argument('--images', type=str, help='Path to images directory')
    parser.add_argument('--labels', type=str, help='Path to labels directory')
    parser.add_argument('--output', type=str, help='Output file/dir (e.g. video path or backup dir)')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples for visualization')
    
    args = parser.parse_args()
    
    if args.action == 'qa':
        if not args.images or not args.labels:
            print("Error: --images and --labels required for QA")
            return
        qa = DatasetQA(args.images, args.labels)
        qa.report_health()
        
    elif args.action == 'cleanup':
        if not args.images or not args.labels:
            print("Error: --images and --labels required for cleanup")
            return
        qa = DatasetQA(args.images, args.labels)
        qa.cleanup(backup_dir=args.output)
        
    elif args.action == 'stats':
        if not args.labels:
            print("Error: --labels required for stats")
            return
        calculate_dataset_statistics(args.labels)
        
    elif args.action == 'viz':
        if not args.images or not args.labels:
            print("Error: --images and --labels required for visualization")
            return
        visualize_dataset_sample(args.images, args.labels, num_samples=args.samples, save_path=args.output)
        
    elif args.action == 'video':
        if not args.images or not args.output:
            print("Error: --images and --output required for video creation")
            return
        create_video_from_images(args.images, args.output)


if __name__ == '__main__':
    main()
