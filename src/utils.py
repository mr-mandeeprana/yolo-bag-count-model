"""
Utility Functions for YOLO Bag Counting System
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import json


def visualize_dataset_sample(
    images_dir: str, 
    labels_dir: str, 
    num_samples: int = 5,
    save_path: str = None
):
    """
    Visualize annotated samples from dataset
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO labels
        num_samples: Number of samples to visualize
        save_path: Path to save visualization (optional)
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
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


def calculate_dataset_statistics(labels_dir: str):
    """
    Calculate statistics about the dataset
    
    Args:
        labels_dir: Directory containing YOLO labels
        
    Returns:
        Dictionary with statistics
    """
    labels_path = Path(labels_dir)
    label_files = list(labels_path.glob('*.txt'))
    
    total_images = len(label_files)
    total_bags = 0
    bags_per_image = []
    bbox_sizes = []
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        num_bags = len(lines)
        total_bags += num_bags
        bags_per_image.append(num_bags)
        
        for line in lines:
            parts = line.strip().split()
            width, height = float(parts[3]), float(parts[4])
            bbox_sizes.append((width, height))
    
    stats = {
        'total_images': total_images,
        'total_bags': total_bags,
        'avg_bags_per_image': np.mean(bags_per_image) if bags_per_image else 0,
        'max_bags_per_image': max(bags_per_image) if bags_per_image else 0,
        'min_bags_per_image': min(bags_per_image) if bags_per_image else 0,
        'avg_bbox_width': np.mean([w for w, h in bbox_sizes]) if bbox_sizes else 0,
        'avg_bbox_height': np.mean([h for w, h in bbox_sizes]) if bbox_sizes else 0,
    }
    
    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"{'='*60}\n")
    
    return stats


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


if __name__ == '__main__':
    # Example usage
    print("Utility functions for YOLO bag counting")
    print("Import these functions in your scripts")
