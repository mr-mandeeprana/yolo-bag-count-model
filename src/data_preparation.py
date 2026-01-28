"""
Data Preparation Utilities for YOLO Bag Counting
Handles dataset collection, annotation conversion, and augmentation
"""

import os
import shutil
import yaml
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# os:file management, shutil:file operations, yaml:yaml file operations(reading & writing), 
# pathlib:path operations, typing:type hints, cv2:image operations, 
# numpy:numerical operations, tqdm:progress bar, albumentations:image augmentation
class DatasetPreparer:
    """Prepare and organize dataset for YOLO training"""
    
    def __init__(self, raw_dir: str, output_dir: str):
        """
        Initialize dataset preparer
        
        Args:
            raw_dir: Directory containing raw images/videos
            output_dir: Directory for processed dataset
        """
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        
        # Create directory structure (yolo format)
        self.setup_directories()
        
    def setup_directories(self):
        """Create YOLO dataset directory structure"""
        dirs = [
            'images/train', 'images/val', 'images/test',
            'labels/train', 'labels/val', 'labels/test'
        ]
        for dir_path in dirs:
            (self.output_dir / dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory structure in {self.output_dir}")
    #Create directories for images and labels and output directory (Lines 33-43) 
    
    def extract_frames_from_video(
        self, 
        video_path: str, 
        output_dir: str, 
        frame_interval: int = 30
    ) -> List[str]:

      #Extract frames from video  (Lines 44-86)
        """
        Extract frames from video for annotation
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            frame_interval: Extract every Nth frame (default: 30 = 1 per second at 30fps)
            
        Returns:
            List of saved frame paths
        """
        cap = cv2.VideoCapture(video_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        saved_frames = []
        
        video_name = Path(video_path).stem
        
        with tqdm(desc=f"Extracting frames from {video_name}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_path = output_path / f"{video_name}_frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    saved_frames.append(str(frame_path))
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        print(f"✓ Extracted {len(saved_frames)} frames from {video_name}")
        return saved_frames
    
    #Convert labelimg to yolo (Lines 88-130) (data->Image->train/val/test)
    #frames saved from images to labels(data->Label->train/val/test)
    def convert_labelimg_to_yolo(self, xml_dir: str, output_label_dir: str):
        """
        Convert LabelImg XML annotations to YOLO format
        
        Args:
            xml_dir: Directory containing XML files
            output_label_dir: Directory to save YOLO format labels
        """
        # Note: Requires xml.etree.ElementTree
        import xml.etree.ElementTree as ET
        
        xml_path = Path(xml_dir)
        output_path = Path(output_label_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for xml_file in xml_path.glob('*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image dimensions
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            # Convert each object
            yolo_annotations = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name != 'bag':
                    continue  # Skip non-bag objects
                
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # Convert to YOLO format (normalized center x, y, width, height)
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                # Class ID 0 for 'bag'
                yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save YOLO format label
            if yolo_annotations:
                label_file = output_path / f"{xml_file.stem}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
        
        print(f"✓ Converted {len(list(xml_path.glob('*.xml')))} XML files to YOLO format")
    #split dataset into train/val/test (Lines 149-193)
    def split_dataset(
        self, 
        images_dir: str, 
        labels_dir: str, 
        train_ratio: float = 0.8, 
        val_ratio: float = 0.1
    ):
        """
        Split dataset into train/val/test sets
        
        Args:
            images_dir: Directory containing all images
            labels_dir: Directory containing all labels
            train_ratio: Proportion for training (default: 0.8)
            val_ratio: Proportion for validation (default: 0.1)
        """
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        
        # Get all image files
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        np.random.shuffle(image_files)
        
        total = len(image_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:]
        }
        

        for split_name, files in splits.items():
            for img_file in tqdm(files, desc=f"Copying {split_name} set"):
                # Copy image
                dst_img = self.output_dir / 'images' / split_name / img_file.name
                shutil.copy(img_file, dst_img)
                
                # Copy corresponding label
                label_file = labels_path / f"{img_file.stem}.txt"
                if label_file.exists():
                    dst_label = self.output_dir / 'labels' / split_name / label_file.name
                    shutil.copy(label_file, dst_label)
        
        print(f"✓ Dataset split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    #create augmentation pipeline (Lines 196-238)

    def create_augmentation_pipeline(self) -> A.Compose:
     # Augmentation pipeline using Albumentations to enhance dataset diversity.
     # Includes horizontal flipping, brightness/contrast adjustment, Gaussian noise,
     # blur, CLAHE for dust and lighting variations, and gamma correction.
     # BboxParams ensure YOLO-format bounding boxes remain accurate after transforms.
     # Gaussian noise improves robustness by altering pixel values, while BboxParams ensures bounding boxes remain accurate during all geometric image transformations.
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.3),  # For dust/lighting variations
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        return transform
     
      #line(215-278)
      # This function increases the dataset size by creating augmented copies of images.
      # For each image, it reads the YOLO label file, applies image augmentations,
      # and updates bounding boxes correctly.
      # Augmented images and their corresponding labels are saved with new filenames.

    def augment_dataset(self, split: str = 'train', augment_factor: int = 2):
        """
        Apply augmentation to increase dataset size
        
        Args:
            split: Dataset split to augment ('train', 'val', or 'test')
            augment_factor: Number of augmented versions per image
        """
        transform = self.create_augmentation_pipeline()
        
        images_dir = self.output_dir / 'images' / split
        labels_dir = self.output_dir / 'labels' / split
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        for img_file in tqdm(image_files, desc=f"Augmenting {split} set"):
            image = cv2.imread(str(img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Read YOLO labels
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            bboxes = []
            class_labels = []
            for line in lines:
                parts = line.strip().split()
                class_labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])
            
            # Generate augmented versions
            for i in range(augment_factor):
                try:
                    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_labels = transformed['class_labels']
                    
                    # Save augmented image
                    aug_img_name = f"{img_file.stem}_aug{i}{img_file.suffix}"
                    aug_img_path = images_dir / aug_img_name
                    cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                    
                    # Save augmented labels
                    aug_label_path = labels_dir / f"{img_file.stem}_aug{i}.txt"
                    with open(aug_label_path, 'w') as f:
                        for cls, bbox in zip(aug_labels, aug_bboxes):
                            f.write(f"{cls} {' '.join([f'{x:.6f}' for x in bbox])}\n")
                
                except Exception as e:
                    print(f"Warning: Failed to augment {img_file.name}: {e}")
                    continue
        
        print(f"✓ Augmented {split} set with factor {augment_factor}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLO bag counting')
    parser.add_argument('--raw', type=str, default='data/raw', help='Raw data directory')
    parser.add_argument('--output', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--extract-video', type=str, help='Extract frames from video')
    parser.add_argument('--frame-interval', type=int, default=30, help='Frame extraction interval')
    parser.add_argument('--augment', action='store_true', help='Apply augmentation')
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(args.raw, args.output)
    
    if args.extract_video:
        preparer.extract_frames_from_video(
            args.extract_video, 
            os.path.join(args.raw, 'extracted_frames'),
            args.frame_interval
        )
    
    if args.augment:
        preparer.augment_dataset('train', augment_factor=2)
    
    print("\n✓ Data preparation complete!")
    print(f"Next steps:")
    print(f"1. Annotate images in {args.output} using Roboflow or LabelImg")
    print(f"2. Run split_dataset() to organize train/val/test sets")
    print(f"3. Update config/data.yaml with correct paths")


if __name__ == '__main__':
    main()
