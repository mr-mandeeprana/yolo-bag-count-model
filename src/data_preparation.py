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
    def convert_labelimg_to_yolo(self, xml_dir: str, image_dir: str, output_label_dir: str):
        """
        Convert LabelImg XML annotations to YOLO format.
        
        Note: This method now validates dimensions using the actual image file
        to prevent coordinate skew if the XML 'size' tag is incorrect.
        """
        import xml.etree.ElementTree as ET
        
        xml_path = Path(xml_dir)
        img_path = Path(image_dir)
        output_path = Path(output_label_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        converted_count = 0
        for xml_file in xml_path.glob('*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Try to get image dimensions from file for better correctness
            img_file = img_path / f"{xml_file.stem}.jpg"
            if not img_file.exists():
                img_file = img_path / f"{xml_file.stem}.png"
            
            if img_file.exists():
                img = cv2.imread(str(img_file))
                if img is not None:
                    img_height, img_width = img.shape[:2]
                else:
                    size = root.find('size')
                    img_width = int(size.find('width').text)
                    img_height = int(size.find('height').text)
            else:
                size = root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)
            
            if img_width == 0 or img_height == 0:
                print(f"Warning: Skipping {xml_file.name} due to zero dimensions")
                continue
            
            yolo_annotations = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text.strip().lower()
                if class_name != 'bag': continue
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Correctness fix: Boundary clipping
                xmin = max(0, min(xmin, img_width))
                xmax = max(0, min(xmax, img_width))
                ymin = max(0, min(ymin, img_height))
                ymax = max(0, min(ymax, img_height))
                
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            if yolo_annotations:
                label_file = output_path / f"{xml_file.stem}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                converted_count += 1
        
        print(f"✓ Converted {converted_count} valid XML files to YOLO format")
    #split dataset into train/val/test (Lines 149-193)
    def split_dataset(
        self, 
        images_dir: str, 
        labels_dir: str, 
        train_ratio: float = 0.8, 
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Split dataset into train/val/test sets.
        Correctness fix: Fixed seed for reproducibility and filters for images that have labels.
        """
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        
        # Get all image files that have corresponding labels
        all_image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        image_files = [f for f in all_image_files if (labels_path / f"{f.stem}.txt").exists()]
        
        if len(image_files) < len(all_image_files):
            print(f"ℹ Found {len(all_image_files)-len(image_files)} images without labels. These will be skipped.")
            
        # Correctness fix: Use fixed random seed for reproducible splits
        np.random.seed(seed)
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
                shutil.copy(str(img_file), str(dst_img))
                
                # Copy corresponding label
                label_file = labels_path / f"{img_file.stem}.txt"
                dst_label = self.output_dir / 'labels' / split_name / label_file.name
                shutil.copy(str(label_file), str(dst_label))
        
        print(f"✓ Reproducible dataset split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

     


    def generate_data_yaml(self, nc: int = 1, names: list = ['bag']):
        """Generate a basic data.yaml file for YOLO training"""
        data = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': nc,
            'names': names
        }
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"✓ Generated data.yaml in {self.output_dir}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLO bag counting')
    parser.add_argument('--raw', type=str, default='data/raw', help='Raw data directory')
    parser.add_argument('--output', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--extract-video', type=str, help='Extract frames from video')
    parser.add_argument('--frame-interval', type=int, default=30, help='Frame extraction interval')
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(args.raw, args.output)
    
    if args.extract_video:
        frames_dir = os.path.join(args.raw, 'extracted_frames')
        preparer.extract_frames_from_video(
            args.extract_video, 
            frames_dir,
            args.frame_interval
        )
        print(f"Next: Annotate frames in {frames_dir} using LabelImg")
    
    # Example full workflow if data exists
    raw_images = preparer.raw_dir / 'images'
    raw_xmls = preparer.raw_dir / 'annotations'
    temp_labels = preparer.raw_dir / 'temp_yolo_labels'
    
    if raw_images.exists() and raw_xmls.exists():
        print("\nFound raw images and XMLs. Processing...")
        preparer.convert_labelimg_to_yolo(str(raw_xmls), str(raw_images), str(temp_labels))
        preparer.split_dataset(str(raw_images), str(temp_labels))
        preparer.generate_data_yaml()
        
    
    print("\n✓ Data preparation complete!")


if __name__ == '__main__':
    main()
