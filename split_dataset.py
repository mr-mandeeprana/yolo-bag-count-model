"""
Split annotated dataset into train/val/test sets
Automatically organizes your annotated images and labels
"""

import os
import shutil
import random
from pathlib import Path

def split_dataset(
    images_dir="data/raw/extracted_frames",
    labels_dir="data/processed/labels/train",
    output_base="data/processed",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
):
    """
    Split annotated dataset into train/val/test sets
    
    Args:
        images_dir: Directory containing original images
        labels_dir: Directory containing YOLO format labels (.txt files)
        output_base: Base directory for organized dataset
        train_ratio: Fraction for training set (default: 0.8)
        val_ratio: Fraction for validation set (default: 0.1)
        test_ratio: Fraction for test set (default: 0.1)
    """
    
    # Convert to Path objects
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_base = Path(output_base)
    
    print("=" * 60)
    print("Dataset Splitting Tool")
    print("=" * 60)
    
    # Get all annotated images (those with corresponding .txt files)
    label_files = list(labels_dir.glob("*.txt"))
    
    if not label_files:
        print(f"\n❌ Error: No label files found in {labels_dir}")
        print("Please annotate your images first using LabelImg")
        return
    
    image_names = [f.stem for f in label_files]
    
    # Shuffle for random split
    random.seed(42)
    random.shuffle(image_names)
    
    # Calculate split indices
    total = len(image_names)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_names = image_names[:train_end]
    val_names = image_names[train_end:val_end]
    test_names = image_names[val_end:]
    
    print(f"\nTotal annotated images: {total}")
    print(f"Train: {len(train_names)} ({len(train_names)/total*100:.1f}%)")
    print(f"Val:   {len(val_names)} ({len(val_names)/total*100:.1f}%)")
    print(f"Test:  {len(test_names)} ({len(test_names)/total*100:.1f}%)")
    
    # Create output directories
    for split in ["train", "val", "test"]:
        (output_base / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_base / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    print("\n✓ Created output directories")
    
    # Copy files to respective splits
    print("\nCopying files...")
    
    for split_name, names in [("train", train_names), ("val", val_names), ("test", test_names)]:
        print(f"\n{split_name.upper()}:")
        
        for i, name in enumerate(names, 1):
            # Find image file (could be .jpg, .jpeg, .png)
            src_img = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                potential_img = images_dir / f"{name}{ext}"
                if potential_img.exists():
                    src_img = potential_img
                    break
            
            if not src_img:
                print(f"  ⚠ Warning: Image not found for {name}")
                continue
            
            # Copy image
            dst_img = output_base / "images" / split_name / src_img.name
            if src_img.resolve() != dst_img.resolve():
                shutil.copy(src_img, dst_img)
            
            # Copy label (skip if source and destination are the same)
            src_lbl = labels_dir / f"{name}.txt"
            dst_lbl = output_base / "labels" / split_name / f"{name}.txt"
            
            if src_lbl.resolve() != dst_lbl.resolve():
                shutil.copy(src_lbl, dst_lbl)
            
            if i % 50 == 0:
                print(f"  Copied {i}/{len(names)} files...")
        
        print(f"  ✓ Processed {len(names)} files")
    
    print("\n" + "=" * 60)
    print("✓ Dataset split complete!")
    print("=" * 60)
    print(f"\nDataset organized in: {output_base}")
    print("\nNext steps:")
    print("1. Verify data.yaml points to this directory")
    print("2. Run training: python src/train.py --data config/data.yaml")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split annotated dataset')
    parser.add_argument('--images', type=str, default='data/raw/extracted_frames',
                       help='Directory containing images')
    parser.add_argument('--labels', type=str, default='data/processed/labels/train',
                       help='Directory containing YOLO labels')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output base directory')
    parser.add_argument('--train', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    
    args = parser.parse_args()
    
    # Calculate test ratio
    test_ratio = 1.0 - args.train - args.val
    
    if test_ratio < 0:
        print("Error: Train + Val ratios exceed 1.0")
        exit(1)
    
    split_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        output_base=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=test_ratio
    )
