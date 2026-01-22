# Local Model Training Without Roboflow - Complete Guide

## Overview

Train your YOLO bag detection model **completely offline** using free, open-source tools. No cloud services required!

## 🎯 What You'll Need

1. **Annotation Tool**: LabelImg (free, offline)
2. **Training Data**: Your 341 extracted frames
3. **Training Script**: Already in your project (`src/train.py`)
4. **Time**: 2-3 hours for annotation + 2-4 hours for training

---

## Step 1: Install LabelImg (Annotation Tool)

LabelImg is a free, offline tool for drawing bounding boxes on images.

```powershell
# Install LabelImg
pip install labelImg

# Launch it
labelImg
```

**Alternative**: Download pre-built executable from https://github.com/HumanSignal/labelImg/releases

---

## Step 2: Annotate Your Images

### A. Setup LabelImg

1. **Launch LabelImg**: Run `labelImg` command
2. **Open Dir**: Click "Open Dir" → Select `data/raw/extracted_frames/`
3. **Change Save Dir**: Click "Change Save Dir" → Select `data/processed/labels/train/`
4. **Set Format**: Click "PascalVOC" and change to **"YOLO"** format

### B. Annotate Each Image

For each of the 341 images:

1. **Draw box**: Press `W` or click "Create RectBox"
2. **Draw around bag**: Click and drag to create bounding box around each bag
3. **Label it**: Type `bag` as the class name (first time only, then it auto-selects)
4. **Save**: Press `Ctrl+S` or click "Save"
5. **Next image**: Press `D` or click "Next Image"

**Tips**:
- Press `W` to draw, `D` for next image (faster workflow)
- Include partially visible bags
- Make boxes tight around bags
- Be consistent

**Time**: ~20-30 seconds per image = 2-3 hours total

---

## Step 3: Organize Your Dataset

After annotation, organize the data:

```powershell
# Create directory structure
New-Item -ItemType Directory -Force -Path "data/processed/images/train"
New-Item -ItemType Directory -Force -Path "data/processed/images/val"
New-Item -ItemType Directory -Force -Path "data/processed/images/test"
New-Item -ItemType Directory -Force -Path "data/processed/labels/train"
New-Item -ItemType Directory -Force -Path "data/processed/labels/val"
New-Item -ItemType Directory -Force -Path "data/processed/labels/test"
```

### Split Your Data (80/10/10)

Run this Python script to split your annotated data:

```python
# save as split_dataset.py
import os
import shutil
import random
from pathlib import Path

# Paths
images_dir = Path("data/raw/extracted_frames")
labels_dir = Path("data/processed/labels/train")  # Where LabelImg saved labels
output_base = Path("data/processed")

# Get all annotated images (those with corresponding .txt files)
label_files = list(labels_dir.glob("*.txt"))
image_names = [f.stem for f in label_files]

# Shuffle
random.seed(42)
random.shuffle(image_names)

# Split: 80% train, 10% val, 10% test
total = len(image_names)
train_split = int(total * 0.8)
val_split = int(total * 0.9)

train_names = image_names[:train_split]
val_names = image_names[train_split:val_split]
test_names = image_names[val_split:]

print(f"Total annotated: {total}")
print(f"Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")

# Copy files
for split_name, names in [("train", train_names), ("val", val_names), ("test", test_names)]:
    for name in names:
        # Copy image
        src_img = images_dir / f"{name}.jpg"
        dst_img = output_base / "images" / split_name / f"{name}.jpg"
        shutil.copy(src_img, dst_img)
        
        # Copy label
        src_lbl = labels_dir / f"{name}.txt"
        dst_lbl = output_base / "labels" / split_name / f"{name}.txt"
        shutil.copy(src_lbl, dst_lbl)

print("✓ Dataset split complete!")
```

Run it:
```powershell
python split_dataset.py
```

---

## Step 4: Verify Data Configuration

Check that `config/data.yaml` is correct:

```yaml
# config/data.yaml
path: ../data/processed
train: images/train
val: images/val
test: images/test

nc: 1
names: ['bag']
```

---

## Step 5: Train Your Model

Now train the model locally:

```powershell
# Full training (50 epochs, 2-4 hours on GPU)
python src/train.py --data config/data.yaml --epochs 50

# Quick test (5 epochs, ~15 minutes)
python src/train.py --data config/data.yaml --epochs 5
```

**What happens**:
- Model trains on your annotated data
- Progress shown in terminal
- Results saved to `runs/detect/bag_counter/`
- Best model saved to `models/weights/best.pt`

---

## Step 6: Test Your Trained Model

```powershell
# Test on image
python src/inference_image.py --weights models/weights/best.pt --source "data/raw/extracted_frames/WhatsApp Video 2026-01-20 at 12.38.18_frame_0000.jpg"

# Test on video
python src/inference_video.py --weights models/weights/best.pt --source "WhatsApp Video 2026-01-20 at 12.38.18.mp4" --output outputs/final_result.mp4
```

---

## 📊 Expected Results

With 341 annotated images:
- **Training time**: 2-4 hours (50 epochs on GPU)
- **Expected mAP@0.5**: >0.80
- **Counting accuracy**: >90%

---

## 🎨 Alternative Annotation Tools

If you don't like LabelImg, try:

| Tool | Pros | Cons |
|------|------|------|
| **LabelImg** | Simple, offline, YOLO format | Basic UI |
| **CVAT** | Advanced, tracking support | Requires setup |
| **Label Studio** | Modern UI, ML-assisted | More complex |
| **Makesense.ai** | Web-based, no install | Requires internet |

---

## ⚡ Quick Start Summary

```powershell
# 1. Install annotation tool
pip install labelImg

# 2. Annotate images
labelImg
# (Open data/raw/extracted_frames/, set YOLO format, annotate all)

# 3. Split dataset
python split_dataset.py

# 4. Train model
python src/train.py --data config/data.yaml --epochs 50

# 5. Test model
python src/inference_video.py --weights models/weights/best.pt --source "video.mp4"
```

---

## ✅ Advantages of Local Training

- ✅ **Fully offline** - No internet required
- ✅ **Free** - No cloud costs
- ✅ **Private** - Your data stays on your machine
- ✅ **Full control** - Customize everything
- ✅ **No dependencies** - No Roboflow account needed

---

**Ready to start?** Install LabelImg and begin annotating your 341 frames!
