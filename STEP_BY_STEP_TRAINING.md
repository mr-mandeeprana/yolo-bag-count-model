# Step-by-Step Training Guide

Complete walkthrough for training your cement bag detection model from scratch.

---

## 📋 Overview

**Total Time**: ~5-7 hours
- Annotation: 2-3 hours
- Training: 2-4 hours
- Testing: 30 minutes

**What You'll Get**: A trained model that can detect and count cement bags in videos

---

## Step 1: Launch LabelImg (5 minutes)

### Open the annotation tool:

```powershell
labelImg
```

**If it doesn't open**, try:
```powershell
python -m labelImg
```

You should see a window with buttons like "Open Dir", "Change Save Dir", etc.

---

## Step 2: Configure LabelImg (2 minutes)

### A. Set Image Directory
1. Click **"Open Dir"** button
2. Navigate to: `c:\Users\mrman\OneDrive\Desktop\Beumer Data\Yolo_bag_count_model\data\raw\extracted_frames`
3. Click **"Select Folder"**
4. You should see the first image appear

### B. Set Save Directory
1. Click **"Change Save Dir"** button
2. Navigate to: `c:\Users\mrman\OneDrive\Desktop\Beumer Data\Yolo_bag_count_model\data\processed\labels\train`
3. Click **"Select Folder"**

### C. Change Format to YOLO
1. Look for the format selector (shows "PascalVOC" by default)
2. Click on it to change to **"YOLO"**
3. Verify it now shows "YOLO"

**✓ Configuration Complete!** You're ready to annotate.

---

## Step 3: Annotate Images (2-3 hours)

### For Each Image (341 total):

#### 1. Draw Bounding Box
- Press **`W`** key (or click "Create RectBox")
- Click and drag to draw a box around **each bag** in the image
- Make the box tight around the bag (include the whole bag)

#### 2. Label the Box
- A dialog will appear asking for the class name
- Type: **`bag`** (lowercase)
- Press Enter
- **Note**: After the first time, "bag" will be auto-selected

#### 3. Save Annotation
- Press **`Ctrl+S`** (or click "Save")
- A `.txt` file is created with the same name as the image

#### 4. Move to Next Image
- Press **`D`** key (or click "Next Image")

### Quick Workflow:
```
W → Draw box → Enter → Ctrl+S → D
(Repeat for each bag in the image)
```

### Tips:
- ✅ Include partially visible bags
- ✅ Draw tight boxes (no extra space)
- ✅ Annotate ALL bags in each image
- ❌ Don't skip any bags
- ❌ Don't include conveyor or background

### Progress Tracking:
- LabelImg shows: "Image: X/341" at the bottom
- Take breaks every 50-100 images
- You can close and resume anytime

**✓ Annotation Complete!** You should have 341 `.txt` files in `data/processed/labels/train/`

---

## Step 4: Split Dataset (1 minute)

### Organize your annotated data:

```powershell
python split_dataset.py
```

**What this does:**
- Splits your 341 annotated images into:
  - **Train**: 273 images (80%)
  - **Val**: 34 images (10%)
  - **Test**: 34 images (10%)
- Copies images and labels to proper folders

**Expected Output:**
```
Total annotated images: 341
Train: 273 (80.0%)
Val:   34 (10.0%)
Test:  34 (10.0%)
✓ Dataset split complete!
```

**✓ Dataset Ready!** Your data is now organized for training.

---

## Step 5: Verify Configuration (1 minute)

### Check that `config/data.yaml` is correct:

```powershell
cat config/data.yaml
```

**Should show:**
```yaml
path: ../data/processed
train: images/train
val: images/val
test: images/test

nc: 1
names: ['bag']
```

If it looks different, the paths might need adjustment.

**✓ Configuration Verified!**

---

## Step 6: Train the Model (2-4 hours)

### Start Training:

**Option A: Full Training (Recommended)**
```powershell
python src/train.py --data config/data.yaml --epochs 50
```
- Time: 2-4 hours (depends on your GPU/CPU)
- Best quality results

**Option B: Quick Test (For Testing)**
```powershell
python src/train.py --data config/data.yaml --epochs 5
```
- Time: 15-20 minutes
- Lower quality, just for testing the pipeline

### What You'll See:

```
Loading model: yolov8n.pt
✓ Using GPU: NVIDIA GeForce RTX 3060
Starting Training for Fillpac Bag Detection
Epoch 1/50: 100%|████████| 17/17 [00:45<00:00]
  train/loss: 0.8234
  val/mAP50: 0.6543
Epoch 2/50: 100%|████████| 17/17 [00:42<00:00]
  ...
```

### Training Progress:
- Each epoch processes all training images
- Validation runs after each epoch
- Best model is saved automatically
- You can monitor the loss decreasing

### When Training Completes:

```
Training Complete!
✓ Best model saved to: models/weights/best.pt
```

**✓ Model Trained!** Your custom model is ready.

---

## Step 7: Test on Image (2 minutes)

### Test on a single frame:

```powershell
python src/inference_image.py --weights models/weights/best.pt --source "data/raw/extracted_frames/WhatsApp Video 2026-01-20 at 12.38.18_frame_0000.jpg" --show
```

**What happens:**
- Model detects bags in the image
- Shows detection results
- Displays image with bounding boxes

**Expected Output:**
```
Image: WhatsApp Video 2026-01-20 at 12.38.18_frame_0000.jpg
Bags detected: 3
  Bag 1: Confidence=0.892
  Bag 2: Confidence=0.876
  Bag 3: Confidence=0.845
```

**✓ Image Test Passed!**

---

## Step 8: Test on Video (5 minutes)

### Test on full video with counting:

```powershell
python src/inference_video.py --weights models/weights/best.pt --source "WhatsApp Video 2026-01-20 at 12.38.18.mp4" --output outputs/trained_result.mp4 --log outputs/trained_log.txt
```

**What happens:**
- Processes entire video
- Detects and tracks bags
- Counts bags crossing the line
- Saves annotated video
- Creates count log

**Expected Output:**
```
✓ Opened video: WhatsApp Video 2026-01-20 at 12.38.18.mp4
Video properties: 478x850 @ 29 FPS, 2748 frames
Processing video... Press 'q' to quit

Processing Complete
Total frames processed: 2748
Total bags counted: 127
Processing time: 248.50 seconds
Average FPS: 11.05
```

**✓ Video Test Complete!**

---

## Step 9: Review Results (10 minutes)

### Check Training Metrics:

```powershell
cd runs/detect/bag_counter
ls
```

**You'll find:**
- `weights/best.pt` - Best model
- `results.png` - Training graphs
- `confusion_matrix.png` - Performance matrix
- `val_batch0_pred.jpg` - Validation predictions

### Check Video Output:

```powershell
# Open the annotated video
outputs/trained_result.mp4

# Check the count log
cat outputs/trained_log.txt
```

### Evaluate Performance:

**Good Results:**
- mAP@0.5 > 0.80
- Bags are detected consistently
- Count matches manual count (±5%)

**If Results Are Poor:**
- Train for more epochs (100+)
- Add more annotated images
- Adjust confidence threshold

**✓ Training Complete!**

---

## 🎯 Quick Reference

### Commands Summary:

```powershell
# 1. Annotate
labelImg

# 2. Split dataset
python split_dataset.py

# 3. Train model
python src/train.py --data config/data.yaml --epochs 50

# 4. Test on image
python src/inference_image.py --weights models/weights/best.pt --source image.jpg

# 5. Test on video
python src/inference_video.py --weights models/weights/best.pt --source video.mp4 --output result.mp4
```

---

## 🆘 Troubleshooting

### LabelImg won't open
```powershell
python -m labelImg
```

### Training fails with "No images found"
- Check that `split_dataset.py` ran successfully
- Verify images are in `data/processed/images/train/`

### Training is very slow
- Using CPU instead of GPU (expected on CPU)
- Reduce batch size in `config/model_config.yaml`

### Low detection accuracy
- Annotate more images (aim for 500+)
- Train for more epochs (100+)
- Check annotation quality

### Model detects too many false positives
- Increase confidence threshold: `--conf 0.7`

---

## ✅ Success Checklist

- [ ] LabelImg configured and working
- [ ] All 341 images annotated
- [ ] Dataset split successfully
- [ ] Model trained (50 epochs)
- [ ] Test on image shows detections
- [ ] Test on video counts bags
- [ ] Results saved to outputs/

**Congratulations!** You now have a working bag detection model! 🎉

---

## 📊 Expected Timeline

| Step | Time | Cumulative |
|------|------|------------|
| 1-2. Setup LabelImg | 5 min | 5 min |
| 3. Annotate 341 images | 2-3 hours | ~3 hours |
| 4-5. Split & verify | 2 min | ~3 hours |
| 6. Train model | 2-4 hours | ~6 hours |
| 7-9. Test & review | 15 min | ~6.5 hours |

**Total**: ~6-7 hours (can be done over multiple days)
