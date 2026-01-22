# Training Data Preparation - Next Steps

## ✅ Frames Extracted Successfully!

**Total frames extracted: 341**
- Video 1 (12.38.18): ~95 frames
- Video 2 (14.12.53): ~246 frames

All frames saved to: `data/raw/extracted_frames/`

---

## 📋 Next Steps: Annotate Your Data

### Step 1: Create Roboflow Account

1. Go to **https://roboflow.com**
2. Click "Sign Up" (it's FREE)
3. Create account with email

### Step 2: Create New Project

1. Click "Create New Project"
2. Project Name: **Fillpac Bag Detection**
3. Project Type: **Object Detection**
4. Annotation Group: **bag** (single class)
5. Click "Create Project"

### Step 3: Upload Images

1. Click "Upload" button
2. Navigate to: `c:\Users\mrman\OneDrive\Desktop\Beumer Data\Yolo_bag_count_model\data\raw\extracted_frames\`
3. Select ALL 341 images
4. Click "Upload"
5. Wait for upload to complete (2-5 minutes)

### Step 4: Annotate Bags

**This is the most important step!**

1. Click on first image
2. Click "Bounding Box" tool
3. Draw a box around EACH bag in the image:
   - Click and drag to create box
   - Make sure box tightly fits the bag
   - Label it as "bag"
4. Click "Save" and move to next image
5. Repeat for all 341 images

**Tips for annotation:**
- ✓ Include the entire bag in the box
- ✓ Annotate partially visible bags
- ✓ Be consistent with box sizes
- ✗ Don't include conveyor or background
- ✗ Don't skip any bags

**Time estimate:** 1-2 hours for 341 images (20-30 seconds per image)

### Step 5: Generate Dataset

1. After annotating all images, click "Generate"
2. Choose split:
   - Train: 80%
   - Valid: 10%
   - Test: 10%
3. Click "Generate"

### Step 6: Export for YOLOv8

1. Click "Export"
2. Format: **YOLOv8**
3. Click "Download ZIP"
4. Save to your computer

### Step 7: Extract and Organize

1. Extract the downloaded ZIP file
2. You'll see folders: `train/`, `valid/`, `test/`
3. Copy contents to your project:

```powershell
# Copy to your project
# Assuming downloaded to Downloads folder

# Copy train images
Copy-Item "Downloads\Fillpac-Bag-Detection-1\train\images\*" "data\processed\images\train\"
Copy-Item "Downloads\Fillpac-Bag-Detection-1\train\labels\*" "data\processed\labels\train\"

# Copy valid images
Copy-Item "Downloads\Fillpac-Bag-Detection-1\valid\images\*" "data\processed\images\val\"
Copy-Item "Downloads\Fillpac-Bag-Detection-1\valid\labels\*" "data\processed\labels\val\"

# Copy test images
Copy-Item "Downloads\Fillpac-Bag-Detection-1\test\images\*" "data\processed\images\test\"
Copy-Item "Downloads\Fillpac-Bag-Detection-1\test\labels\*" "data\processed\labels\test\"
```

### Step 8: Update Configuration

Edit `config/data.yaml`:

```yaml
path: c:/Users/mrman/OneDrive/Desktop/Beumer Data/Yolo_bag_count_model/data/processed
train: images/train
val: images/val
test: images/test

nc: 1
names: ['bag']
```

### Step 9: Train Your Model!

```powershell
# Activate environment
.\venv\Scripts\activate

# Train model (will take 2-4 hours on GPU)
python src/train.py --data config/data.yaml --validate

# Or quick test (5 epochs, ~10 minutes)
python src/train.py --data config/data.yaml --epochs 5
```

---

## 🎯 Alternative: Quick Start with Fewer Images

If you want to test the pipeline first:

1. Annotate only 50-100 images (instead of all 341)
2. Train a quick model (10 epochs)
3. Test if it works
4. Then annotate all images for production model

---

## 📊 Expected Results

With 341 annotated images:
- **Training time**: 2-4 hours (50 epochs)
- **Expected accuracy**: mAP@0.5 > 0.80
- **Counting accuracy**: >90%

---

## 🆘 Need Help?

**Annotation taking too long?**
- Annotate in batches (100 images per day)
- Use keyboard shortcuts in Roboflow (faster)
- Consider hiring annotation service (Roboflow offers this)

**Questions about annotation?**
- Watch Roboflow tutorial: https://www.youtube.com/watch?v=x0ThXHbtqCQ
- Check Roboflow docs: https://docs.roboflow.com

---

## Summary

**Current Status:**
✅ Videos processed  
✅ 341 frames extracted  
⏳ Ready for annotation  

**Next Action:**
👉 Go to https://roboflow.com and start annotating!

**After annotation:**
👉 Train model with `python src/train.py`
