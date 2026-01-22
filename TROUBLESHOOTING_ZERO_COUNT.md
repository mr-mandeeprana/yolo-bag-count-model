# Troubleshooting: 0 Bags Counted

## 🔍 Problem

Your trained model processed the video successfully but counted **0 bags**, even though:
- ✅ Model trained successfully (77.7% mAP)
- ✅ Video processed (2,748 frames)
- ✅ Output video created (60 MB)

## 🎯 Root Causes & Solutions

### Cause 1: Model Not Detecting Bags (Most Likely)

**Why**: The model was trained on 87 images, which might not be enough variety to recognize bags in all scenarios.

**Solution**: Test if model is detecting anything

```powershell
# Test on a single frame with visualization
python src/inference_image.py --weights models/weights/best.pt --source "data/raw/extracted_frames/WhatsApp Video 2026-01-20 at 12.38.18_frame_0100.jpg" --conf 0.25 --save
```

**Check**: Look at the output image in `runs/detect/predict/` - are there any bounding boxes?

---

### Cause 2: Confidence Threshold Too High

**Why**: Default confidence is 0.5 (50%). Model might be detecting bags but with lower confidence.

**Solution**: Lower the confidence threshold

```powershell
# Try with 25% confidence
python src/inference_video.py --weights models/weights/best.pt --source "WhatsApp Video 2026-01-20 at 12.38.18.mp4" --output outputs/low_conf_result.mp4 --conf 0.25

# Try with 10% confidence (very permissive)
python src/inference_video.py --weights models/weights/best.pt --source "WhatsApp Video 2026-01-20 at 12.38.18.mp4" --output outputs/very_low_conf.mp4 --conf 0.1
```

---

### Cause 3: Counting Line Position Wrong

**Why**: The virtual counting line might be positioned where bags don't cross.

**Solution**: Adjust counting line position

The counting line is configured in the script. You need to check where it's positioned relative to the conveyor belt.

**Check the output video**: Open `outputs/trained_result.mp4` and look for the counting line. Is it where bags actually cross?

---

### Cause 4: Training Data Mismatch

**Why**: The 87 images you annotated might be from different parts of the video or different lighting/angles than the test video.

**Solution**: Verify training data

1. Check which frames you annotated:
   ```powershell
   Get-ChildItem "data/processed/labels/train" | Select-Object -First 10 Name
   ```

2. Compare with test video frames - are they similar?

---

## 🔧 Quick Diagnostic Steps

### Step 1: Test Single Image Detection

```powershell
python src/inference_image.py --weights models/weights/best.pt --source "data/raw/extracted_frames/WhatsApp Video 2026-01-20 at 12.38.18_frame_0100.jpg" --conf 0.1 --save
```

**Expected**: Should see bounding boxes around bags in `runs/detect/predict/`

**If NO boxes**: Model isn't detecting bags → Need more/better training data

**If YES boxes**: Model works → Problem is with counting line or confidence threshold

---

### Step 2: Check Output Video

Open `outputs/trained_result.mp4` and look for:
- ❓ Are there any green bounding boxes around bags?
- ❓ Is there a counting line visible?
- ❓ Where is the line positioned?

---

### Step 3: Test with Lower Confidence

```powershell
python src/inference_video.py --weights models/weights/best.pt --source "WhatsApp Video 2026-01-20 at 12.38.18.mp4" --output outputs/test_conf_0.1.mp4 --conf 0.1 --log outputs/test_log.txt
```

Check the log file:
```powershell
cat outputs/test_log.txt
```

---

## 💡 Most Likely Solution

Based on 0 detections, the model probably isn't recognizing bags. Here's what to do:

### Option 1: Annotate More Images (Recommended)

You trained on only 87 images. For better results:

1. **Annotate more frames** - Aim for 200-300 images
2. **Include variety** - Different angles, lighting, bag positions
3. **Retrain** - `python src/train.py --data config/data.yaml`

### Option 2: Check if Annotations Were Correct

1. Look at a training image:
   ```powershell
   explorer "runs/detect/bag_counter3/train_batch0.jpg"
   ```

2. Verify bounding boxes are around bags

3. If boxes look wrong, re-annotate and retrain

### Option 3: Use Base Model for Comparison

Test with the base YOLOv8 model to see if it detects anything:

```powershell
python src/inference_video.py --weights yolov8n.pt --source "WhatsApp Video 2026-01-20 at 12.38.18.mp4" --output outputs/base_model_result.mp4 --conf 0.25
```

---

## 📊 Next Steps

1. **Run diagnostic test**:
   ```powershell
   python src/inference_image.py --weights models/weights/best.pt --source "data/raw/extracted_frames/WhatsApp Video 2026-01-20 at 12.38.18_frame_0100.jpg" --conf 0.1 --save
   ```

2. **Check the output** in `runs/detect/predict/`

3. **Based on results**:
   - **If detections visible** → Adjust confidence or counting line
   - **If no detections** → Need more training data

---

## 🎯 Expected Behavior

When working correctly, you should see:
- Green boxes around bags in the video
- Confidence scores (e.g., "bag 0.85")
- Count incrementing as bags cross the line
- Final count > 0

**Your current result (0 bags) means the model isn't detecting OR counting isn't working.**

Run the diagnostic test above to determine which!
