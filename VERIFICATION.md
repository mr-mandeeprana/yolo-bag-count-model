# Quick Verification Guide

Follow these steps to verify the system is working:

## Step 1: Check Python Installation

```powershell
python --version
```
**Expected**: Python 3.8 or higher ✅ (You have 3.12.6)

---

## Step 2: Create Virtual Environment

```powershell
cd "c:\Users\mrman\OneDrive\Desktop\Beumer Data\Yolo_bag_count_model"
python -m venv venv
```

---

## Step 3: Activate Virtual Environment

```powershell
.\venv\Scripts\activate
```

**Expected**: Your prompt should show `(venv)` at the beginning

---

## Step 4: Install Dependencies

```powershell
pip install --upgrade pip
pip install ultralytics opencv-python supervision numpy pyyaml
```

**This will take 2-5 minutes**

---

## Step 5: Quick Test - Import Check

```powershell
python -c "from ultralytics import YOLO; print('✓ Ultralytics installed')"
python -c "import cv2; print('✓ OpenCV installed')"
python -c "import supervision; print('✓ Supervision installed')"
```

**Expected**: All three should print ✓ messages

---

## Step 6: Test YOLO Model Download

```powershell
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('✓ YOLO model downloaded and loaded')"
```

**Expected**: Downloads YOLOv8n model (~6MB) and prints success message

---

## Step 7: Test Project Files

```powershell
python -c "import os; files = ['src/train.py', 'src/inference_video.py']; print('✓ All files exist' if all(os.path.exists(f) for f in files) else '✗ Missing files')"
```

**Expected**: ✓ All files exist

---

## Step 8: Test with Dummy Image (Full System Test)

```powershell
python -c "from ultralytics import YOLO; import numpy as np; model = YOLO('yolov8n.pt'); img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8); results = model(img, verbose=False); print(f'✓ Inference successful - detected {len(results[0].boxes)} objects')"
```

**Expected**: ✓ Inference successful - detected X objects

---

## Troubleshooting

### Issue: "pip is not recognized"
**Solution**: 
```powershell
python -m pip install --upgrade pip
```

### Issue: "No module named 'ultralytics'"
**Solution**: Make sure virtual environment is activated (you should see `(venv)` in prompt)
```powershell
.\venv\Scripts\activate
pip install ultralytics
```

### Issue: Import errors
**Solution**: Install all requirements
```powershell
pip install -r requirements.txt
```

---

## Quick Commands Reference

```powershell
# Always activate environment first
.\venv\Scripts\activate

# Test if packages are installed
python -c "import ultralytics, cv2, supervision; print('✓ All packages installed')"

# Download YOLO model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Test on any image you have
python src/inference_image.py --weights yolov8n.pt --source "path/to/image.jpg" --show
```

---

## Success Criteria

✅ Virtual environment activated  
✅ Packages installed (ultralytics, opencv, supervision)  
✅ YOLO model downloads successfully  
✅ Can run inference on dummy image  

**If all above pass → System is ready!**

---

## Next Steps After Verification

1. Collect Fillpac images/videos
2. Annotate with Roboflow
3. Train model: `python src/train.py`
4. Run counting: `python src/inference_video.py`
