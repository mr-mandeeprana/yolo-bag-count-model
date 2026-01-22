# YOLO Bag Detection - Quick Start Guide

## 🎯 Local Training Workflow (No Cloud Required)

Train your cement bag detection model completely offline using free tools.

---

## Step 1: Annotate Your Images

### Install LabelImg ✅
```powershell
# Already installed!
labelImg
```

### Annotate 341 Frames
1. **Open Dir**: Select `data/raw/extracted_frames/`
2. **Change Save Dir**: Select `data/processed/labels/train/`
3. **Set Format**: Change to **YOLO** format
4. **Annotate**: Press `W` to draw box, `D` for next image
5. **Label**: Type `bag` as class name

**Time**: 2-3 hours for 341 images

---

## Step 2: Split Dataset

```powershell
python split_dataset.py
```

This organizes your annotated data into train/val/test sets (80/10/10).

---

## Step 3: Train Model

```powershell
# Full training (50 epochs, 2-4 hours)
python src/train.py --data config/data.yaml --epochs 50

# Quick test (5 epochs, ~15 minutes)
python src/train.py --data config/data.yaml --epochs 5
```

Model will be saved to: `models/weights/best.pt`

---

## Step 4: Test Your Model

### On Image:
```powershell
python src/inference_image.py --weights models/weights/best.pt --source "data/raw/extracted_frames/WhatsApp Video 2026-01-20 at 12.38.18_frame_0000.jpg"
```

### On Video:
```powershell
python src/inference_video.py --weights models/weights/best.pt --source "WhatsApp Video 2026-01-20 at 12.38.18.mp4" --output outputs/result.mp4 --log outputs/count_log.txt
```

---

## 📊 Expected Results

- **Training time**: 2-4 hours (50 epochs on GPU)
- **Expected mAP@0.5**: >0.80
- **Counting accuracy**: >90%
- **Model size**: ~6-12 MB

---

## 🔧 Troubleshooting

**LabelImg not opening?**
```powershell
python -m labelImg
```

**Training fails?**
- Check that `data/processed/` has images and labels
- Verify `config/data.yaml` paths are correct

**Need help?**
- See [`LOCAL_TRAINING_GUIDE.md`](LOCAL_TRAINING_GUIDE.md) for detailed instructions
- Check [`ANNOTATION_GUIDE.md`](ANNOTATION_GUIDE.md) for annotation tips

---

## 📁 Project Structure

```
Yolo_bag_count_model/
├── data/
│   ├── raw/extracted_frames/     # 341 frames to annotate
│   └── processed/                # Organized dataset (after split)
├── src/
│   ├── train.py                  # Training script
│   ├── inference_video.py        # Video inference
│   └── inference_image.py        # Image inference
├── models/weights/               # Trained model saved here
├── split_dataset.py              # Dataset organization tool
└── LOCAL_TRAINING_GUIDE.md       # Detailed guide
```

---

**Ready?** Start by running `labelImg` and annotating your images!
