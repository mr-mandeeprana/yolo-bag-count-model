# Quick Start Guide: YOLO Bag Counting for Fillpac

This guide will help you get started with the YOLO bag counting system in under 30 minutes.

## Prerequisites

- Windows PC with Python 3.8+ installed
- NVIDIA GPU (recommended, but CPU works for testing)
- Sample images or videos from your Fillpac machine

## Step 1: Environment Setup (5 minutes)

Open PowerShell and navigate to the project:

```powershell
cd "c:\Users\mrman\OneDrive\Desktop\Beumer Data\Yolo_bag_count_model"
```

Create and activate virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Verify installation:

```powershell
python -c "from ultralytics import YOLO; print('✓ Installation successful')"
```

## Step 2: Prepare Sample Dataset (10 minutes)

### Option A: Use Existing Images

If you have images from Fillpac:

1. Create dataset structure:
```powershell
mkdir -p data/raw/images
```

2. Copy your images to `data/raw/images/`

### Option B: Extract from Video

If you have a Fillpac video:

```powershell
python src/data_preparation.py --extract-video "path/to/fillpac_video.mp4" --frame-interval 30
```

This extracts 1 frame per second.

### Annotate Images

**Recommended: Use Roboflow**

1. Go to [Roboflow](https://roboflow.com)
2. Create free account
3. Upload images
4. Draw bounding boxes around bags (label as "bag")
5. Export in "YOLOv8" format
6. Download and extract to `data/processed/`

**Alternative: Use LabelImg**

```powershell
pip install labelImg
labelImg data/raw/extracted_frames
```

- Draw boxes around bags
- Save as YOLO format
- Move to `data/processed/`

### Update Configuration

Edit `config/data.yaml` to point to your dataset:

```yaml
path: c:/Users/mrman/OneDrive/Desktop/Beumer Data/Yolo_bag_count_model/data/processed
train: images/train
val: images/val
```

## Step 3: Train Model (10 minutes for quick test)

Quick training test (2 epochs):

```powershell
python src/train.py --data config/data.yaml
```

For production (50 epochs, ~1-2 hours on GPU):

```powershell
# Edit config/model_config.yaml and set epochs: 50
python src/train.py --data config/data.yaml --validate
```

Training will save weights to `runs/detect/bag_counter/weights/best.pt`

## Step 4: Test Inference (5 minutes)

### Test on Image

```powershell
python src/inference_image.py --weights models/weights/best.pt --source "data/processed/images/test/sample.jpg" --show
```

### Test on Video

```powershell
python src/inference_video.py --weights models/weights/best.pt --source "path/to/test_video.mp4" --output "outputs/result.mp4"
```

### Test on Live Camera

```powershell
python src/inference_video.py --weights models/weights/best.pt --source 0
```

(Use `0` for default webcam, or RTSP URL for IP camera)

## Step 5: Production Deployment

### For Edge Device (Jetson)

Export to TensorRT:

```powershell
python -c "from ultralytics import YOLO; YOLO('models/weights/best.pt').export(format='engine')"
```

Copy `.engine` file to Jetson and run:

```bash
python src/inference_video.py --weights best.engine --source rtsp://camera-ip
```

### For Cloud/Server

Run directly with PyTorch:

```powershell
python src/inference_video.py --weights models/weights/best.pt --source rtsp://fillpac-camera --log outputs/counts/production.txt
```

## Common Issues

**Issue**: "CUDA out of memory"
**Solution**: Reduce batch size in `config/model_config.yaml` (try 8 or 4)

**Issue**: "No module named 'ultralytics'"
**Solution**: Ensure virtual environment is activated: `.\venv\Scripts\activate`

**Issue**: Low detection accuracy
**Solution**: 
- Collect more training images (aim for 500+)
- Train for more epochs (100+)
- Adjust confidence threshold: `--conf 0.3`

**Issue**: Double counting bags
**Solution**: 
- Verify tracking is enabled (it is by default in `inference_video.py`)
- Adjust counting line position
- Lower NMS IoU threshold

## Next Steps

1. **Collect More Data**: Aim for 1000+ annotated images for production
2. **Fine-tune**: Experiment with hyperparameters in `config/model_config.yaml`
3. **Integrate**: Connect to Fillpac monitoring system
4. **Monitor**: Set up alerts for count discrepancies

## Getting Help

- Check `README.md` for detailed documentation
- Review `docs/architecture.md` for technical details
- See `implementation_plan.md` for full verification steps

## Quick Reference Commands

```powershell
# Activate environment
.\venv\Scripts\activate

# Train model
python src/train.py --data config/data.yaml

# Detect in image
python src/inference_image.py --weights models/weights/best.pt --source image.jpg --show

# Count in video
python src/inference_video.py --weights models/weights/best.pt --source video.mp4 --output result.mp4 --log counts.txt

# Batch process images
python src/inference_image.py --weights models/weights/best.pt --source images_folder/ --batch --save results/

# Live camera counting
python src/inference_video.py --weights models/weights/best.pt --source 0
```

---

**Estimated Time to Production**: 
- With 100 images: 1-2 days
- With 1000 images: 3-5 days (including annotation)
- Full deployment: 1 week

Good luck! 🚀
