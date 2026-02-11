# YOLO Bag Counting System - Complete Documentation

This document contains all instructions for training, deploying, and troubleshooting the BEUMER Fillpac bag counting system.

---

## 📋 Table of Contents

1. [Quick Start](#1-quick-start)
2. [Training Guide](#2-training-guide)
3. [Deployment & Camera Setup](#3-deployment--camera-setup)
4. [Troubleshooting](#4-troubleshooting)
5. [RTSP Camera Troubleshooting (Local)](#5-rtsp-camera-troubleshooting-local)
6. [System Architecture](#6-system-architecture)

---

## 1. Quick Start

### Environment Setup

1. **Open PowerShell** in the project folder.
2. **Create & Activate Virtual Environment**:

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```powershell
   pip install -r requirements.txt
   ```

### Run a Test

- **Image**: `python src/inference_image.py --weights models/weights/best.pt --source "data/raw/sample.jpg" --show`
- **Video**: `python src/inference_video.py --weights models/weights/best.pt --source "video.mp4" --output "outputs/result.mp4"`

---

## 2. Training Guide

### Annotation

1. Run `labelImg` to annotate your frames in `data/raw/`.
2. Save annotations in **YOLO format** to `data/processed/labels/train/`.

### Dataset Organization

```bash
python src/data_preparation.py --output data/processed
```

### Training

```bash
# Recommended: 50 epochs
python src/train.py --data config/data.yaml --epochs 50 --batch 16
```

Model weights will be saved to `models/weights/best.pt`.

---

## 3. Deployment & Camera Setup

### RTSP Camera Integration

Use the following format for industrial cameras:
`rtsp://username:password@camera-ip:port/stream_path`

### Edge Deployment (NVIDIA Jetson)

Export to TensorRT for 60+ FPS:

```python
from ultralytics import YOLO
YOLO('models/weights/best.pt').export(format='engine', device=0, half=True)
```

### Hardware Recommendations

- **Production**: NVIDIA Jetson Orin Nano / Xavier NX.
- **Testing**: Desktop with RTX GPU.

---

## 4. Troubleshooting

| Issue | Solution |
|-------|----------|
| **0 Bags Counted** | Lower `--conf` to 0.25; Check counting line position in video. |
| **Video Slow** | Run on GPU; Use TensorRT; Reduce resolution (`imgsz=416`). |
| **Memory Errors** | Reduce batch size to 4 or 8 in `src/train.py`. |
| **No Module found** | Ensure virtual environment is activated. |

---

## 5. RTSP Camera Troubleshooting (Local)

If you see a `RuntimeError` or `CAP_IMAGES` error when connecting to your camera, follow these steps on **your local computer**:

1. **Verify Connectivity**:
   Ensure you can ping the camera: `ping 192.168.1.5`
2. **Run Diagnostics**:
   Use the dedicated tool to check the stream and save a test frame:

   ```powershell
   python scripts/camera_diagnostic.py --config config/video_config.yaml
   ```

3. **Common Fixes**:
   - **URL Encoding**: If your password has special characters (like `@`), it must be encoded as `%40`.
   - **Protocol**: Try removing `&proto=Onvif` from the URL if the connection fails.
   - **Backend**: The system now explicitly uses FFMPEG for RTSP to improve stability.

---

## 6. System Architecture

The system uses **YOLOv8** for detection and **ByteTrack** for robust object tracking.

1. **Input**: Frames are captured and resized to 478x850.
2. **Processing**: YOLO detects bags; ByteTrack assigns unique IDs.
3. **Counting**: A virtual line/ROI triggers increments when center-points cross the boundary.
4. **Output**: Real-time visualization and count logging to `.txt`.

---

For technical support, contact the ML engineering team.
