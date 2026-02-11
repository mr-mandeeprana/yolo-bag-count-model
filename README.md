# YOLO-Based Bag Counting System for BEUMER Fillpac Machines

## Overview

This project implements a real-time bag counting system using YOLOv8 object detection for BEUMER Group's Fillpac bagging machines. The system processes video feeds from production lines to accurately detect and count bags, supporting quality control and production monitoring for high-speed operations (up to 2,600 bags/hour).

### Target Machines

- **Fillpac FFS (Form-Fill-Seal)**: Creates bags from PE tubular film, fills, and seals in one process
- **Fillpac R (Rotating)**: High-capacity valve bag filling with rotating modules (300-6,000 bags/hour)

### Key Features

- **Real-time Detection**: YOLOv8-based object detection optimized for industrial environments
- **Accurate Counting**: ByteTrack integration for unique bag tracking and counting
- **Robust Performance**: Handles dust, varying lighting, and high-speed conveyor movement
- **Flexible Deployment**: Supports edge devices (NVIDIA Jetson) and cloud processing
- **Production Monitoring**: Virtual line/zone counting with alerts for production discrepancies

## System Architecture

### Model Structure

The system uses **YOLOv8** (You Only Look Once v8) from Ultralytics:

- **Backbone**: CSPDarknet with C2f modules for feature extraction
- **Neck**: Path Aggregation Network (PAN) for multi-scale feature fusion
- **Head**: Detection heads for bounding box regression and classification
- **Input**: 640x640 images from Fillpac camera feeds
- **Output**: Bounding boxes, confidence scores, and class predictions

### Processing Pipeline

```
Camera Feed → Frame Capture → YOLO Detection → Object Tracking → 
Zone/Line Counting → Count Logging → Alerts (if needed)
```

## Project Structure

```
Yolo_bag_count_model/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config/
│   ├── data.yaml               # Dataset configuration
│   └── model_config.yaml       # Model hyperparameters
├── data/
│   ├── raw/                    # Raw images/videos from Fillpac
│   ├── processed/              # Annotated dataset
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   └── augmented/              # Augmented training data
├── src/
│   ├── data_preparation.py     # Dataset utilities
│   ├── train.py                # Model training script
│   ├── inference_image.py      # Image inference
│   ├── inference_video.py      # Video inference with counting
│   ├── evaluate.py             # Model evaluation
│   └── utils.py                # Helper functions
├── models/
│   └── weights/                # Trained model weights
├── outputs/
│   ├── logs/                   # Training logs
│   ├── results/                # Detection results
│   └── counts/                 # Bag count records
└── docs/
    ├── architecture.md         # Detailed architecture
    ├── deployment.md           # Deployment guide
    └── training_guide.md       # Training instructions
```

## Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd "c:\Users\mrman\OneDrive\Desktop\Beumer Data\Yolo_bag_count_model"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Collect images/videos from your Fillpac machine and annotate bags:

```bash
python src/data_preparation.py --input data/raw --output data/processed
```

### 3. Training

Train the YOLOv8 model on your dataset:

```bash
python src/train.py --data config/data.yaml --epochs 50 --batch 16
```

### 4. Inference

**For images:**

```bash
python src/inference_image.py --weights models/weights/best.pt --source path/to/image.jpg
```

**For videos (with counting):**

```bash
python src/inference_video.py --weights models/weights/best.pt --source path/to/video.mp4
```

## Technical Specifications

### Model Variants

| Variant | Parameters | Speed (FPS) | Accuracy (mAP) | Use Case |
|---------|-----------|-------------|----------------|----------|
| YOLOv8n | 3.2M | 80+ | ~0.85 | Edge devices, real-time |
| YOLOv8s | 11.2M | 60+ | ~0.88 | Balanced performance |
| YOLOv8m | 25.9M | 40+ | ~0.91 | Higher accuracy needed |

### Performance Targets

- **Detection Accuracy**: mAP@0.5 > 0.85
- **Counting Accuracy**: >98% for controlled environments
- **Processing Speed**: 30+ FPS for real-time monitoring
- **Latency**: <50ms per frame on GPU

### Hardware Requirements

**Training:**

- GPU: NVIDIA RTX 3060 or better (8GB+ VRAM)
- RAM: 16GB+
- Storage: 50GB+ for datasets

**Deployment:**

- Edge: NVIDIA Jetson Nano/Xavier (for factory floor)
- Cloud: Any GPU instance (AWS, Azure, GCP)
- Camera: 1080p, 30 FPS minimum

## Dataset Recommendations

### Collection Guidelines

- **Quantity**: 1,000-2,000 annotated images minimum
- **Diversity**: Multiple angles, lighting conditions, bag types
- **Scenarios**: Empty conveyor, single bags, overlapping bags, dust/occlusions
- **Video**: Capture at production speed for realistic training

### Public Datasets (for pre-training/augmentation)

- **Roboflow Universe**: "Bag" datasets (2,145 images available)
- **Bag6k Dataset**: 21,457 annotated bags from videos
- **COCO Dataset**: "handbag" class for transfer learning

### Annotation Tools

- **Roboflow**: Cloud-based, exports YOLO format directly
- **LabelImg**: Desktop tool for bounding boxes
- **CVAT**: Advanced annotation with tracking support

## Deployment Options

### Edge Deployment (Recommended for Fillpac)

Deploy on NVIDIA Jetson at the production line:

```bash
# Export to TensorRT for optimization
python src/export_model.py --weights models/weights/best.pt --format engine

# Run inference on Jetson
python src/inference_video.py --weights models/weights/best.engine --source rtsp://camera-ip
```

### Cloud Deployment

For centralized monitoring across multiple Fillpac machines:

- Stream video to cloud (AWS Kinesis, Azure Stream Analytics)
- Run inference on GPU instances
- Store counts in database (PostgreSQL, InfluxDB)
- Visualize with dashboards (Grafana, custom web app)

## Integration with Fillpac Systems

### Production Monitoring

- Compare YOLO counts with Fillpac machine counters
- Alert on discrepancies (e.g., >2% deviation)
- Log counts for quality control and reporting

### API Integration

Expose counting service via REST API:

```python
# Example endpoint
POST /api/count
{
  "video_url": "rtsp://fillpac-camera-1",
  "duration": 60  # seconds
}

Response:
{
  "bag_count": 42,
  "confidence": 0.96,
  "timestamp": "2026-01-20T11:07:00Z"
}
```

## Troubleshooting

### Common Issues

**Low Detection Accuracy:**

- Increase dataset size and diversity
- Adjust confidence threshold (default 0.5)
- Fine-tune for longer (100+ epochs)

**Missed Counts (False Negatives):**

- Check camera angle and resolution
- Adjust virtual line/zone position
- Lower confidence threshold cautiously

**Double Counting:**

- Verify ByteTrack tracker settings
- Adjust IOU threshold for NMS
- Review zone crossing logic

**Performance Issues:**

- Use smaller model (YOLOv8n)
- Reduce input resolution (e.g., 416x416)
- Export to TensorRT/ONNX

## Documentation

All project instructions, training guides, and deployment steps have been consolidated into a single comprehensive document:

- [📖 **Complete Project Documentation**](docs/DOCUMENTATION.md)

This guide covers:

- environment setup and quick start
- training and annotation workflow
- deployment to edge devices (NVIDIA Jetson)
- camera setup and RTSP streaming
- troubleshooting and performance tuning

## License

This project is for internal use with BEUMER Fillpac machines. Ensure compliance with Ultralytics AGPL-3.0 license for YOLOv8.

## Contact

For questions or support, contact your ML engineering team.
