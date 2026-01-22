# Example: Training YOLO Bag Detection Model

This notebook demonstrates how to train a YOLOv8 model for bag detection on Fillpac machines.

## Setup

```python
# Import required libraries
from ultralytics import YOLO
import yaml
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Load Configuration

```python
# Load data configuration
with open('../config/data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

print("Dataset configuration:")
print(f"  Path: {data_config['path']}")
print(f"  Classes: {data_config['names']}")
print(f"  Number of classes: {data_config['nc']}")
```

## Initialize Model

```python
# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # nano version for speed

# View model architecture
print(model.model)
```

## Train Model

```python
# Training parameters
results = model.train(
    data='../config/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='bag_counter_demo',
    patience=20,
    save=True,
    plots=True
)
```

## Validate Model

```python
# Run validation
metrics = model.val()

print(f"\nValidation Results:")
print(f"  mAP@0.5: {metrics.box.map50:.4f}")
print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"  Precision: {metrics.box.mp:.4f}")
print(f"  Recall: {metrics.box.mr:.4f}")
```

## Test Inference

```python
import cv2
from IPython.display import Image, display

# Test on sample image
test_image = '../data/processed/images/test/sample.jpg'
results = model(test_image)

# Display results
results[0].save('test_result.jpg')
display(Image('test_result.jpg'))

# Count bags
bag_count = sum(1 for det in results[0].boxes if int(det.cls) == 0)
print(f"Bags detected: {bag_count}")
```

## Export Model

```python
# Export to ONNX for deployment
model.export(format='onnx')
print("Model exported to ONNX format")
```

## Visualize Training Results

```python
import matplotlib.pyplot as plt
from PIL import Image as PILImage

# Load training plots
results_path = 'runs/detect/bag_counter_demo'

# Display results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

plots = ['results.png', 'confusion_matrix.png', 'F1_curve.png', 'PR_curve.png']
for ax, plot_name in zip(axes.flat, plots):
    try:
        img = PILImage.open(f'{results_path}/{plot_name}')
        ax.imshow(img)
        ax.set_title(plot_name.replace('.png', '').replace('_', ' ').title())
        ax.axis('off')
    except:
        ax.text(0.5, 0.5, f'{plot_name} not found', ha='center', va='center')
        ax.axis('off')

plt.tight_layout()
plt.show()
```

## Next Steps

1. Fine-tune hyperparameters based on validation results
2. Collect more training data if accuracy is low
3. Test on video streams
4. Deploy to production environment
