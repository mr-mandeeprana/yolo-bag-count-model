# YOLO Bag Detection - Evaluation Report

## Summary

### Detection Metrics

- **mAP@0.5**: 0.6333
- **mAP@0.5:0.95**: 0.3734
- **Precision**: 1.0000
- **Recall**: 0.2667

### Counting Accuracy (Deployment-Aligned)

- **Method**: ROI and Area filtering enabled
- **Exact Accuracy**: 0.00%
- **Within ±1**: 22.22%
- **Mean Absolute Error**: 2.00
- **ROI Filter**: Enabled
- **Min Area Filter**: 2000 pixels

### Detection Quality (Confusion Matrix)

Actual counts from IoU matching on test set:
- **True Positives (Correct)**: 6
- **False Positives (Extras)**: 0
- **False Negatives (Missed)**: 13

### Performance

- **Average FPS**: 5.60
- **Mean Inference Time**: 178.47 ms

## Recommendations

Based on the evaluation results:

- [WARNING] Detection accuracy below target (0.85). Consider collecting more training data or training for more epochs.
- [WARNING] Counting accuracy below 90%. Review false positives/negatives and adjust confidence threshold.
- [WARNING] FPS below 30. Consider using a smaller model (YOLOv8n) or exporting to TensorRT for optimization.