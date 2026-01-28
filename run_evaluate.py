"""
Wrapper script to run YOLO bag detection model evaluation
This script provides a convenient way to evaluate the trained model with proper encoding handling.

Usage:
    python run_evaluate.py
    python run_evaluate.py --conf 0.3 --iou 0.5
    python run_evaluate.py --weights models/weights/best.pt --output outputs/eval_v2

The script will:
1. Evaluate detection performance (mAP, precision, recall)
2. Evaluate counting accuracy (MAE, exact accuracy)
3. Measure inference speed (FPS, inference time)
4. Generate comprehensive reports in outputs/evaluation/
"""

import sys
import os
import io

# Force UTF-8 encoding for stdout/stderr to handle Unicode characters
# This prevents encoding errors on Windows when YOLO outputs special characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Import and run the evaluation
from src.evaluate import main

if __name__ == '__main__':
    print("="*60)
    print("YOLO Bag Detection Model - Evaluation Script")
    print("="*60)
    print("\nStarting evaluation...\n")
    
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Evaluation completed successfully!")
    print("Check outputs/evaluation/ for detailed results")
    print("="*60)
