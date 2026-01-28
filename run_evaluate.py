"""
Wrapper script to run YOLO bag detection model evaluation
This script provides a convenient way to evaluate the trained model with proper encoding handling.

Usage:
    python run_evaluate.py
    python run_evaluate.py --config config/video_config.yaml
"""

import sys
import os
import io
import argparse
import subprocess

# Simple color class for Windows console
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

# Force UTF-8 encoding for stdout/stderr to handle Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Bag Detection Model Evaluation Script")
    parser.add_argument('--weights', type=str, default='models/weights/best.pt',
                        help='Path to model weights file')
    parser.add_argument('--data', type=str, default='config/data.yaml',
                        help='Path to dataset.yaml')
    parser.add_argument('--output', type=str, default='outputs/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IOU threshold for NMS')
    parser.add_argument('--config', type=str, default='config/video_config.yaml',
                       help='Path to production video_config.yaml (for ROI/Area alignment)')
    
    args = parser.parse_args()
    
    # Check if config exists
    config_path = args.config if (args.config and os.path.exists(args.config)) else None
    
    # Prepare command to run src/evaluate.py as a subprocess
    cmd = [
        sys.executable, 'src/evaluate.py',
        '--weights', args.weights,
        '--data', args.data,
        '--output', args.output,
        '--conf', str(args.conf),
        '--iou', str(args.iou)
    ]
    if config_path:
        cmd.extend(['--config', config_path])
    
    print(f"\n{Colors.HEADER}{'='*60}")
    print(f"Fillpac YOLO Bag Detection - Automated Evaluator")
    print(f"{'='*60}{Colors.ENDC}")
    print(f"\n{Colors.OKBLUE}Starting evaluation with command:{Colors.ENDC}")
    print(f"  {' '.join(cmd)}\n")
    
    try:
        # Run the evaluation script as a subprocess
        process = subprocess.run(cmd, check=True, text=True)
        
    except subprocess.CalledProcessError as e:
        print(f"\n{Colors.FAIL}[ERROR] Evaluation failed with exit code {e.returncode}{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}[ERROR] An unexpected error occurred: {e}{Colors.ENDC}")
        sys.exit(1)
    
    print("\n" + Colors.OKGREEN + "="*60 + Colors.ENDC)
    print(f"{Colors.OKGREEN}Evaluation completed successfully!{Colors.ENDC}")
    print(f"Check {args.output} for detailed results (report.md and metrics.json)")
    print(Colors.OKGREEN + "="*60 + Colors.ENDC)
