"""
Production Pipeline Orchestrator for YOLO Bag Counting System
Unified entry point for data prep, training, evaluation, and deployment.
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path
import logging
from typing import Optional

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ProductionPipeline:
    """Orchestrates the ML lifecycle for industrial bag counting"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.python_exec = sys.executable
        
    def _run_script(self, script_name: str, args: list):
        """Helper to run project scripts as subprocesses"""
        script_path = self.project_root / 'src' / script_name
        if not script_path.exists():
            # Try root if not in src
            script_path = self.project_root / script_name
            
        command = [self.python_exec, str(script_path)] + args
        logger.info(f"Executing: {' '.join(command)}")
        
        try:
            result = subprocess.run(command, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error(f"Pipe error in {script_name}: {e}")
            return False

    def prepare(self, raw_dir: str, output_dir: str):
        """Step 1: Prepare data (extract, convert, split)"""
        logger.info("--- Starting Data Preparation Phase ---")
        return self._run_script('data_preparation.py', ['--raw', raw_dir, '--output', output_dir])

    def train(self, config: str, data: str):
        """Step 2: Train model"""
        logger.info("--- Starting Model Training Phase ---")
        return self._run_script('train.py', ['--config', config, '--data', data, '--validate'])

    def evaluate(self, weights: str, data: str, config: str):
        """Step 3: Evaluate performance"""
        logger.info("--- Starting Industrial Evaluation Phase ---")
        return self._run_script('evaluate.py', ['--weights', weights, '--data', data, '--config', config])

    def run(self, source: str, weights: str, config: str, conf: Optional[float] = None):
        """Step 4: Real-time Deployment"""
        logger.info("--- Starting Production Inference ---")
        cmd_args = ['--source', source, '--weights', weights, '--config', config]
        if conf is not None:
            cmd_args.extend(['--conf', str(conf)])
        return self._run_script('inference_video.py', cmd_args)

def main():
    pipeline = ProductionPipeline()
    parser = argparse.ArgumentParser(
        description='🚀 YOLO Bag Counting Production Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Prepare Data:
    python src/pipeline.py prepare --raw data/raw --output data/processed
    
  Train Model:
    python src/pipeline.py train --config config/model_config.yaml --data config/data.yaml
    
  Evaluate:
    python src/pipeline.py evaluate --weights models/weights/best.pt --config config/video_config.yaml
    
  Production Run:
    python src/pipeline.py run --source "rtsp://..." --config config/video_config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='phase', help='Pipeline phase')

    # Prepare Subcommand
    prep_parser = subparsers.add_parser('prepare', help='Prepare dataset')
    prep_parser.add_argument('--raw', type=str, default='data/raw', help='Path to raw data')
    prep_parser.add_argument('--output', type=str, default='data/processed', help='Path to processed output')

    # Train Subcommand
    train_parser = subparsers.add_parser('train', help='Train YOLO model')
    train_parser.add_argument('--config', type=str, default='config/model_config.yaml', help='Model config file')
    train_parser.add_argument('--data', type=str, default='config/data.yaml', help='Data description file')

    # Evaluate Subcommand
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate industrial metrics')
    eval_parser.add_argument('--weights', type=str, default='models/weights/best.pt', help='Model weights')
    eval_parser.add_argument('--data', type=str, default='config/data.yaml', help='Data config')
    eval_parser.add_argument('--config', type=str, default='config/video_config.yaml', help='Production config')

    # Run Subcommand
    run_parser = subparsers.add_parser('run', help='Start production counter')
    run_parser.add_argument('--source', type=str, required=True, help='Video source (RTSP/File/ID)')
    run_parser.add_argument('--weights', type=str, default='models/weights/best.pt', help='Model weights')
    run_parser.add_argument('--config', type=str, default='config/video_config.yaml', help='Production config')
    run_parser.add_argument('--conf', type=float, help='Confidence threshold')

    args = parser.parse_args()

    if args.phase == 'prepare':
        pipeline.prepare(args.raw, args.output)
    elif args.phase == 'train':
        pipeline.train(args.config, args.data)
    elif args.phase == 'evaluate':
        pipeline.evaluate(args.weights, args.data, args.config)
    elif args.phase == 'run':
        pipeline.run(args.source, args.weights, args.config, args.conf)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
