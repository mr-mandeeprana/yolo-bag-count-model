"""
Training Script for YOLO Bag Detection Model
Fine-tunes YOLOv8 on Fillpac bag dataset
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch


class BagDetectionTrainer:
    """Trainer for YOLO bag detection model"""
    
    def __init__(self, config_path: str = 'config/model_config.yaml'):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to model configuration YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.results = None
        
    def setup_model(self):
        """Load pre-trained YOLO model"""
        model_name = self.config.get('model', 'yolov8n.pt')
        print(f"Loading model: {model_name}")
        self.model = YOLO(model_name)
        
        # Check for GPU
        device = self.config.get('device', 0)
        if device != 'cpu' and torch.cuda.is_available():
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ Using CPU (training will be slower)")
            self.config['device'] = 'cpu'
    
    def train(self, data_yaml: str = 'config/data.yaml'):
        """
        Train the model
        
        Args:
            data_yaml: Path to dataset configuration
        """
        if self.model is None:
            self.setup_model()
        
        print(f"\n{'='*60}")
        print(f"Starting Training for Fillpac Bag Detection")
        print(f"{'='*60}\n")
        
        # Training parameters from config
        train_params = {
            'data': data_yaml,
            'epochs': self.config.get('epochs', 50),
            'batch': self.config.get('batch', 16),
            'imgsz': self.config.get('imgsz', 640),
            'device': self.config.get('device', 0),
            'optimizer': self.config.get('optimizer', 'AdamW'),
            'lr0': self.config.get('lr0', 0.001),
            'lrf': self.config.get('lrf', 0.01),
            'momentum': self.config.get('momentum', 0.937),
            'weight_decay': self.config.get('weight_decay', 0.0005),
            'patience': self.config.get('patience', 20),
            'save': self.config.get('save', True),
            'save_period': self.config.get('save_period', 10),
            'project': self.config.get('project', 'runs/detect'),
            'name': self.config.get('name', 'bag_counter'),
            'exist_ok': self.config.get('exist_ok', False),
            'verbose': self.config.get('verbose', True),
            'workers': self.config.get('workers', 8),
            'cache': self.config.get('cache', False),
            'amp': self.config.get('amp', True),
        }
        
        # Augmentation parameters
        augmentation_params = {
            'hsv_h': self.config.get('hsv_h', 0.015),
            'hsv_s': self.config.get('hsv_s', 0.7),
            'hsv_v': self.config.get('hsv_v', 0.4),
            'degrees': self.config.get('degrees', 10.0),
            'translate': self.config.get('translate', 0.1),
            'scale': self.config.get('scale', 0.5),
            'shear': self.config.get('shear', 0.0),
            'perspective': self.config.get('perspective', 0.0),
            'flipud': self.config.get('flipud', 0.0),
            'fliplr': self.config.get('fliplr', 0.5),
            'mosaic': self.config.get('mosaic', 1.0),
            'mixup': self.config.get('mixup', 0.1),
        }
        
        # Merge parameters
        all_params = {**train_params, **augmentation_params}
        
        # Start training
        self.results = self.model.train(**all_params)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}\n")
        
        # Save best model to dedicated weights folder
        best_weights = Path(train_params['project']) / train_params['name'] / 'weights' / 'best.pt'
        if best_weights.exists():
            weights_dir = Path('models/weights')
            weights_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(best_weights, weights_dir / 'best.pt')
            print(f"✓ Best model saved to: {weights_dir / 'best.pt'}")
    
    def validate(self, data_yaml: str = 'config/data.yaml'):
        """
        Validate the trained model
        
        Args:
            data_yaml: Path to dataset configuration
        """
        if self.model is None:
            print("Error: No model loaded. Train or load a model first.")
            return
        
        print("\nRunning validation...")
        metrics = self.model.val(data=data_yaml)
        
        print(f"\n{'='*60}")
        print(f"Validation Metrics")
        print(f"{'='*60}")
        print(f"mAP@0.5: {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def export_model(self, format: str = 'onnx'):
        """
        Export model for deployment
        
        Args:
            format: Export format ('onnx', 'torchscript', 'engine', etc.)
        """
        if self.model is None:
            print("Error: No model loaded.")
            return
        
        print(f"\nExporting model to {format.upper()} format...")
        export_path = self.model.export(format=format)
        print(f"✓ Model exported to: {export_path}")
        return export_path


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLO bag detection model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml', 
                       help='Path to model config')
    parser.add_argument('--data', type=str, default='config/data.yaml', 
                       help='Path to data config')
    parser.add_argument('--validate', action='store_true', 
                       help='Run validation after training')
    parser.add_argument('--export', type=str, choices=['onnx', 'torchscript', 'engine'],
                       help='Export format after training')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = BagDetectionTrainer(args.config)
    
    # Train model
    trainer.train(args.data)
    
    # Optional validation
    if args.validate:
        trainer.validate(args.data)
    
    # Optional export
    if args.export:
        trainer.export_model(args.export)
    
    print("\n✓ Training pipeline complete!")
    print(f"\nNext steps:")
    print(f"1. Check training results in runs/detect/bag_counter/")
    print(f"2. Review metrics and visualizations")
    print(f"3. Test inference with src/inference_image.py or src/inference_video.py")


if __name__ == '__main__':
    main()
