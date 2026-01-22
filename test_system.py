"""
Quick Test Script to Verify YOLO Bag Counting System
Tests all core components without requiring trained model or dataset
"""

import sys
import importlib

def test_imports():
    """Test if all required packages are installed"""
    print("="*60)
    print("Testing Package Imports")
    print("="*60)
    
    required_packages = {
        'ultralytics': 'YOLO',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'yaml': 'pyyaml',
        'supervision': 'supervision',
        'torch': 'torch',
        'PIL': 'pillow'
    }
    
    results = {}
    for package, display_name in required_packages.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            results[display_name] = ('✓', version)
            print(f"✓ {display_name:20s} - version {version}")
        except ImportError as e:
            results[display_name] = ('✗', str(e))
            print(f"✗ {display_name:20s} - NOT INSTALLED")
    
    print("="*60)
    return all(status == '✓' for status, _ in results.values())


def test_yolo_download():
    """Test YOLO model download and basic inference"""
    print("\n" + "="*60)
    print("Testing YOLO Model Download and Inference")
    print("="*60)
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        print("Downloading YOLOv8n model (this may take a minute)...")
        model = YOLO('yolov8n.pt')
        print("✓ YOLOv8n model loaded successfully")
        
        # Test with dummy image
        print("Testing inference on dummy image...")
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_image, verbose=False)
        print(f"✓ Inference successful - detected {len(results[0].boxes)} objects")
        
        print("="*60)
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        print("="*60)
        return False


def test_supervision():
    """Test Supervision library for tracking"""
    print("\n" + "="*60)
    print("Testing Supervision Library")
    print("="*60)
    
    try:
        import supervision as sv
        
        # Test ByteTrack
        tracker = sv.ByteTrack()
        print("✓ ByteTrack tracker initialized")
        
        # Test LineZone
        line_zone = sv.LineZone(start=sv.Point(0, 100), end=sv.Point(640, 100))
        print("✓ LineZone created")
        
        print("="*60)
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        print("="*60)
        return False


def test_project_structure():
    """Test if project files exist"""
    print("\n" + "="*60)
    print("Testing Project Structure")
    print("="*60)
    
    from pathlib import Path
    
    required_files = [
        'src/data_preparation.py',
        'src/train.py',
        'src/inference_image.py',
        'src/inference_video.py',
        'src/evaluate.py',
        'src/utils.py',
        'config/data.yaml',
        'config/model_config.yaml',
        'requirements.txt',
        'README.md'
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    print("="*60)
    return all_exist


def test_gpu():
    """Test GPU availability"""
    print("\n" + "="*60)
    print("Testing GPU Availability")
    print("="*60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠ No GPU detected - will use CPU (slower)")
            print("  Training and inference will work but be slower")
        
        print("="*60)
        return True
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        print("="*60)
        return False


def test_import_modules():
    """Test importing project modules"""
    print("\n" + "="*60)
    print("Testing Project Module Imports")
    print("="*60)
    
    try:
        # Add src to path
        sys.path.insert(0, 'src')
        
        modules = ['data_preparation', 'train', 'utils']
        for module_name in modules:
            try:
                importlib.import_module(module_name)
                print(f"✓ {module_name}.py imports successfully")
            except Exception as e:
                print(f"✗ {module_name}.py - Error: {e}")
                return False
        
        print("="*60)
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        print("="*60)
        return False


def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# YOLO Bag Counting System - Verification Test")
    print("#"*60 + "\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("GPU Detection", test_gpu),
        ("YOLO Model", test_yolo_download),
        ("Supervision Library", test_supervision),
        ("Project Modules", test_import_modules)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "#"*60)
    print("# Test Summary")
    print("#"*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:12s} - {test_name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print("\n" + "="*60)
    print(f"Results: {total_passed}/{total_tests} tests passed")
    print("="*60)
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Collect images/videos from Fillpac machine")
        print("2. Annotate data using Roboflow")
        print("3. Train model: python src/train.py --data config/data.yaml")
        print("4. Run inference: python src/inference_video.py --weights models/weights/best.pt --source video.mp4")
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Ensure you're in the project directory")
        print("- Check Python version (3.8+ required)")
    
    return total_passed == total_tests


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
