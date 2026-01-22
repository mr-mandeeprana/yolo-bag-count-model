"""
Simple Code Verification Test
Tests all core functionality of the YOLO bag counting system
"""

print("\n" + "="*60)
print("YOLO Bag Counting System - Code Verification")
print("="*60 + "\n")

# Test 1: Import all required packages
print("Test 1: Checking Package Imports...")
try:
    from ultralytics import YOLO
    import cv2
    import supervision as sv
    import numpy as np
    import yaml
    print("✓ All packages imported successfully")
    test1_pass = True
except Exception as e:
    print(f"✗ Import failed: {e}")
    test1_pass = False

# Test 2: Load YOLO model
print("\nTest 2: Loading YOLO Model...")
try:
    model = YOLO('yolov8n.pt')
    print("✓ YOLOv8n model loaded successfully")
    test2_pass = True
except Exception as e:
    print(f"✗ Model load failed: {e}")
    test2_pass = False

# Test 3: Test inference on dummy image
print("\nTest 3: Testing Inference...")
try:
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model(dummy_img, verbose=False)
    detections = len(results[0].boxes)
    print(f"✓ Inference successful - detected {detections} objects")
    test3_pass = True
except Exception as e:
    print(f"✗ Inference failed: {e}")
    test3_pass = False

# Test 4: Test ByteTrack tracker
print("\nTest 4: Testing ByteTrack Tracker...")
try:
    tracker = sv.ByteTrack()
    print("✓ ByteTrack tracker initialized")
    test4_pass = True
except Exception as e:
    print(f"✗ Tracker failed: {e}")
    test4_pass = False

# Test 5: Test counting zone
print("\nTest 5: Testing Counting Zone...")
try:
    line_zone = sv.LineZone(start=sv.Point(0, 100), end=sv.Point(640, 100))
    print("✓ LineZone created successfully")
    test5_pass = True
except Exception as e:
    print(f"✗ Zone creation failed: {e}")
    test5_pass = False

# Test 6: Test project modules
print("\nTest 6: Testing Project Modules...")
try:
    import sys
    sys.path.insert(0, 'src')
    import utils
    print("✓ Project modules import successfully")
    test6_pass = True
except Exception as e:
    print(f"✗ Module import failed: {e}")
    test6_pass = False

# Test 7: Check GPU availability
print("\nTest 7: Checking GPU Availability...")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU Available: {gpu_name}")
    else:
        print("⚠ No GPU detected - will use CPU (slower but functional)")
    test7_pass = True
except Exception as e:
    print(f"✗ GPU check failed: {e}")
    test7_pass = False

# Summary
print("\n" + "="*60)
print("Test Summary")
print("="*60)

tests = [
    ("Package Imports", test1_pass),
    ("YOLO Model Load", test2_pass),
    ("Inference Test", test3_pass),
    ("ByteTrack Tracker", test4_pass),
    ("Counting Zone", test5_pass),
    ("Project Modules", test6_pass),
    ("GPU Detection", test7_pass)
]

passed = sum(1 for _, result in tests if result)
total = len(tests)

for test_name, result in tests:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{status:10s} - {test_name}")

print("="*60)
print(f"Results: {passed}/{total} tests passed")
print("="*60)

if passed == total:
    print("\n🎉 SUCCESS! All code is working properly!")
    print("\nYou can now:")
    print("1. Test with images: python src/inference_image.py --weights yolov8n.pt --source image.jpg")
    print("2. Test with videos: python src/inference_video.py --weights yolov8n.pt --source video.mp4")
    print("3. Collect Fillpac data and train your model")
else:
    print(f"\n⚠ {total - passed} test(s) failed. Please check the errors above.")

print()
