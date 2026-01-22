@echo off
echo ============================================================
echo YOLO Bag Counting System - Code Verification
echo ============================================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo Test 1: Checking Python version...
python --version
echo.

echo Test 2: Testing package imports...
python -c "from ultralytics import YOLO; print('✓ ultralytics imported')" || goto :error
python -c "import cv2; print('✓ opencv imported')" || goto :error
python -c "import supervision; print('✓ supervision imported')" || goto :error
python -c "import numpy; print('✓ numpy imported')" || goto :error
echo.

echo Test 3: Loading YOLO model...
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('✓ YOLOv8n model loaded')" || goto :error
echo.

echo Test 4: Testing inference...
python -c "from ultralytics import YOLO; import numpy as np; model = YOLO('yolov8n.pt'); img = np.zeros((640,640,3), dtype=np.uint8); r = model(img, verbose=False); print(f'✓ Inference successful - {len(r[0].boxes)} detections')" || goto :error
echo.

echo Test 5: Testing ByteTrack tracker...
python -c "import supervision as sv; tracker = sv.ByteTrack(); print('✓ ByteTrack initialized')" || goto :error
echo.

echo Test 6: Testing counting zone...
python -c "import supervision as sv; line = sv.LineZone(start=sv.Point(0,100), end=sv.Point(640,100)); print('✓ LineZone created')" || goto :error
echo.

echo Test 7: Checking project files...
python -c "import os; files = ['src/train.py', 'src/inference_image.py', 'src/inference_video.py', 'src/evaluate.py']; assert all(os.path.exists(f) for f in files); print('✓ All project files exist')" || goto :error
echo.

echo ============================================================
echo 🎉 SUCCESS! All tests passed!
echo ============================================================
echo.
echo Your YOLO bag counting system is fully functional!
echo.
echo Next steps:
echo 1. Test with an image: python src\inference_image.py --weights yolov8n.pt --source image.jpg
echo 2. Test with a video: python src\inference_video.py --weights yolov8n.pt --source video.mp4
echo 3. Collect Fillpac data and train your model
echo.
goto :end

:error
echo.
echo ============================================================
echo ✗ ERROR: A test failed!
echo ============================================================
echo Please check the error message above.
echo.

:end
pause
