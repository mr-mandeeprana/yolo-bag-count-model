@echo off
REM LabelImg Launcher Batch File
echo ============================================================
echo Starting LabelImg Annotation Tool
echo ============================================================
echo.
echo Instructions:
echo 1. Click 'Open Dir' and select: data\raw\extracted_frames\
echo 2. Click 'Change Save Dir' and select: data\processed\labels\train\
echo 3. Change format to 'YOLO' (click PascalVOC dropdown)
echo 4. Press 'W' to draw box, 'D' for next image
echo 5. Type 'bag' as label (first time only)
echo 6. Press Ctrl+S to save
echo ============================================================
echo.

REM Try different methods to launch LabelImg
python -c "from labelImg.labelImg import main; main()"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo LabelImg failed to launch.
    echo Trying alternative method...
    python -m labelImg.labelImg
)

pause
