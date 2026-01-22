"""
LabelImg Launcher
Simple script to launch LabelImg annotation tool
"""

import sys
import os

# Add labelImg to path
labelimg_path = r"C:\Users\mrman\AppData\Roaming\Python\Python312\site-packages"
if labelimg_path not in sys.path:
    sys.path.insert(0, labelimg_path)

# Import and run labelImg
try:
    from labelImg import labelImg
    from labelImg.labelImg import MainWindow
    from PyQt5.QtWidgets import QApplication
    
    print("=" * 60)
    print("Starting LabelImg Annotation Tool")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Click 'Open Dir' → Select: data/raw/extracted_frames/")
    print("2. Click 'Change Save Dir' → Select: data/processed/labels/train/")
    print("3. Change format to 'YOLO' (click PascalVOC dropdown)")
    print("4. Press 'W' to draw box, 'D' for next image")
    print("5. Type 'bag' as label (first time only)")
    print("6. Press Ctrl+S to save")
    print("=" * 60)
    print("\nLaunching LabelImg...\n")
    
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
    
except ImportError as e:
    print(f"Error importing LabelImg: {e}")
    print("\nTrying alternative method...")
    
    # Alternative: Run labelImg directly
    from labelImg.labelImg import main
    main()
