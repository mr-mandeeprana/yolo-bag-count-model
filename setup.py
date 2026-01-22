from setuptools import setup, find_packages

setup(
    name="fill_pac_bag_counting",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "supervision>=0.16.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pillow>=10.0.0",
        "albumentations>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "filterpy>=1.4.5",
        "scikit-image>=0.21.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
    ],
    author="Mandeep Rana",
    description="Real-time bag detection and counting for Fillpac conveyor monitoring",
    python_requires=">=3.8",
)
