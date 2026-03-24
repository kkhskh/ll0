# Quick Setup for EO4WildFires Training on Colab/Kaggle
# Run this cell first to install dependencies and setup

!pip install xarray h5netcdf albumentations

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# For Colab: Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted!")
    
    # Example path - adjust to your Drive structure
    DATA_DIR = "/content/drive/MyDrive/eo4wildfires"
    
except ImportError:
    # For Kaggle or local setup
    print("Not in Colab, using local paths")
    DATA_DIR = "/kaggle/input/eo4wildfires"  # Adjust as needed

# Create directories for outputs
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print("Setup complete!")

# Quick data exploration function
def explore_dataset(data_dir, num_files=3):
    """Quick exploration of the dataset structure"""
    import xarray as xr
    
    data_path = Path(data_dir)
    nc_files = list(data_path.glob("*.nc"))[:num_files]
    
    print(f"Found {len(list(data_path.glob('*.nc')))} total .nc files")
    
    for i, file_path in enumerate(nc_files):
        print(f"\n--- File {i+1}: {file_path.name} ---")
        try:
            ds = xr.open_dataset(file_path, engine='h5netcdf')
            
            print("Dimensions:")
            