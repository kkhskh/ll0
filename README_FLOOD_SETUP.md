# Flood Detection Setup Guide

## What You Have So Far

✅ **Sen1Floods11 Repository Cloned** - Contains code and metadata  
❌ **Actual Data Not Downloaded Yet** - Need to get the 14GB dataset  
✅ **Flood Detection Pipeline Created** - Ready to train once data is available

---

## Step 1: Install Google Cloud SDK Tools

You need `gsutil` to download the flood dataset from Google Cloud Storage.

### Option A: Quick Install (pip)
```bash
pip install gsutil
```

### Option B: Full Google Cloud SDK (Recommended)
```bash
# macOS (using Homebrew)
brew install --cask google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

---

## Step 2: Download the Sen1Floods11 Dataset (~14 GB)

Once `gsutil` is installed, run:

```bash
# Create data directory
mkdir -p datasets/floods/sen1floods11

# Download the full dataset (30-60 minutes)
gsutil -m rsync -r gs://sen1floods11 datasets/floods/sen1floods11
```

**What you'll get:**
- 4,831 flood detection chips (512x512 pixels)
- Sentinel-1 SAR imagery (2 bands: VV, VH)
- Sentinel-2 optical imagery (13 bands)
- Hand-labeled flood masks
- Global coverage from multiple flood events

---

## Step 3: Dataset Structure

After download, your directory will look like:

```
datasets/floods/sen1floods11/
├── v1.1/
│   └── data/
│       └── flood_events/
│           └── HandLabeled/
│               ├── S1Hand/          # Sentinel-1 SAR chips
│               │   └── Bolivia_103757_S1Hand.tif
│               ├── S2Hand/          # Sentinel-2 optical chips
│               │   └── Bolivia_103757_S2Hand.tif
│               └── LabelHand/       # Flood masks (ground truth)
│                   └── Bolivia_103757_LabelHand.tif
└── Sen1Floods11_Metadata.geojson   # Event metadata
```

---

## Step 4: Quick Data Verification

Check if download was successful:

```bash
# Count files
find datasets/floods/sen1floods11 -name "*_S1Hand.tif" | wc -l
# Should show ~4831 files

# Check disk usage
du -sh datasets/floods/sen1floods11
# Should show ~14GB
```

---

## Step 5: Train the Flood Detection Model

Once data is downloaded:

```bash
python train_flood_model.py
```

### Training Configuration:
- **Input**: Sentinel-1 (2 bands) + Sentinel-2 (6 bands) = 8 channels
- **Output**: Binary flood mask (512x512)
- **Model**: UNet architecture
- **Loss**: BCE + Dice
- **Batch Size**: 4 (adjust based on GPU memory)
- **Epochs**: 30

### Expected Training Time:
- **GPU (RTX 3080 / V100)**: ~2-3 hours
- **CPU**: Not recommended (will take days)

---

## Step 6: Use Google Colab if No Local GPU

If your Mac can't handle training, upload to Google Colab:

```python
# In Colab notebook
from google.colab import drive
drive.mount('/content/drive')

# Upload flood_detection_pipeline.py and train_flood_model.py to Drive
# Then download dataset directly in Colab:
!gsutil -m rsync -r gs://sen1floods11 /content/floods_data
```

---

## Differences from Wildfire Pipeline

| Aspect | Wildfire (EO4WildFires) | Flood (Sen1Floods11) |
|--------|-------------------------|----------------------|
| **Data Format** | NetCDF (.nc) | GeoTIFF (.tif) |
| **Image Size** | Variable (96-995 pixels) | Fixed 512x512 |
| **Patches** | Extract 256x256 | Use full 512x512 |
| **Weather Data** | Yes (31 timesteps) | No |
| **SAR Bands** | 3 (ASC VV/VH + DESC VV) | 2 (VV, VH) |
| **Optical Bands** | 6 (S2A subset) | 13 (all S2 bands) |
| **Label Type** | Burned area mask | Flood water mask |
| **Data Loading** | xarray | rasterio |

---

## Key Advantages of Flood Dataset

1. **Fixed Size**: No need for complex patch extraction
2. **Simpler Format**: GeoTIFF is easier to work with than NetCDF
3. **Smaller Scale**: Faster to train and iterate
4. **SAR Focus**: SAR can see through clouds (critical for floods!)

---

## Expected Results

After training, you should achieve:
- **IoU (Intersection over Union)**: ~0.70-0.80
- **F1 Score**: ~0.75-0.85
- **Precision**: ~0.70-0.85
- **Recall**: ~0.75-0.90

These are competitive results for flood segmentation!

---

## Next Steps After Flood Model Works

1. ✅ Wildfire Detection (DONE)
2. ✅ Flood Detection (IN PROGRESS)
3. 🔜 Hurricane Detection
4. 🔜 Volcanic Eruption Detection
5. 🔜 Tsunami Damage Assessment
6. 🔜 Hailstorm Detection

---

## Troubleshooting

### Problem: `gsutil: command not found`
**Solution**: Install Google Cloud SDK (see Step 1)

### Problem: Download is too slow
**Solution**: 
- Use `-m` flag for multi-threading: `gsutil -m rsync`
- Download overnight
- Consider using Colab and download directly in cloud

### Problem: Out of memory during training
**Solution**:
- Reduce batch size to 2 or 1
- Use mixed precision training (FP16)
- Train on Colab Pro+ with A100 GPU

### Problem: Can't find label files
**Solution**: Check if data is in `v1.1/` or root directory. The code handles both.

---

## Quick Test (Before Full Training)

Test the pipeline with sample data:

```bash
cd Sen1Floods11
python
```

```python
import rasterio
import matplotlib.pyplot as plt

# Load sample
with rasterio.open('sample/S1/Spain_7370579_S1Hand.tif') as src:
    s1 = src.read()

with rasterio.open('sample/Labels/Spain_7370579_LabelHand.tif') as src:
    label = src.read(1)

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(s1[0], cmap='gray')
plt.title('SAR VV')
plt.subplot(132)
plt.imshow(s1[1], cmap='gray')
plt.title('SAR VH')
plt.subplot(133)
plt.imshow(label, cmap='Blues')
plt.title('Flood Mask')
plt.show()
```

---

## Questions?

Check the original Sen1Floods11 paper:
- https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html

Or reach out if you hit issues!
