# NEXRAD Download Fix

## The Problem
The NEXRAD S3 bucket access has changed. The `--no-sign-request` method isn't working.

## Alternative Solutions

### Option 1: Use NOAA's Direct Download (Easier)

Instead of AWS S3, download directly from NOAA:

```bash
# NOAA NCEI Archive (web access)
# Go to: https://www.ncei.noaa.gov/has/HAS.FileAppRouter

# Or use direct links:
wget https://noaa-nexrad-level2.s3.amazonaws.com/2023/05/15/KTLX/KTLX20230515_120000_V06
```

### Option 2: Configure AWS CLI Properly

```bash
# Configure AWS (even for public access)
aws configure
# Leave credentials blank, just set region:
# AWS Access Key ID: [press enter]
# AWS Secret Access Key: [press enter]  
# Default region name: us-east-1
# Default output format: json

# Then try:
aws s3 ls s3://noaa-nexrad-level2/2023/05/15/KTLX/
```

### Option 3: Use Iowa State's Archive (BEST)

Iowa State University mirrors NEXRAD data with easier access:

```bash
# Access via web:
https://mesonet.agron.iastate.edu/archive/data/

# Example: Download for Oklahoma City (KTLX) on May 15, 2023
wget https://mesonet.agron.iastate.edu/archive/data/2023/05/15/GIS/ridge/composite/n0r/

# Or use their Python API:
pip install metpy
```

### Option 4: Use Existing Hail Datasets (FASTEST)

Instead of downloading raw NEXRAD, use pre-processed hail datasets:

**GitHub Datasets:**
1. **Severe Weather Dataset**
   - https://github.com/djgagne/ams-2020-ml-python-course
   - Has preprocessed radar data with hail labels

2. **Storm Events Database**
   - Already processed by researchers
   - CSV format, easy to use

---

## Recommended Fix for Your Project

Since you're on a 4-week timeline, **skip raw NEXRAD** for now and use:

### Quick Solution: NOAA Storm Events CSV + Satellite

```bash
# This WILL work and is much simpler:

# 1. Download hail reports (already in download_hailstorm_data.py)
python download_hailstorm_data.py

# 2. Instead of NEXRAD, use GOES satellite (you already have gsutil)
gsutil -m rsync -r gs://gcp-public-data-goes-16/ABI-L2-MCMIP/2023/135/ datasets/hailstorms/goes/

# 3. Train model on satellite imagery (like your other models)
# Skip complex radar processing for prototype
```

This gives you:
- ✅ Hail event locations (NOAA reports)
- ✅ Satellite imagery (GOES)
- ✅ Can use your existing CNN architecture
- ✅ No complex radar processing needed

---

## For Production (Later):

After your demo works, if you want real radar data:

1. **Sign up for AWS Account** (free tier)
2. **Use Requester Pays** bucket access
3. **Or use NOAA Big Data Program** partnership

But for **Week 3 prototype**, satellite-based hail detection is good enough!

---

## Updated Hailstorm Plan

**SKIP NEXRAD** (too complex for now)

**USE INSTEAD:**
- GOES satellite cloud-top temps (indicates hail)
- NOAA hail reports (ground truth)
- Your existing CNN/LSTM architecture

**Same stochastic approach, simpler data!**
