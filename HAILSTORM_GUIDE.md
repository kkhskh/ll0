# Hailstorm Detection - Implementation Guide

## Quick Summary

**What You Need:**
1. NEXRAD radar data (AWS S3)
2. NOAA hail reports (ground truth)
3. LSTM/CNN model for sequence prediction
4. Stochastic output for uncertainty

**Timeline:** 1-2 weeks  
**Dataset Size:** 10-15 GB  
**Complexity:** Medium (radar data + time series)

---

## Data Sources (FREE)

### 1. NEXRAD Radar (Primary)
```bash
# AWS S3 bucket (no auth needed)
aws s3 ls s3://noaa-nexrad-level2/ --no-sign-request

# Best stations for hail:
# - KTLX (Oklahoma City) - most hail on Earth
# - KFTG (Denver)
# - KDFW (Dallas-Fort Worth)
# - KICT (Wichita)
```

**What NEXRAD Gives You:**
- Reflectivity (dBZ) - shows precipitation/hail intensity
- Velocity - wind patterns, rotation
- Dual-polarization - hail size estimation
- 5-minute updates

### 2. NOAA Hail Reports (Labels)
```bash
# Download from:
https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

# Contains:
# - Hail size (inches)
# - Location, time
# - Ground truth for training
```

### 3. GOES Satellite (Optional)
```bash
# Google Cloud bucket
gsutil ls gs://gcp-public-data-goes-16/

# Provides:
# - Cloud-top temperature
# - Lightning data
# - Overshooting tops (hail indicator)
```

---

## Where Hailstorms Happen

### Priority Regions (for training data):

**US Great Plains (80% of training data):**
- Oklahoma City, OK - #1 hail-prone city globally
- Denver, CO - #2
- Dallas-Fort Worth, TX
- Wichita, KS
- **Season:** April-July (peak: May)

**Why Focus Here:**
- Most hail events
- Best radar coverage
- Well-documented
- Year-round data

**Secondary (if you want global):**
- Argentina (Córdoba) - Nov-Mar
- Southeast China - May-Jul
- Northern Italy/Alps - Jun-Aug

---

## Stochastic Prediction Model

### Approach: Sequence → Probability

**Input:**
```
Past 6 hours of radar images (60 frames @ 6-min intervals)
+ Current atmospheric state
```

**Output:**
```
P(hail in next 30-90 min) with uncertainty
Max hail size (inches)
Time to hail (minutes)
```

### Two Model Options:

#### Option 1: LSTM (Simpler)
```python
# Process radar images sequentially
LSTM(radar_sequence) → probability, size, time

# Good for: Learning temporal patterns
# Trains faster, easier to debug
```

#### Option 2: 3D CNN (Better)
```python
# Process space + time together
Conv3D(time, height, width) → probability

# Good for: Spatiotemporal patterns
# More powerful but slower
```

### Stochastic Output (Uncertainty)

Use **Monte Carlo Dropout**:
```python
# Multiple forward passes with dropout enabled
predictions = []
for _ in range(20):
    pred = model(radar_data)  # dropout active
    predictions.append(pred)

mean = average(predictions)
std = std_dev(predictions)  # uncertainty!

# Output: P(hail) = 0.75 ± 0.12
```

**Why This Matters:**
- High uncertainty → less confident, warn forecasters
- Low uncertainty → confident prediction, take action
- Reflects atmospheric chaos/unpredictability

---

## Implementation Steps

### Week 3: Hailstorm Detection

**Day 1-2: Data Collection**
```bash
# 1. Download hail reports
python download_hailstorm_data.py

# 2. Install AWS CLI
brew install awscli

# 3. Download NEXRAD radar for top 100 hail events
aws s3 cp s3://noaa-nexrad-level2/2023/05/15/KTLX/ . --recursive --no-sign-request
```

**Day 3-5: Model Training**
```python
# Use hailstorm_stochastic_model.py
# Train LSTM on radar sequences
# Expected accuracy: 75-85%
```

**Day 6-7: Integration**
```python
# Add to your backend API
# Real-time NEXRAD monitoring
# Alert when P(hail) > 0.7
```

---

## Scripts I Created For You

1. **`download_hailstorm_data.py`**
   - Downloads NOAA hail reports
   - Creates manifest of top 100 events
   - Guides NEXRAD radar download

2. **`hailstorm_stochastic_model.py`**
   - LSTM + 3D CNN architectures
   - Stochastic prediction with uncertainty
   - Ready to train on NEXRAD data

---

## Expected Results

### Performance Targets:
- **Accuracy:** 75-85% (hail vs no-hail)
- **False Alarm Rate:** 20-30% (industry standard)
- **Lead Time:** 30-90 minutes warning
- **Uncertainty Estimates:** ±10-15%

### What's "Good Enough":
- Better than random guessing
- Comparable to operational NWS algorithms
- Provides uncertainty for decision-making

---

## Comparison to Other Disasters

| Aspect | Wildfire | Flood | **Hailstorm** |
|--------|----------|-------|---------------|
| **Data Type** | Optical + SAR | SAR | **Radar** |
| **Task** | Segmentation | Segmentation | **Time Series** |
| **Prediction** | Current state | Current state | **Future (30-90min)** |
| **Complexity** | Low | Low | **Medium** |
| **Unique Challenge** | Weather fusion | Water detection | **Temporal prediction** |

**Key Difference:** Hailstorms require **predicting the future**, not just detecting current state.

---

## Why This Approach Works

### 1. Radar "Sees Inside" Storms
- Optical satellites see cloud tops only
- Radar penetrates clouds
- Detects hail formation before it falls

### 2. Temporal Patterns
- Hail develops over 30-60 minutes
- LSTM learns: "when I see THIS pattern evolve, hail follows"
- Similar to weather forecasting

### 3. Stochastic Nature
- Atmosphere is chaotic
- Small changes → big differences
- Uncertainty quantification is essential

---

## Practical Notes

### NEXRAD Data Format:
- **Level II**: Raw radar scans (what you need)
- **File size**: ~10-15 MB per scan (every 5 min)
- **Format**: Archive II (binary, need PyART or Py-ART)
- **Processing**: Use Py-ART or wradlib libraries

### Tools You'll Need:
```bash
pip install pyart  # Radar data processing
pip install wradlib  # Weather radar library
pip install netCDF4  # For GOES satellite
```

### Data Processing:
```python
import pyart

# Load NEXRAD file
radar = pyart.io.read_nexrad_archive('KTLX20230515_120000_V06')

# Get reflectivity
reflectivity = radar.get_field(0, 'reflectivity')

# Preprocess for CNN/LSTM
# Normalize, resize, create sequences
```

---

## Integration with Your Project

### Backend API Endpoint:
```python
@app.post("/predict/hailstorm")
async def predict_hailstorm(radar_sequence: List[np.ndarray]):
    """
    Input: Last 6 hours of radar images
    Output: Hail probability + uncertainty
    """
    model = load_hailstorm_model()
    prediction = model.predict_with_uncertainty(radar_sequence)
    
    return {
        'probability': prediction['hail_probability']['mean'],
        'uncertainty': prediction['hail_probability']['std'],
        'max_size': prediction['max_hail_size']['mean'],
        'time_to_hail': prediction['time_to_hail']['mean'],
        'alert_level': 'HIGH' if prediction > 0.7 else 'MEDIUM' if prediction > 0.5 else 'LOW'
    }
```

### Frontend Display:
```javascript
// Show probability with uncertainty bars
HailProbability: 0.75 ± 0.12
MaxSize: 1.5" ± 0.3"
TimeToHail: 45 ± 15 minutes
AlertLevel: HIGH
```

---

## Bottom Line

**Hailstorms are your most complex disaster type because:**
1. Requires radar data (different from optical/SAR)
2. Needs time series modeling (LSTM/3D CNN)
3. Must predict future, not current state
4. Stochastic output essential (uncertainty matters)

**But it's still achievable in 1-2 weeks if you:**
- Focus on US Great Plains (best data)
- Use simple LSTM first
- Train on 100-200 events (not thousands)
- Accept 75-85% accuracy (good enough for demo)

**Don't overcomplicate. Build working prototype. Improve later.**

---

## Next Steps

1. Run `python download_hailstorm_data.py`
2. Install AWS CLI and download NEXRAD
3. Train `hailstorm_stochastic_model.py`
4. Integrate into your backend

**Timeline: Week 3 of your 4-week plan**
