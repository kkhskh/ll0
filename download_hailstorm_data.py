"""
Hailstorm Data Download Script
Downloads NEXRAD radar and NOAA hail reports for training
"""

import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Configuration
DATA_DIR = Path("datasets/hailstorms")
NEXRAD_DIR = DATA_DIR / "nexrad"
EVENTS_DIR = DATA_DIR / "events"
GOES_DIR = DATA_DIR / "goes"

os.makedirs(NEXRAD_DIR, exist_ok=True)
os.makedirs(EVENTS_DIR, exist_ok=True)
os.makedirs(GOES_DIR, exist_ok=True)

print("=" * 60)
print("HAILSTORM DATA DOWNLOAD")
print("=" * 60)

# Step 1: Download NOAA Storm Events (Hail Reports)
def download_storm_events(year=2023):
    """
    Download hail event reports from NOAA
    These provide ground truth labels for training
    """
    print(f"\n1. Downloading NOAA Storm Events for {year}...")
    
    base_url = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
    
    # Download details file (has specific hail events)
    details_file = f"StormEvents_details-ftp_v1.0_d{year}_c20240116.csv.gz"
    url = base_url + details_file
    
    output_file = EVENTS_DIR / details_file
    
    print(f"   Downloading from: {url}")
    print(f"   Saving to: {output_file}")
    
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"   ✓ Downloaded: {output_file.stat().st_size / 1e6:.1f} MB")
            
            # Extract and filter hail events
            df = pd.read_csv(output_file, compression='gzip')
            hail_events = df[df['EVENT_TYPE'] == 'Hail']
            
            # Save filtered hail events
            hail_csv = EVENTS_DIR / f"hail_events_{year}.csv"
            hail_events.to_csv(hail_csv, index=False)
            
            print(f"   ✓ Found {len(hail_events)} hail events in {year}")
            print(f"   ✓ Saved to: {hail_csv}")
            
            return hail_events
            
        else:
            print(f"   ✗ Failed to download: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return None


# Step 2: Get NEXRAD Radar Data
def download_nexrad_sample():
    """
    Download sample NEXRAD radar data
    For production, use AWS CLI or boto3
    """
    print("\n2. NEXRAD Radar Data Access...")
    print("   NEXRAD data is on AWS S3: s3://noaa-nexrad-level2/")
    print("   Full download requires AWS CLI:")
    print()
    print("   Install AWS CLI:")
    print("   $ brew install awscli  # or: pip install awscli")
    print()
    print("   Download specific radar station (e.g., KTLX Oklahoma City):")
    print("   $ aws s3 ls s3://noaa-nexrad-level2/2023/05/15/KTLX/ --no-sign-request")
    print("   $ aws s3 cp s3://noaa-nexrad-level2/2023/05/15/KTLX/KTLX20230515_120000_V06 . --no-sign-request")
    print()
    print("   Key NEXRAD Stations for Hail:")
    stations = {
        'KTLX': 'Oklahoma City, OK (most hail)',
        'KFTG': 'Denver, CO',
        'KDFW': 'Dallas-Fort Worth, TX',
        'KICT': 'Wichita, KS',
        'KOUN': 'Norman, OK (research station)'
    }
    
    for station, location in stations.items():
        print(f"   - {station}: {location}")
    
    print()
    print("   For training, download 50-100 hail events from these stations")
    print("   Match timestamps with NOAA hail reports from Step 1")
    print()
    
    # Save station info
    station_info = NEXRAD_DIR / "nexrad_stations.txt"
    with open(station_info, 'w') as f:
        f.write("NEXRAD Stations for Hailstorm Detection\n")
        f.write("=" * 50 + "\n\n")
        for station, location in stations.items():
            f.write(f"{station}: {location}\n")
    
    print(f"   ✓ Station info saved to: {station_info}")


# Step 3: GOES Satellite Access
def download_goes_info():
    """
    Provide information about GOES satellite data access
    """
    print("\n3. GOES Satellite Data Access...")
    print("   GOES-16/17 data is on Google Cloud:")
    print("   gs://gcp-public-data-goes-16/")
    print()
    print("   Install gsutil:")
    print("   $ pip install gsutil")
    print()
    print("   List available data:")
    print("   $ gsutil ls gs://gcp-public-data-goes-16/ABI-L2-MCMIP/2023/135/12/")
    print()
    print("   Download specific scene:")
    print("   $ gsutil cp gs://gcp-public-data-goes-16/ABI-L2-MCMIP/2023/135/12/*.nc .")
    print()
    print("   Key GOES Products for Hail Detection:")
    products = {
        'ABI-L2-MCMIP': 'Cloud & Moisture Imagery (RGB)',
        'ABI-L2-CMIP': 'Cloud-Top Temperature',
        'GLM-L2-LCFA': 'Lightning Detection',
        'ABI-L2-RRQPE': 'Rainfall Rate'
    }
    
    for product, description in products.items():
        print(f"   - {product}: {description}")
    
    # Save product info
    product_info = GOES_DIR / "goes_products.txt"
    with open(product_info, 'w') as f:
        f.write("GOES Satellite Products for Hailstorm Detection\n")
        f.write("=" * 50 + "\n\n")
        for product, description in products.items():
            f.write(f"{product}: {description}\n")
    
    print(f"\n   ✓ Product info saved to: {product_info}")


# Step 4: Create Dataset Manifest
def create_dataset_manifest(hail_events):
    """
    Create a manifest of hail events to download radar/satellite data for
    """
    print("\n4. Creating Dataset Manifest...")
    
    if hail_events is None or len(hail_events) == 0:
        print("   ⚠ No hail events loaded. Run Step 1 first.")
        return
    
    # Select top hail events (by size)
    # Filter for >1 inch hail (significant)
    significant_hail = hail_events[hail_events['MAGNITUDE'] >= 1.0].copy()
    
    # Sort by size
    significant_hail = significant_hail.sort_values('MAGNITUDE', ascending=False)
    
    # Take top 100 events
    top_events = significant_hail.head(100)
    
    # Create manifest with data needed
    manifest = []
    for idx, event in top_events.iterrows():
        manifest_entry = {
            'event_id': event.get('EVENT_ID'),
            'date': event.get('BEGIN_DATE_TIME'),
            'state': event.get('STATE'),
            'latitude': event.get('BEGIN_LAT'),
            'longitude': event.get('BEGIN_LON'),
            'magnitude': event.get('MAGNITUDE'),  # inches
            'nearest_nexrad': get_nearest_nexrad(
                event.get('BEGIN_LAT'), 
                event.get('BEGIN_LON')
            )
        }
        manifest.append(manifest_entry)
    
    # Save manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_file = DATA_DIR / "hail_dataset_manifest.csv"
    manifest_df.to_csv(manifest_file, index=False)
    
    print(f"   ✓ Created manifest with {len(manifest)} hail events")
    print(f"   ✓ Saved to: {manifest_file}")
    print()
    print("   Top 5 Largest Hail Events:")
    for i, row in manifest_df.head(5).iterrows():
        print(f"   {i+1}. {row['date']}: {row['magnitude']}\" hail in {row['state']}")
    
    print()
    print("   Next step: Use this manifest to download NEXRAD + GOES data")
    print("   for each event using AWS CLI and gsutil")


def get_nearest_nexrad(lat, lon):
    """
    Simple function to estimate nearest NEXRAD station
    In reality, use actual station coordinates
    """
    # Simplified - just assign based on region
    if lon < -100 and lat > 35:  # Oklahoma/Kansas area
        return 'KTLX'
    elif lon < -104 and lat > 38:  # Colorado
        return 'KFTG'
    elif lon < -96 and lat < 35:  # Texas
        return 'KDFW'
    else:
        return 'NEAREST'  # Would need actual calculation


# Main execution
if __name__ == "__main__":
    print("\nStarting hailstorm dataset collection...")
    print("This script will guide you through downloading:")
    print("  1. NOAA hail event reports (ground truth)")
    print("  2. NEXRAD radar data (training images)")
    print("  3. GOES satellite data (additional context)")
    print()
    
    # Download hail events (ground truth)
    hail_events = download_storm_events(year=2023)
    
    # NEXRAD info
    download_nexrad_sample()
    
    # GOES info
    download_goes_info()
    
    # Create manifest
    if hail_events is not None:
        create_dataset_manifest(hail_events)
    
    print()
    print("=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Install AWS CLI: brew install awscli")
    print("2. Install gsutil: pip install gsutil")
    print("3. Use the manifest to download NEXRAD radar data")
    print("4. Download corresponding GOES satellite imagery")
    print("5. Train your hailstorm detection model")
    print()
    print("Estimated dataset size: 10-15 GB for 100 events")
    print("=" * 60)
