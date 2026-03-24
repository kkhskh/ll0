"""
Quick test to verify flood dataset downloaded correctly
No fancy dependencies needed
"""

from pathlib import Path
import sys

print("=" * 60)
print("FLOOD DATASET VERIFICATION")
print("=" * 60)

# Check data directory
data_dir = Path("datasets/floods/sen1floods11")

if not data_dir.exists():
    print(f"❌ ERROR: Data directory not found at {data_dir}")
    sys.exit(1)

print(f"✓ Data directory exists: {data_dir}")

# Check v1.1 structure
v11_dir = data_dir / "v1.1"
if v11_dir.exists():
    print(f"✓ Version 1.1 structure found")
    
    # Check for flood event data
    flood_dir = v11_dir / "data" / "flood_events" / "HandLabeled"
    
    if flood_dir.exists():
        s1_dir = flood_dir / "S1Hand"
        s2_dir = flood_dir / "S2Hand"
        label_dir = flood_dir / "LabelHand"
        
        # Count files
        s1_files = list(s1_dir.glob("*.tif")) if s1_dir.exists() else []
        s2_files = list(s2_dir.glob("*.tif")) if s2_dir.exists() else []
        label_files = list(label_dir.glob("*.tif")) if label_dir.exists() else []
        
        print(f"\n📊 DATASET STATISTICS:")
        print(f"   Sentinel-1 chips: {len(s1_files)}")
        print(f"   Sentinel-2 chips: {len(s2_files)}")
        print(f"   Label masks: {len(label_files)}")
        
        if len(label_files) > 0:
            print(f"\n✓ SUCCESS! Flood dataset is ready to use!")
            print(f"\n📁 Example files:")
            for i, f in enumerate(label_files[:3]):
                chip_id = f.stem.replace("_LabelHand", "")
                print(f"   {i+1}. {chip_id}")
            
            # Check splits
            splits_dir = v11_dir / "splits" / "flood_handlabeled"
            if splits_dir.exists():
                train_csv = splits_dir / "flood_train_data.csv"
                val_csv = splits_dir / "flood_valid_data.csv"
                test_csv = splits_dir / "flood_test_data.csv"
                
                print(f"\n📋 DATA SPLITS:")
                if train_csv.exists():
                    print(f"   ✓ Training split CSV found")
                if val_csv.exists():
                    print(f"   ✓ Validation split CSV found")
                if test_csv.exists():
                    print(f"   ✓ Test split CSV found")
            
            # Check for pre-trained checkpoint
            checkpoint_dir = v11_dir / "checkpoints"
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.cp"))
                if len(checkpoints) > 0:
                    print(f"\n🎁 BONUS: Found {len(checkpoints)} pre-trained checkpoint(s)!")
                    print(f"   You can use this to skip training initially!")
        else:
            print(f"\n⚠️  WARNING: No label files found. Download might be incomplete.")
    else:
        print(f"❌ Flood event directory not found")
else:
    print(f"⚠️  v1.1 directory not found, checking root...")
    # Check alternative structure
    s1_dir = data_dir / "S1Hand"
    if s1_dir.exists():
        print(f"✓ Found v1.0 structure (older format)")

# Calculate total size
total_size = 0
for f in data_dir.rglob("*"):
    if f.is_file():
        total_size += f.stat().st_size

size_gb = total_size / (1024**3)
print(f"\n💾 Total dataset size: {size_gb:.2f} GB")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("1. Install missing Python packages:")
print("   Run in Terminal (NOT in sandbox):")
print("   pip install albumentations rasterio geopandas")
print("")
print("2. Then test the pipeline:")
print("   python -c \"from flood_detection_pipeline import Sen1Floods11Dataset; print('✓ Works!')\"")
print("")
print("3. Start training:")
print("   python train_flood_model.py")
print("=" * 60)
