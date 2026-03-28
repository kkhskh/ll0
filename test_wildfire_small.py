"""
Quick test of wildfire pipeline on SMALL SUBSET (500 files)
Run locally on your Mac to verify the pipeline works.
"""

import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your existing pipeline
from eo4wildfires_pipeline import (
    EO4WildFiresDataset,
    UNetWithWeather,
    WildfireLoss,
    create_data_splits,
    compute_normalization_stats
)

# Config
DATA_ROOT = "/Users/shkh/Downloads/eo4wildfires"
NUM_FILES = 200  # Small subset for testing
BATCH_SIZE = 2
NUM_EPOCHS = 3
PATCH_SIZE = 128
TASK = "segmentation"

print(f"Testing wildfire pipeline on {NUM_FILES} files...")

# 1. Create small splits
print("\n1. Creating data splits...")
all_files = sorted([p.name for p in Path(DATA_ROOT).glob("*.nc")])[:NUM_FILES]
split_idx = max(1, int(0.8 * len(all_files)))
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

print(f"   Train: {len(train_files)}, Val: {len(val_files)}")

# 2. Compute normalization stats (quick, just 50 samples)
print("\n2. Computing normalization stats...")
norm_stats = compute_normalization_stats(DATA_ROOT, train_files[:50])

# 3. Create datasets
print("\n3. Creating datasets...")
train_dataset = EO4WildFiresDataset(
    DATA_ROOT,
    train_files,
    patch_size=PATCH_SIZE,
    overlap=0.0,
    mode='train',
    task=TASK,
    normalize_stats=norm_stats,
    augment=True,
    min_burn_ratio=0.0
)

val_dataset = EO4WildFiresDataset(
    DATA_ROOT,
    val_files,
    patch_size=PATCH_SIZE,
    overlap=0.0,
    mode='val',
    task=TASK,
    normalize_stats=norm_stats,
    augment=False,
    min_burn_ratio=0.0
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# 4. Create model
print("\n4. Creating model...")
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"   Device: {device}")

model = UNetWithWeather(
    in_channels=10,  # 4 (S1) + 6 (S2)
    weather_features=9,
    weather_timesteps=31,
    task=TASK,
    dropout=0.3
).to(device)

criterion = WildfireLoss(task=TASK, seg_weight=1.0, reg_weight=0.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 5. Train for 3 epochs
print("\n5. Training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch in pbar:
        imagery = batch['imagery'].to(device)
        weather = batch['weather'].to(device)
        burn_mask = batch['burned_mask'].to(device)
        burned_area = batch['burned_area'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(imagery, weather)
        loss_dict = criterion(outputs, {
            'burned_mask': burn_mask,
            'burned_area': burned_area,
            'valid_mask': valid_mask
        })
        loss = loss_dict['total']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = train_loss / len(train_loader)
    print(f"   Epoch {epoch+1} avg loss: {avg_loss:.4f}")

# 6. Test on validation
print("\n6. Validating...")
if len(val_loader) == 0:
    print("   Skipping validation (no validation batches).")
else:
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            imagery = batch['imagery'].to(device)
            weather = batch['weather'].to(device)
            burn_mask = batch['burned_mask'].to(device)
            burned_area = batch['burned_area'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            outputs = model(imagery, weather)
            loss_dict = criterion(outputs, {
                'burned_mask': burn_mask,
                'burned_area': burned_area,
                'valid_mask': valid_mask
            })
            val_loss += loss_dict['total'].item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"   Validation loss: {avg_val_loss:.4f}")

# 7. Save test model
print("\n7. Saving model...")
torch.save(model.state_dict(), 'wildfire_test_model.pth')
print("   Saved: wildfire_test_model.pth")

print("\n✅ Pipeline test complete!")
print(f"\nYour existing pipeline WORKS. Now scale to full dataset when ready.")
