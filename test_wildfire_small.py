"""
Quick test of wildfire pipeline on SMALL SUBSET (500 files)
Run locally on your Mac to verify the pipeline works.
"""

import torch
import sys
import os
import json
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
SAVE_EVERY = 1
RESUME = True

# Output directory (Drive if available)
DEFAULT_OUTPUT = "/content/drive/MyDrive/ll0_checkpoints"
OUTPUT_DIR = DEFAULT_OUTPUT if os.path.exists(DEFAULT_OUTPUT) else "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
def save_checkpoint(epoch, model, optimizer, history, best_val, tag="latest"):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "best_val": best_val
    }
    path = os.path.join(OUTPUT_DIR, f"wildfire_{tag}.pth")
    torch.save(ckpt, path)
    return path

def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt

print("\n5. Training...")
start_epoch = 0
best_val = float("inf")
history = {"train_loss": [], "val_loss": []}

latest_ckpt_path = os.path.join(OUTPUT_DIR, "wildfire_latest.pth")
if RESUME and os.path.exists(latest_ckpt_path):
    ckpt = load_checkpoint(latest_ckpt_path, model, optimizer)
    start_epoch = ckpt.get("epoch", -1) + 1
    history = ckpt.get("history", history)
    best_val = ckpt.get("best_val", best_val)
    print(f"   Resuming from epoch {start_epoch}")
for epoch in range(start_epoch, NUM_EPOCHS):
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
    history["train_loss"].append(avg_loss)

    # 6. Validate each epoch
    print("\n6. Validating...")
    if len(val_loader) == 0:
        print("   Skipping validation (no validation batches).")
        avg_val_loss = float("inf")
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
    
    history["val_loss"].append(avg_val_loss)
    
    # Save checkpoints
    if (epoch + 1) % SAVE_EVERY == 0:
        save_checkpoint(epoch, model, optimizer, history, best_val, tag="latest")
    if avg_val_loss < best_val:
        best_val = avg_val_loss
        save_checkpoint(epoch, model, optimizer, history, best_val, tag="best")

# 7. Save final model + history
print("\n7. Saving model...")
final_path = save_checkpoint(NUM_EPOCHS - 1, model, optimizer, history, best_val, tag="final")
with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
    json.dump(history, f, indent=2)
print(f"   Saved: {final_path}")

print("\n✅ Pipeline test complete!")
print(f"\nYour existing pipeline WORKS. Now scale to full dataset when ready.")
