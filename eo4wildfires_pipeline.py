import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import random
from typing import Tuple, Optional, Dict, Any
import albumentations as A
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EO4WildFiresDataset(Dataset):
    """
    PyTorch Dataset for EO4WildFires wildfire severity prediction
    
    Handles:
    - Multi-sensor satellite imagery (Sentinel-1 SAR + Sentinel-2 MSI)
    - Weather time series data
    - Burned area masks and scalar targets
    - Patch extraction and augmentation
    """
    
    def __init__(
        self,
        data_dir: str,
        file_list: list,
        patch_size: int = 256,
        overlap: float = 0.5,
        mode: str = 'train',
        task: str = 'segmentation',  # 'segmentation', 'regression', 'multitask'
        normalize_stats: Optional[Dict] = None,
        augment: bool = True,
        min_burn_ratio: float = 0.01,  # Skip patches with < 1% burned area
    ):
        self.data_dir = Path(data_dir)
        self.file_list = file_list
        self.patch_size = patch_size
        self.overlap = overlap
        self.mode = mode
        self.task = task
        self.normalize_stats = normalize_stats or {}
        self.augment = augment and mode == 'train'
        self.min_burn_ratio = min_burn_ratio
        
        # Build patch index
        self.patch_index = self._build_patch_index()
        
        # Setup augmentations
        self.setup_augmentations()
        
    def _build_patch_index(self):
        """Build index of all valid patches across all files"""
        patch_index = []
        
        for file_path in self.file_list:
            try:
                # Load dataset to get dimensions
                ds = xr.open_dataset(self.data_dir / file_path, engine='h5netcdf')
                
                # Get spatial dimensions
                height, width = ds.sizes['y'], ds.sizes['x']
                
                # Calculate patch grid
                step_size = int(self.patch_size * (1 - self.overlap))
                
                for y_start in range(0, height - self.patch_size + 1, step_size):
                    for x_start in range(0, width - self.patch_size + 1, step_size):
                        y_end = y_start + self.patch_size
                        x_end = x_start + self.patch_size
                        
                        # For segmentation, check if patch has enough burned area
                        if self.task in ['segmentation', 'multitask']:
                            burned_patch = ds.burned_mask.isel(
                                y=slice(y_start, y_end),
                                x=slice(x_start, x_end)
                            ).values
                            
                            # Skip patches with too little burned area
                            valid_pixels = ~np.isnan(burned_patch)
                            if valid_pixels.sum() == 0:
                                continue
                                
                            burn_ratio = np.nansum(burned_patch) / valid_pixels.sum()
                            if burn_ratio < self.min_burn_ratio and self.mode == 'train':
                                # Randomly skip some low-burn patches
                                if random.random() < 0.8:
                                    continue
                        
                        patch_index.append({
                            'file_path': file_path,
                            'y_start': y_start,
                            'y_end': y_end,
                            'x_start': x_start,
                            'x_end': x_end,
                        })
                
                ds.close()
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
                
        logger.info(f"Built patch index with {len(patch_index)} patches")
        return patch_index
    
    def setup_augmentations(self):
        """Setup albumentations for data augmentation"""
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Sentinel-2 specific augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
                # Add Gaussian noise for SAR data robustness
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            ], additional_targets={
                'burned_mask': 'mask'
            })
        else:
            self.transform = None
    
    def normalize_imagery(self, s1_data: np.ndarray, s2_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize satellite imagery using dataset statistics"""
        
        # Sentinel-1 normalization (dB values, typically -30 to 10)
        if 's1_mean' in self.normalize_stats:
            s1_std = max(self.normalize_stats.get('s1_std', 0.0), 1e-6)
            s1_data = (s1_data - self.normalize_stats['s1_mean']) / s1_std
        else:
            # Default normalization for SAR data
            s1_data = np.clip(s1_data, -30, 10)
            s1_data = (s1_data + 30) / 40  # Scale to [0, 1]
        
        # Sentinel-2 normalization (reflectance values 0-1)
        if 's2_mean' in self.normalize_stats:
            s2_std = max(self.normalize_stats.get('s2_std', 0.0), 1e-6)
            s2_data = (s2_data - self.normalize_stats['s2_mean']) / s2_std
        else:
            # Default normalization for optical data
            s2_data = np.clip(s2_data, 0, 1)
        
        return s1_data, s2_data
    
    def normalize_weather(self, weather_data: np.ndarray) -> np.ndarray:
        """Normalize weather time series"""
        if 'weather_mean' in self.normalize_stats:
            weather_std = max(self.normalize_stats.get('weather_std', 0.0), 1e-6)
            weather_data = (weather_data - self.normalize_stats['weather_mean']) / weather_std
        else:
            # Z-score normalization per feature
            weather_data = (weather_data - np.nanmean(weather_data, axis=0)) / (np.nanstd(weather_data, axis=0) + 1e-8)
        
        # Handle NaNs
        weather_data = np.nan_to_num(weather_data, nan=0.0)
        return weather_data
    
    def __len__(self):
        return len(self.patch_index)
    
    def __getitem__(self, idx):
        patch_info = self.patch_index[idx]
        
        try:
            # Load NetCDF file
            ds = xr.open_dataset(self.data_dir / patch_info['file_path'], engine='h5netcdf')
            
            # Extract patch coordinates
            y_start, y_end = patch_info['y_start'], patch_info['y_end']
            x_start, x_end = patch_info['x_start'], patch_info['x_end']
            
            # Extract Sentinel-1 data (3 bands: ASC VV, ASC VH, DESC VV)
            s1_asc = ds.S1_GRD_A.isel(y=slice(y_start, y_end), x=slice(x_start, x_end)).values
            s1_desc = ds.S1_GRD_D.isel(y=slice(y_start, y_end), x=slice(x_start, x_end)).values
            s1_data = np.concatenate([s1_asc, s1_desc[:1]], axis=0)  # Shape: (3, H, W)
            
            # Extract Sentinel-2 data (6 bands)
            s2_data = ds.S2A.isel(y=slice(y_start, y_end), x=slice(x_start, x_end)).values  # Shape: (6, H, W)
            
            # Extract weather time series (31 timesteps, 9 features)
            weather_features = [
                'RH2M', 'T2M', 'PRECTOTCORR', 'WS2M', 'FRSNO', 
                'GWETROOT', 'SNODP', 'PRECSNOLAND', 'GWETTOP'
            ]
            weather_data = np.stack([
                ds[feat].values for feat in weather_features
            ], axis=1)  # Shape: (31, 9)
            
            # Extract targets
            burned_mask = ds.burned_mask.isel(y=slice(y_start, y_end), x=slice(x_start, x_end)).values
            burned_area = float(ds.BURNED_AREA.values)
            if not np.isfinite(burned_area):
                burned_area = 0.0
            
            ds.close()
            
            # Normalize data
            s1_data, s2_data = self.normalize_imagery(s1_data, s2_data)
            weather_data = self.normalize_weather(weather_data)
            s1_data = np.nan_to_num(s1_data, nan=0.0, posinf=0.0, neginf=0.0)
            s2_data = np.nan_to_num(s2_data, nan=0.0, posinf=0.0, neginf=0.0)
            weather_data = np.nan_to_num(weather_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Combine satellite imagery (12 channels total)
            imagery = np.concatenate([s1_data, s2_data], axis=0)  # Shape: (12, H, W)
            
            # Handle burned mask NaNs and clamp to [0, 1]
            burned_mask_valid = ~np.isnan(burned_mask)
            burned_mask = np.nan_to_num(burned_mask, nan=0.0)
            burned_mask = np.clip(burned_mask, 0.0, 1.0)
            
            # Apply augmentations
            if self.transform:
                # Convert to HWC for albumentations
                imagery_hwc = np.transpose(imagery, (1, 2, 0))
                
                transformed = self.transform(
                    image=imagery_hwc,
                    burned_mask=burned_mask
                )
                
                imagery = np.transpose(transformed['image'], (2, 0, 1))
                burned_mask = transformed['burned_mask']
                burned_mask = np.clip(burned_mask, 0.0, 1.0)
            
            # Convert to tensors
            imagery = torch.from_numpy(imagery).float()
            weather_data = torch.from_numpy(weather_data).float()
            burned_mask = torch.from_numpy(burned_mask).float()
            burned_area = torch.tensor(burned_area).float()
            valid_mask = torch.from_numpy(burned_mask_valid).float()
            
            sample = {
                'imagery': imagery,
                'weather': weather_data,
                'burned_mask': burned_mask,
                'burned_area': burned_area,
                'valid_mask': valid_mask,
                'file_path': patch_info['file_path'],
                'patch_coords': (y_start, y_end, x_start, x_end)
            }
            
            return sample
            
        except Exception as e:
            logger.error(f"Error loading patch {idx}: {e}")
            # Return a dummy sample
            return self.__getitem__(0)


class UNetWithWeather(nn.Module):
    """
    UNet-style model for wildfire severity prediction
    Supports segmentation, regression, and multi-task learning
    """
    
    def __init__(
        self,
        in_channels: int = 12,  # S1 (3) + S2 (6) + weather-derived (3)
        weather_features: int = 9,
        weather_timesteps: int = 31,
        task: str = 'segmentation',
        dropout: float = 0.1
    ):
        super().__init__()
        self.task = task
        
        # Weather processing
        self.weather_processor = nn.Sequential(
            nn.Conv1d(weather_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Weather feature expansion to spatial
        self.weather_spatial = nn.Conv2d(64, 3, kernel_size=1)
        
        # UNet encoder
        self.encoder1 = self._make_encoder_block(in_channels + 3, 64)
        self.encoder2 = self._make_encoder_block(64, 128)
        self.encoder3 = self._make_encoder_block(128, 256)
        self.encoder4 = self._make_encoder_block(256, 512)
        
        # UNet decoder
        self.decoder4 = self._make_decoder_block(512, 256)
        self.decoder3 = self._make_decoder_block(512, 128)  # d4 + e3
        self.decoder2 = self._make_decoder_block(256, 64)   # d3 + e2
        self.decoder1 = self._make_decoder_block(128, 32)   # d2 + e1
        
        # Task-specific heads
        if task in ['segmentation', 'multitask']:
            self.seg_head = nn.Conv2d(32, 1, kernel_size=1)
        
        if task in ['regression', 'multitask']:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.reg_head = nn.Sequential(
                nn.Linear(32 + 64, 128),  # Features + weather
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
    
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, imagery, weather):
        batch_size, _, h, w = imagery.shape
        
        # Process weather data
        weather_feats = self.weather_processor(weather.transpose(1, 2))  # (B, 64)
        
        # Expand weather features spatially
        weather_spatial = weather_feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        weather_spatial = self.weather_spatial(weather_spatial)  # (B, 3, H, W)
        
        # Concatenate imagery with weather features
        x = torch.cat([imagery, weather_spatial], dim=1)  # (B, 15, H, W)
        
        # UNet forward pass
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))
        
        # Decoder
        d4 = self.decoder4(F.interpolate(e4, scale_factor=2))
        d4 = torch.cat([d4, e3], dim=1)
        
        d3 = self.decoder3(F.interpolate(d4, scale_factor=2))
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.decoder2(F.interpolate(d3, scale_factor=2))
        d2 = torch.cat([d2, e1], dim=1)
        
        d1 = self.decoder1(d2)
        
        outputs = {}
        
        if self.task in ['segmentation', 'multitask']:
            seg_output = torch.sigmoid(self.seg_head(d1))
            outputs['segmentation'] = seg_output
        
        if self.task in ['regression', 'multitask']:
            pooled_features = self.global_pool(d1).flatten(1)
            combined_features = torch.cat([pooled_features, weather_feats], dim=1)
            reg_output = self.reg_head(combined_features)
            outputs['regression'] = reg_output
        
        return outputs


class WildfireLoss(nn.Module):
    """Combined loss for wildfire prediction tasks"""
    
    def __init__(self, task='multitask', seg_weight=1.0, reg_weight=1.0):
        super().__init__()
        self.task = task
        self.seg_weight = seg_weight
        self.reg_weight = reg_weight
        
        # Segmentation losses
        self.bce_loss = nn.BCELoss(reduction='none')
        self.mse_loss = nn.MSELoss()
    
    def dice_loss(self, pred, target, valid_mask):
        """Dice loss for segmentation"""
        pred = pred * valid_mask
        target = target * valid_mask
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2 * intersection + 1e-8) / (union + 1e-8)
        return 1 - dice
    
    def forward(self, outputs, targets):
        total_loss = 0
        losses = {}
        
        if self.task in ['segmentation', 'multitask']:
            seg_pred = outputs['segmentation'].squeeze(1)
            seg_pred = torch.nan_to_num(seg_pred, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            burned_mask = targets['burned_mask']
            valid_mask = targets['valid_mask']
            
            # BCE loss with valid mask
            bce = self.bce_loss(seg_pred, burned_mask)
            bce = (bce * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            
            # Dice loss
            dice = self.dice_loss(seg_pred, burned_mask, valid_mask)
            
            seg_loss = bce + dice
            losses['segmentation'] = seg_loss
            total_loss += self.seg_weight * seg_loss
        
        if self.task in ['regression', 'multitask']:
            reg_pred = outputs['regression'].squeeze(1)
            burned_area = targets['burned_area']
            
            # Huber loss for regression
            reg_loss = F.huber_loss(reg_pred, burned_area)
            losses['regression'] = reg_loss
            total_loss += self.reg_weight * reg_loss
        
        losses['total'] = total_loss
        return losses


def create_data_splits(data_dir: str, val_size: float = 0.2, test_size: float = 0.1, random_state: int = 42):
    """Create train/val/test splits stratified by burned area"""
    data_path = Path(data_dir)
    nc_files = list(data_path.glob("*.nc"))
    
    # Load basic info for stratification
    file_info = []
    for file_path in nc_files:
        try:
            ds = xr.open_dataset(file_path, engine='h5netcdf')
            burned_area = float(ds.BURNED_AREA.values)
            file_info.append({
                'file': file_path.name,
                'burned_area': burned_area
            })
            ds.close()
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
    
    df = pd.DataFrame(file_info)
    
    # Create area bins for stratification
    df['area_bin'] = pd.qcut(df['burned_area'], q=5, labels=False, duplicates='drop')
    
    # Split data
    train_files, test_files = train_test_split(
        df['file'].tolist(),
        test_size=test_size,
        stratify=df['area_bin'],
        random_state=random_state
    )
    
    train_df = df[df['file'].isin(train_files)]
    train_files, val_files = train_test_split(
        train_files,
        test_size=val_size / (1 - test_size),
        stratify=train_df['area_bin'],
        random_state=random_state
    )
    
    logger.info(f"Data splits: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    return train_files, val_files, test_files


def compute_normalization_stats(data_dir: str, file_list: list, sample_size: int = 100):
    """Compute normalization statistics from a sample of the dataset"""
    data_path = Path(data_dir)
    
    s1_values, s2_values, weather_values = [], [], []
    
    # Sample files for statistics
    sample_files = random.sample(file_list, min(sample_size, len(file_list)))
    
    for file_name in sample_files:
        try:
            ds = xr.open_dataset(data_path / file_name, engine='h5netcdf')
            
            # Sample patches from each file
            height, width = ds.sizes['y'], ds.sizes['x']
            
            for _ in range(3):  # Sample 3 patches per file
                y_start = random.randint(0, max(0, height - 256))
                x_start = random.randint(0, max(0, width - 256))
                y_end = min(y_start + 256, height)
                x_end = min(x_start + 256, width)
                
                # Extract data
                s1_asc = ds.S1_GRD_A.isel(y=slice(y_start, y_end), x=slice(x_start, x_end)).values
                s1_desc = ds.S1_GRD_D.isel(y=slice(y_start, y_end), x=slice(x_start, x_end)).values
                s1_data = np.concatenate([s1_asc, s1_desc[:1]], axis=0)
                
                s2_data = ds.S2A.isel(y=slice(y_start, y_end), x=slice(x_start, x_end)).values
                
                weather_features = ['RH2M', 'T2M', 'PRECTOTCORR', 'WS2M', 'FRSNO', 
                                    'GWETROOT', 'SNODP', 'PRECSNOLAND', 'GWETTOP']
                weather_data = np.stack([ds[feat].values for feat in weather_features], axis=1)
                
                s1_values.append(s1_data.flatten())
                s2_values.append(s2_data.flatten())
                weather_values.append(weather_data.flatten())
            
            ds.close()
            
        except Exception as e:
            logger.warning(f"Error processing {file_name} for stats: {e}")
    
    # Compute statistics
    s1_all = np.concatenate(s1_values)
    s2_all = np.concatenate(s2_values)
    weather_all = np.concatenate(weather_values)
    
    s1_std = np.nanstd(s1_all)
    s2_std = np.nanstd(s2_all)
    weather_std = np.nanstd(weather_all)

    stats = {
        's1_mean': np.nanmean(s1_all),
        's1_std': max(s1_std, 1e-6),
        's2_mean': np.nanmean(s2_all),
        's2_std': max(s2_std, 1e-6),
        'weather_mean': np.nanmean(weather_all),
        'weather_std': max(weather_std, 1e-6)
    }
    
    logger.info("Computed normalization statistics")
    return stats


# Example usage and training loop
if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'data_dir': '/path/to/eo4wildfires/data',
        'batch_size': 8,
        'patch_size': 256,
        'learning_rate': 1e-3,
        'num_epochs': 100,
        'task': 'multitask',  # 'segmentation', 'regression', 'multitask'
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create data splits
    train_files, val_files, test_files = create_data_splits(CONFIG['data_dir'])
    
    # Compute normalization statistics
    norm_stats = compute_normalization_stats(CONFIG['data_dir'], train_files)
    
    # Create datasets
    train_dataset = EO4WildFiresDataset(
        CONFIG['data_dir'], 
        train_files, 
        patch_size=CONFIG['patch_size'],
        mode='train',
        task=CONFIG['task'],
        normalize_stats=norm_stats,
        augment=True
    )
    
    val_dataset = EO4WildFiresDataset(
        CONFIG['data_dir'], 
        val_files, 
        patch_size=CONFIG['patch_size'],
        mode='val',
        task=CONFIG['task'],
        normalize_stats=norm_stats,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=4
    )
    
    # Initialize model
    model = UNetWithWeather(task=CONFIG['task'])
    model.to(CONFIG['device'])
    
    # Initialize loss and optimizer
    criterion = WildfireLoss(task=CONFIG['task'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['num_epochs'])
    
    # Training loop example
    model.train()
    for epoch in range(CONFIG['num_epochs']):
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            imagery = batch['imagery'].to(CONFIG['device'])
            weather = batch['weather'].to(CONFIG['device'])
            
            # Forward pass
            outputs = model(imagery, weather)
            
            # Compute loss
            targets = {
                'burned_mask': batch['burned_mask'].to(CONFIG['device']),
                'burned_area': batch['burned_area'].to(CONFIG['device']),
                'valid_mask': batch['valid_mask'].to(CONFIG['device'])
            }
            
            losses = criterion(outputs, targets)
            loss = losses['total']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        logger.info(f'Epoch {epoch} completed, Average Loss: {total_loss/len(train_loader):.4f}')
