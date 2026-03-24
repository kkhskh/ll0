"""
Flood Detection Pipeline for Sen1Floods11 Dataset
Adapted from the EO4WildFires wildfire detection pipeline

Handles:
- Sentinel-1 SAR imagery (VV, VH bands)
- Sentinel-2 optical imagery (13 bands)
- Binary flood segmentation masks
- 512x512 pixel chips from global flood events
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import random
from typing import Tuple, Optional, Dict, List
import albumentations as A
from sklearn.model_selection import train_test_split
import logging
import rasterio
from rasterio.windows import Window
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Sen1Floods11Dataset(Dataset):
    """
    PyTorch Dataset for Sen1Floods11 flood detection
    
    Dataset structure:
    - S1 chips: 512x512, 2 bands (VV, VH), Float32, dB values
    - S2 chips: 512x512, 13 bands (B1-B12 + cirrus), UInt16, TOA reflectance
    - QC labels: 512x512, 1 band, Int16 (-1: NoData, 0: Land, 1: Water)
    
    File naming: EVENT_CHIPID_LAYER.tif
    Example: Bolivia_103757_S1Hand.tif
    """
    
    def __init__(
        self,
        data_dir: str,
        chip_ids: List[str],
        mode: str = 'train',
        use_s1: bool = True,
        use_s2: bool = True,
        s2_bands: List[int] = [1, 2, 3, 7, 11, 12],  # Blue, Green, Red, NIR, SWIR1, SWIR2
        patch_size: int = 512,
        augment: bool = True,
        normalize_stats: Optional[Dict] = None,
    ):
        """
        Args:
            data_dir: Path to sen1floods11 data directory
            chip_ids: List of chip IDs (format: "EVENT_CHIPID")
            mode: 'train', 'val', or 'test'
            use_s1: Include Sentinel-1 data
            use_s2: Include Sentinel-2 data
            s2_bands: Which S2 bands to use (0-12)
            patch_size: Size of image patches
            augment: Apply data augmentation (only in train mode)
            normalize_stats: Pre-computed normalization statistics
        """
        self.data_dir = Path(data_dir)
        self.chip_ids = chip_ids
        self.mode = mode
        self.use_s1 = use_s1
        self.use_s2 = use_s2
        self.s2_bands = s2_bands
        self.patch_size = patch_size
        self.normalize_stats = normalize_stats or {}
        self.augment = augment and mode == 'train'
        
        # Verify data exists
        self._verify_data()
        
        # Setup augmentations
        self.setup_augmentations()
        
        logger.info(f"Initialized {mode} dataset with {len(chip_ids)} chips")
    
    def _verify_data(self):
        """Verify that data files exist"""
        # Check for v1.1 structure (recommended)
        if (self.data_dir / 'v1.1').exists():
            self.data_version = 'v1.1'
            self.s1_dir = self.data_dir / 'v1.1' / 'data' / 'flood_events' / 'HandLabeled' / 'S1Hand'
            self.s2_dir = self.data_dir / 'v1.1' / 'data' / 'flood_events' / 'HandLabeled' / 'S2Hand'
            self.label_dir = self.data_dir / 'v1.1' / 'data' / 'flood_events' / 'HandLabeled' / 'LabelHand'
        else:
            # Fall back to root directory structure
            self.data_version = 'v1.0'
            self.s1_dir = self.data_dir / 'S1Hand'
            self.s2_dir = self.data_dir / 'S2Hand'
            self.label_dir = self.data_dir / 'LabelHand'
        
        logger.info(f"Using Sen1Floods11 {self.data_version}")
    
    def setup_augmentations(self):
        """Setup albumentations for data augmentation"""
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Add slight noise for SAR robustness
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
                # Brightness/contrast for S2 (if using optical)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ) if self.use_s2 else A.NoOp(),
            ], additional_targets={
                'mask': 'mask'
            })
        else:
            self.transform = None
    
    def normalize_s1(self, s1_data: np.ndarray) -> np.ndarray:
        """Normalize Sentinel-1 SAR data (dB values)"""
        if 's1_mean' in self.normalize_stats:
            s1_data = (s1_data - self.normalize_stats['s1_mean']) / self.normalize_stats['s1_std']
        else:
            # Default: clip to typical range and scale to [0, 1]
            s1_data = np.clip(s1_data, -30, 10)
            s1_data = (s1_data + 30) / 40
        return s1_data
    
    def normalize_s2(self, s2_data: np.ndarray) -> np.ndarray:
        """Normalize Sentinel-2 optical data (TOA reflectance scaled by 10000)"""
        if 's2_mean' in self.normalize_stats:
            s2_data = (s2_data - self.normalize_stats['s2_mean']) / self.normalize_stats['s2_std']
        else:
            # Default: scale from [0, 10000] to [0, 1]
            s2_data = s2_data / 10000.0
            s2_data = np.clip(s2_data, 0, 1)
        return s2_data
    
    def __len__(self):
        return len(self.chip_ids)
    
    def __getitem__(self, idx):
        chip_id = self.chip_ids[idx]
        
        try:
            channels = []
            
            # Load Sentinel-1 (VV, VH)
            if self.use_s1:
                s1_path = self.s1_dir / f"{chip_id}_S1Hand.tif"
                with rasterio.open(s1_path) as src:
                    s1_data = src.read()  # Shape: (2, 512, 512)
                s1_data = self.normalize_s1(s1_data)
                channels.append(s1_data)
            
            # Load Sentinel-2 (selected bands)
            if self.use_s2:
                s2_path = self.s2_dir / f"{chip_id}_S2Hand.tif"
                with rasterio.open(s2_path) as src:
                    s2_data = src.read()  # Shape: (13, 512, 512)
                # Select specific bands
                s2_data = s2_data[self.s2_bands, :, :]
                s2_data = self.normalize_s2(s2_data)
                channels.append(s2_data)
            
            # Combine all channels
            if len(channels) > 0:
                imagery = np.concatenate(channels, axis=0)
            else:
                raise ValueError("Must use at least S1 or S2 data")
            
            # Load flood mask
            label_path = self.label_dir / f"{chip_id}_LabelHand.tif"
            with rasterio.open(label_path) as src:
                flood_mask = src.read(1)  # Shape: (512, 512)
            
            # Handle label values: -1 (NoData) → 0, 0 (Land) → 0, 1 (Water) → 1
            valid_mask = (flood_mask != -1).astype(np.float32)
            flood_mask = np.clip(flood_mask, 0, 1).astype(np.float32)
            
            # Apply augmentations
            if self.transform:
                # Convert to HWC for albumentations
                imagery_hwc = np.transpose(imagery, (1, 2, 0))
                
                transformed = self.transform(
                    image=imagery_hwc,
                    mask=flood_mask
                )
                
                imagery = np.transpose(transformed['image'], (2, 0, 1))
                flood_mask = transformed['mask']
            
            # Convert to tensors
            imagery = torch.from_numpy(imagery).float()
            flood_mask = torch.from_numpy(flood_mask).float()
            valid_mask = torch.from_numpy(valid_mask).float()
            
            # Calculate flooded area (in pixels)
            flooded_area = torch.sum(flood_mask * valid_mask)
            
            sample = {
                'imagery': imagery,
                'flood_mask': flood_mask,
                'valid_mask': valid_mask,
                'flooded_area': flooded_area,
                'chip_id': chip_id
            }
            
            return sample
            
        except Exception as e:
            logger.error(f"Error loading chip {chip_id}: {e}")
            # Return a dummy sample
            return self.__getitem__(0)


class FloodSegmentationModel(nn.Module):
    """
    UNet-style model for flood segmentation
    Similar architecture to wildfire model but optimized for flood detection
    """
    
    def __init__(
        self,
        in_channels: int = 8,  # S1 (2) + S2 (6)
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Encoder
        self.encoder1 = self._make_encoder_block(in_channels, 64)
        self.encoder2 = self._make_encoder_block(64, 128)
        self.encoder3 = self._make_encoder_block(128, 256)
        self.encoder4 = self._make_encoder_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(512, 1024)
        
        # Decoder
        self.decoder4 = self._make_decoder_block(1024 + 512, 512)
        self.decoder3 = self._make_decoder_block(512 + 256, 256)
        self.decoder2 = self._make_decoder_block(256 + 128, 128)
        self.decoder1 = self._make_decoder_block(128 + 64, 64)
        
        # Output head
        self.output = nn.Conv2d(64, 1, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout)
    
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
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        b = self.dropout(b)
        
        # Decoder with skip connections
        d4 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        # Output
        output = torch.sigmoid(self.output(d1))
        
        return output


class FloodDetectionLoss(nn.Module):
    """Combined loss for flood detection: BCE + Dice"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss(reduction='none')
    
    def dice_loss(self, pred, target, valid_mask):
        """Dice loss for segmentation"""
        pred = pred * valid_mask
        target = target * valid_mask
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2 * intersection + 1e-8) / (union + 1e-8)
        return 1 - dice
    
    def forward(self, pred, target, valid_mask):
        """
        Args:
            pred: (B, 1, H, W) predicted flood probability
            target: (B, H, W) ground truth flood mask
            valid_mask: (B, H, W) valid pixels mask
        """
        pred = pred.squeeze(1)
        
        # BCE loss with valid mask
        bce = self.bce_loss(pred, target)
        bce = (bce * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
        # Dice loss
        dice = self.dice_loss(pred, target, valid_mask)
        
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        
        return total_loss, {'bce': bce.item(), 'dice': dice.item()}


def load_metadata(data_dir: str) -> pd.DataFrame:
    """Load Sen1Floods11 metadata"""
    metadata_path = Path(data_dir).parent / 'Sen1Floods11_Metadata.geojson'
    
    if metadata_path.exists():
        import geopandas as gpd
        gdf = gpd.read_file(metadata_path)
        return pd.DataFrame(gdf.drop(columns='geometry'))
    else:
        logger.warning("Metadata file not found")
        return None


def create_data_splits(
    data_dir: str,
    val_split: float = 0.2,
    test_split: float = 0.1,
    random_state: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/val/test splits from Sen1Floods11 data
    
    Returns:
        train_chips, val_chips, test_chips: Lists of chip IDs
    """
    data_path = Path(data_dir)
    
    # Find all label files to get chip IDs
    if (data_path / 'v1.1').exists():
        label_dir = data_path / 'v1.1' / 'data' / 'flood_events' / 'HandLabeled' / 'LabelHand'
    else:
        label_dir = data_path / 'LabelHand'
    
    label_files = list(label_dir.glob("*_LabelHand.tif"))
    chip_ids = [f.stem.replace('_LabelHand', '') for f in label_files]
    
    logger.info(f"Found {len(chip_ids)} chips")
    
    # Split by event to avoid data leakage
    events = list(set([chip.split('_')[0] for chip in chip_ids]))
    
    train_events, test_events = train_test_split(
        events,
        test_size=test_split,
        random_state=random_state
    )
    
    train_events, val_events = train_test_split(
        train_events,
        test_size=val_split / (1 - test_split),
        random_state=random_state
    )
    
    # Get chips for each split
    train_chips = [c for c in chip_ids if c.split('_')[0] in train_events]
    val_chips = [c for c in chip_ids if c.split('_')[0] in val_events]
    test_chips = [c for c in chip_ids if c.split('_')[0] in test_events]
    
    logger.info(f"Split: Train={len(train_chips)}, Val={len(val_chips)}, Test={len(test_chips)}")
    
    return train_chips, val_chips, test_chips


# Training configuration
if __name__ == "__main__":
    CONFIG = {
        'data_dir': '/Users/shkh/Desktop/DESIGN_HEPHAELION/datasets/floods/sen1floods11',
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'use_s1': True,
        'use_s2': True,
        's2_bands': [1, 2, 3, 7, 11, 12],  # RGB + NIR + SWIR
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Configuration: {CONFIG}")
    print(f"Using device: {CONFIG['device']}")
