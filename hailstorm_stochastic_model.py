"""
Simple Stochastic Hailstorm Prediction Model
Input: Sequence of radar images (past 6 hours)
Output: Probability of hail in next 30-90 minutes
"""

import torch
import torch.nn as nn
import numpy as np

class HailstormLSTM(nn.Module):
    """
    Simple LSTM model for hailstorm prediction
    Uses sequence of radar images to predict hail probability
    """
    
    def __init__(
        self,
        input_size=224*224,  # Flattened radar image
        hidden_size=256,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()
        
        # Feature extraction from radar image
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        
        # LSTM for temporal sequence
        self.lstm = nn.LSTM(
            input_size=64*8*8,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Output heads
        self.hail_probability = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Probability 0-1
        )
        
        self.hail_size = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.ReLU()  # Size in inches (0-6)
        )
        
        self.time_to_hail = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.ReLU()  # Minutes (0-90)
        )
    
    def forward(self, radar_sequence):
        """
        Args:
            radar_sequence: (batch, time_steps, 1, H, W)
        
        Returns:
            probability: (batch, 1) - P(hail in next 90 min)
            size: (batch, 1) - Expected max hail size
            time: (batch, 1) - Expected time to hail
        """
        batch_size, time_steps = radar_sequence.shape[0], radar_sequence.shape[1]
        
        # Extract features from each time step
        features = []
        for t in range(time_steps):
            frame = radar_sequence[:, t, :, :, :]
            feat = self.feature_extractor(frame)
            features.append(feat)
        
        # Stack features: (batch, time_steps, feature_dim)
        features = torch.stack(features, dim=1)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Use last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]
        
        # Make predictions
        probability = self.hail_probability(last_hidden)
        size = self.hail_size(last_hidden)
        time = self.time_to_hail(last_hidden)
        
        return {
            'probability': probability,
            'size': size,
            'time_to_hail': time
        }


class HailstormCNN3D(nn.Module):
    """
    Alternative: 3D CNN for spatiotemporal processing
    Processes (time, height, width) directly
    """
    
    def __init__(self, dropout=0.2):
        super().__init__()
        
        # 3D convolutions (time + space)
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 4, 4)),
            
            nn.Flatten()
        )
        
        # Output layers
        self.probability = nn.Sequential(
            nn.Linear(128*2*4*4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, radar_sequence):
        """
        Args:
            radar_sequence: (batch, 1, time, H, W)
        """
        features = self.conv3d(radar_sequence)
        probability = self.probability(features)
        
        return {'probability': probability}


# Training configuration
CONFIG = {
    'sequence_length': 60,  # 6 hours of 6-min intervals
    'image_size': (224, 224),
    'batch_size': 4,
    'learning_rate': 1e-4,
    'num_epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


# Simple usage example
def predict_hailstorm(model, radar_images, device='cpu'):
    """
    Make hail prediction from sequence of radar images
    
    Args:
        model: Trained HailstormLSTM or HailstormCNN3D
        radar_images: List of numpy arrays (past 6 hours)
        device: 'cpu' or 'cuda'
    
    Returns:
        dict with probability, size, time predictions
    """
    model.eval()
    
    # Preprocess images
    sequence = []
    for img in radar_images:
        # Normalize radar reflectivity (-30 to 70 dBZ → 0 to 1)
        normalized = (img + 30) / 100
        normalized = np.clip(normalized, 0, 1)
        sequence.append(normalized)
    
    # Convert to tensor
    sequence = torch.tensor(sequence, dtype=torch.float32)
    sequence = sequence.unsqueeze(0).unsqueeze(2)  # Add batch and channel dims
    sequence = sequence.to(device)
    
    # Make prediction
    with torch.no_grad():
        predictions = model(sequence)
    
    return {
        'hail_probability': predictions['probability'].item(),
        'max_hail_size_inches': predictions.get('size', torch.tensor([0])).item(),
        'time_to_hail_minutes': predictions.get('time_to_hail', torch.tensor([0])).item(),
        'uncertainty': 'Model outputs deterministic prediction. For uncertainty, use ensemble or MC Dropout'
    }


# Stochastic prediction with uncertainty
class StochasticHailPredictor:
    """
    Wrapper for probabilistic predictions with uncertainty estimation
    Uses Monte Carlo Dropout for uncertainty
    """
    
    def __init__(self, model, num_samples=20):
        self.model = model
        self.num_samples = num_samples
        
        # Enable dropout during inference
        self.enable_dropout()
    
    def enable_dropout(self):
        """Enable dropout layers for MC sampling"""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def predict_with_uncertainty(self, radar_sequence):
        """
        Make stochastic prediction with uncertainty estimation
        
        Returns:
            mean_predictions: Average over MC samples
            std_predictions: Standard deviation (uncertainty)
        """
        predictions = []
        
        # Multiple forward passes with dropout
        for _ in range(self.num_samples):
            pred = self.model(radar_sequence)
            predictions.append({
                'probability': pred['probability'].cpu().numpy(),
                'size': pred.get('size', torch.zeros(1)).cpu().numpy(),
                'time': pred.get('time_to_hail', torch.zeros(1)).cpu().numpy()
            })
        
        # Calculate mean and std
        mean_prob = np.mean([p['probability'] for p in predictions])
        std_prob = np.std([p['probability'] for p in predictions])
        
        mean_size = np.mean([p['size'] for p in predictions])
        std_size = np.std([p['size'] for p in predictions])
        
        mean_time = np.mean([p['time'] for p in predictions])
        std_time = np.std([p['time'] for p in predictions])
        
        return {
            'hail_probability': {
                'mean': float(mean_prob),
                'std': float(std_prob),
                'confidence_95': (float(mean_prob - 2*std_prob), float(mean_prob + 2*std_prob))
            },
            'max_hail_size': {
                'mean': float(mean_size),
                'std': float(std_size)
            },
            'time_to_hail': {
                'mean': float(mean_time),
                'std': float(std_time)
            }
        }


# Example usage
if __name__ == "__main__":
    print("Hailstorm Stochastic Prediction Model")
    print("=" * 60)
    
    # Create model
    model = HailstormLSTM()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example input (random data for demonstration)
    batch_size = 2
    time_steps = 60  # 6 hours
    height, width = 224, 224
    
    radar_sequence = torch.randn(batch_size, time_steps, 1, height, width)
    
    # Forward pass
    predictions = model(radar_sequence)
    
    print("\nExample Predictions:")
    print(f"Hail Probability: {predictions['probability'][0].item():.3f}")
    print(f"Max Hail Size: {predictions['size'][0].item():.2f} inches")
    print(f"Time to Hail: {predictions['time_to_hail'][0].item():.1f} minutes")
    
    # Stochastic prediction with uncertainty
    print("\nStochastic Prediction (with uncertainty):")
    stochastic_model = StochasticHailPredictor(model, num_samples=20)
    uncertain_pred = stochastic_model.predict_with_uncertainty(radar_sequence)
    
    print(f"Hail Probability: {uncertain_pred['hail_probability']['mean']:.3f} ± {uncertain_pred['hail_probability']['std']:.3f}")
    print(f"95% Confidence: {uncertain_pred['hail_probability']['confidence_95']}")
    
    print("\n" + "=" * 60)
    print("Key Features:")
    print("  - LSTM processes temporal sequence (past 6 hours)")
    print("  - Predicts: probability, size, timing")
    print("  - MC Dropout provides uncertainty estimates")
    print("  - Can be trained on NEXRAD radar data")
    print("=" * 60)
