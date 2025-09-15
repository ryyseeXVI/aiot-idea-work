"""
AI Model Training Module
Trains PyTorch neural network for IoT relay control decisions
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddedRelayMLP(nn.Module):
    """
    Multi-layer perceptron for relay control decisions
    Optimized for embedded deployment
    """
    
    def __init__(self, input_size=6, hidden_sizes=[32, 16, 8], dropout_rate=0.2):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.extend([
            nn.Linear(prev_size, 1),
            nn.Sigmoid()
        ])
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def generate_synthetic_data(num_samples=10000, save_path="synthetic_training_data.json") -> List[Dict]:
    """
    Generate synthetic training data for IoT system
    Simulates realistic sensor patterns and control decisions
    """
    logger.info(f"Generating {num_samples} synthetic training samples...")
    
    np.random.seed(42)  # For reproducibility
    data = []
    
    for i in range(num_samples):
        # Generate realistic sensor values
        hour = np.random.randint(0, 24)
        season = np.random.choice(['winter', 'spring', 'summer', 'fall'])
        
        # Temperature with seasonal and daily patterns
        if season == 'winter':
            base_temp = np.random.normal(20, 3)
        elif season == 'summer':
            base_temp = np.random.normal(28, 4)
        else:
            base_temp = np.random.normal(24, 3)
        
        # Daily temperature variation
        daily_factor = 2 * np.sin((hour - 6) * np.pi / 12)
        temperature = max(15, min(40, base_temp + daily_factor + np.random.normal(0, 1)))
        
        # Humidity (inversely correlated with temperature)
        humidity = max(20, min(90, 80 - 0.5 * temperature + np.random.normal(0, 5)))
        
        # Vibration (occasional spikes)
        if np.random.random() < 0.1:
            vibration = np.random.randint(700, 1000)  # High vibration
        else:
            vibration = np.random.randint(300, 600)  # Normal vibration
        
        # Water leak (rare event)
        water_leak = np.random.random() < 0.02
        
        # Off-peak hours (22:00 to 06:00)
        off_peak = hour >= 22 or hour < 6
        
        # Alert conditions
        alert_active = (
            temperature > 30 or 
            vibration > 700 or 
            water_leak or
            np.random.random() < 0.1
        )
        
        # Relay decision logic (ground truth)
        relay_on = False
        
        # Temperature-based decision
        if season == 'summer':
            temp_threshold = 26
        elif season == 'winter':
            temp_threshold = 22
        else:
            temp_threshold = 24
        
        if temperature > temp_threshold:
            relay_on = True
        
        # Water leak override
        if water_leak:
            relay_on = False  # Safety override
        
        # Vibration consideration
        if vibration > 800:
            relay_on = False  # Avoid running during high vibration
        
        # Energy optimization
        if off_peak and temperature > temp_threshold - 2:
            relay_on = True  # More liberal during off-peak
        
        # Add some randomness for realistic noise
        if np.random.random() < 0.05:
            relay_on = not relay_on
        
        sample = {
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "vibration": int(vibration),
            "water_leak": water_leak,
            "off_peak": off_peak,
            "alert_active": alert_active,
            "relay_on": relay_on,
            "timestamp": time.time() + i,
            "season": season,
            "hour": hour
        }
        
        data.append(sample)
    
    # Save synthetic data
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Synthetic data saved to {save_path}")
    
    return data

def load_training_data(data_path="synthetic_training_data.json") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare training data
    """
    if not Path(data_path).exists():
        logger.info("Training data not found, generating synthetic data...")
        generate_synthetic_data(save_path=data_path)
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} training samples")
    
    # Extract features and labels
    features = []
    labels = []
    
    for sample in data:
        feature_vector = [
            sample["temperature"],
            sample["humidity"],
            sample["vibration"],
            float(sample["water_leak"]),
            float(sample["off_peak"]),
            float(sample["alert_active"])
        ]
        features.append(feature_vector)
        labels.append(float(sample["relay_on"]))
    
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)

def train_model(features: np.ndarray, labels: np.ndarray, 
                model_save_path="relay_model.pth", 
                scaler_save_path="scaler.pkl") -> Tuple[EmbeddedRelayMLP, StandardScaler]:
    """
    Train the neural network model
    """
    logger.info("Starting model training...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = EmbeddedRelayMLP(input_size=6)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    num_epochs = 100
    train_losses = []
    
    logger.info(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        test_outputs = model(X_test_tensor)
        
        train_predictions = (train_outputs > 0.5).float().squeeze().numpy()
        test_predictions = (test_outputs > 0.5).float().squeeze().numpy()
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, test_predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, test_predictions))
    
    # Save model and scaler
    torch.save(model.state_dict(), model_save_path)
    joblib.dump(scaler, scaler_save_path)
    
    logger.info(f"Model saved to {model_save_path}")
    logger.info(f"Scaler saved to {scaler_save_path}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    logger.info("Training loss plot saved as training_loss.png")
    
    return model, scaler

def test_model_inference(model_path="relay_model.pth", scaler_path="scaler.pkl"):
    """
    Test the trained model with sample inputs
    """
    logger.info("Testing model inference...")
    
    # Load model and scaler
    model = EmbeddedRelayMLP()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    scaler = joblib.load(scaler_path)
    
    # Test cases
    test_cases = [
        {
            "name": "Hot summer day",
            "features": [32.0, 65.0, 450, 0, 0, 0],  # temp, humidity, vibration, water_leak, off_peak, alert
            "expected": True
        },
        {
            "name": "Cool winter evening",
            "features": [20.0, 55.0, 400, 0, 1, 0],
            "expected": False
        },
        {
            "name": "Water leak emergency",
            "features": [25.0, 60.0, 500, 1, 0, 1],
            "expected": False
        },
        {
            "name": "High vibration",
            "features": [28.0, 70.0, 850, 0, 0, 1],
            "expected": False
        },
        {
            "name": "Off-peak optimization",
            "features": [26.0, 65.0, 400, 0, 1, 0],
            "expected": True
        }
    ]
    
    print("\n🧪 Model Inference Tests:")
    print("=" * 50)
    
    for test_case in test_cases:
        features = np.array(test_case["features"]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        with torch.no_grad():
            input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
            output = model(input_tensor)
            confidence = output.item()
            prediction = confidence > 0.5
        
        print(f"Test: {test_case['name']}")
        print(f"  Features: {test_case['features']}")
        print(f"  Prediction: {'ON' if prediction else 'OFF'} (confidence: {confidence:.3f})")
        print(f"  Expected: {'ON' if test_case['expected'] else 'OFF'}")
        print(f"  Status: {'✅ PASS' if prediction == test_case['expected'] else '❌ FAIL'}")
        print()

def main():
    """
    Main training pipeline
    """
    print("🤖 AI Model Training Pipeline")
    print("=" * 40)
    
    # Load or generate training data
    features, labels = load_training_data()
    
    # Display data statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {len(features)}")
    print(f"  Positive samples (relay ON): {np.sum(labels):.0f} ({np.mean(labels)*100:.1f}%)")
    print(f"  Negative samples (relay OFF): {len(labels) - np.sum(labels):.0f} ({(1-np.mean(labels))*100:.1f}%)")
    print(f"  Feature dimensions: {features.shape[1]}")
    
    feature_names = ["temperature", "humidity", "vibration", "water_leak", "off_peak", "alert_active"]
    print(f"\n📈 Feature Statistics:")
    for i, name in enumerate(feature_names):
        values = features[:, i]
        print(f"  {name}: mean={np.mean(values):.2f}, std={np.std(values):.2f}, range=[{np.min(values):.2f}, {np.max(values):.2f}]")
    
    # Train model
    model, scaler = train_model(features, labels)
    
    # Test inference
    test_model_inference()
    
    print("\n🎉 Training completed successfully!")
    print("   Model files: relay_model.pth, scaler.pkl")
    print("   Ready for embedded conversion with embedded_ai_converter.py")

if __name__ == "__main__":
    main()