#!/usr/bin/env python3
"""
Edge Impulse Data Generator
Converts IoT sensor data to Edge Impulse JSON format for machine learning
"""

import json
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

def generate_fallback_data(num_samples: int) -> List[Dict]:
    """
    Fallback data generation if main model import fails
    """
    np.random.seed(42)
    data = []
    
    for i in range(num_samples):
        # Simple random data generation
        temperature = np.random.uniform(15, 35)
        humidity = np.random.uniform(30, 80)
        vibration = np.random.randint(300, 900)
        water_leak = np.random.random() < 0.02
        off_peak = np.random.choice([True, False])
        alert_active = np.random.choice([True, False])
        
        # Simple relay logic
        relay_on = temperature > 25 and not water_leak
        
        sample = {
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "vibration": int(vibration),
            "water_leak": water_leak,
            "off_peak": off_peak,
            "alert_active": alert_active,
            "relay_on": relay_on,
            "timestamp": time.time() + i,
            "season": np.random.choice(['winter', 'spring', 'summer', 'fall']),
            "hour": np.random.randint(0, 24)
        }
        data.append(sample)
    
    return data

def generate_edge_impulse_sample(sample_data: Dict, sample_id: int, label: Optional[str] = None) -> Dict:
    """
    Convert a single sample to Edge Impulse format
    
    Edge Impulse expects:
    {
        "device_name": "string",
        "device_type": "string", 
        "interval_ms": number,
        "sensors": [{"name": "string", "units": "string"}],
        "values": [[sensor1_val, sensor2_val, ...], ...]
    }
    """
    
    # Convert our sensor data to Edge Impulse format
    sensors = [
        {"name": "temperature", "units": "°C"},
        {"name": "humidity", "units": "%"},
        {"name": "vibration", "units": "raw"},
        {"name": "water_leak", "units": "boolean"},
        {"name": "off_peak", "units": "boolean"},
        {"name": "alert_active", "units": "boolean"}
    ]
    
    # Extract sensor values in the correct order
    values = [
        sample_data["temperature"],
        sample_data["humidity"], 
        sample_data["vibration"],
        1.0 if sample_data["water_leak"] else 0.0,
        1.0 if sample_data["off_peak"] else 0.0,
        1.0 if sample_data["alert_active"] else 0.0
    ]
    
    # Create Edge Impulse sample format
    edge_sample = {
        "device_name": f"iot_device_{sample_id:04d}",
        "device_type": "ESP32_IoT_Controller",
        "interval_ms": 1000,  # 1 second intervals
        "sensors": sensors,
        "values": [values],  # Single time point
        "hmac_key": "",
        "protected": False
    }
    
    # Add label if provided (for classification)
    if label is not None:
        edge_sample["label"] = label
    
    # Add metadata
    edge_sample["metadata"] = {
        "season": sample_data.get("season", "unknown"),
        "hour": sample_data.get("hour", 0),
        "relay_decision": sample_data.get("relay_on", False),
        "timestamp": sample_data.get("timestamp", time.time())
    }
    
    return edge_sample

def generate_edge_impulse_dataset(num_samples: int = 1000, output_file: str = "edge_impulse_dataset.json") -> Dict:
    """
    Generate a complete dataset in Edge Impulse format
    """
    print(f"🧠 Generating {num_samples} samples for Edge Impulse...")
    
    # Import our existing data generation logic
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
    
    # Generate synthetic data using our existing function
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../model'))
        from main import generate_synthetic_data
    except ImportError:
        # Fallback if import fails
        def generate_synthetic_data(num_samples, save_path=None):
            return generate_fallback_data(num_samples)
    
    raw_data = generate_synthetic_data(num_samples, save_path=None)
    
    # Convert to Edge Impulse format
    edge_samples = []
    
    for i, sample in enumerate(raw_data):
        # Create label based on relay decision
        label = "relay_on" if sample["relay_on"] else "relay_off"
        
        edge_sample = generate_edge_impulse_sample(sample, i, label)
        edge_samples.append(edge_sample)
    
    # Create the complete dataset structure
    dataset = {
        "version": 1,
        "has_data": True,
        "project_id": "iot_relay_control",
        "project_name": "ESP32 IoT Relay Control",
        "project_description": "Edge AI for intelligent IoT relay control with environmental sensors",
        "created_at": datetime.now().isoformat(),
        "sample_rate_hz": 1,  # 1 Hz sampling
        "total_samples": len(edge_samples),
        "sensors": [
            {"name": "temperature", "units": "°C", "type": "float"},
            {"name": "humidity", "units": "%", "type": "float"},
            {"name": "vibration", "units": "raw", "type": "integer"},
            {"name": "water_leak", "units": "boolean", "type": "boolean"},
            {"name": "off_peak", "units": "boolean", "type": "boolean"},
            {"name": "alert_active", "units": "boolean", "type": "boolean"}
        ],
        "labels": ["relay_on", "relay_off"],
        "data": edge_samples
    }
    
    # Save to file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Edge Impulse dataset saved to: {output_path}")
    print(f"   Total samples: {len(edge_samples)}")
    print(f"   Relay ON samples: {sum(1 for s in edge_samples if s['label'] == 'relay_on')}")
    print(f"   Relay OFF samples: {sum(1 for s in edge_samples if s['label'] == 'relay_off')}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return dataset

def generate_time_series_data(duration_minutes: int = 60, output_file: str = "edge_impulse_timeseries.json") -> Dict:
    """
    Generate time-series data with multiple readings per sample (better for Edge Impulse)
    """
    print(f"📊 Generating {duration_minutes} minutes of time-series data...")
    
    # Import our existing data generation logic
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
    
    # Generate continuous data
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../model'))
        from main import generate_synthetic_data
    except ImportError:
        generate_synthetic_data = generate_fallback_data
    
    # Generate continuous data
    samples_per_minute = 60  # 1 second intervals
    total_samples = duration_minutes * samples_per_minute
    raw_data = generate_synthetic_data(total_samples)
    
    # Group into 10-second windows (10 readings per window)
    window_size = 10
    windows = []
    
    for i in range(0, len(raw_data) - window_size, window_size):
        window_data = raw_data[i:i + window_size]
        
        # Extract sensor values for this window
        window_values = []
        for sample in window_data:
            values = [
                sample["temperature"],
                sample["humidity"],
                sample["vibration"],
                1.0 if sample["water_leak"] else 0.0,
                1.0 if sample["off_peak"] else 0.0,
                1.0 if sample["alert_active"] else 0.0
            ]
            window_values.append(values)
        
        # Determine label (majority vote for relay decision in window)
        relay_decisions = [sample["relay_on"] for sample in window_data]
        label = "relay_on" if sum(relay_decisions) > len(relay_decisions) / 2 else "relay_off"
        
        # Create Edge Impulse sample
        edge_sample = {
            "device_name": f"iot_timeseries_{len(windows):04d}",
            "device_type": "ESP32_IoT_Controller_TimeSeries",
            "interval_ms": 1000,
            "sensors": [
                {"name": "temperature", "units": "°C"},
                {"name": "humidity", "units": "%"},
                {"name": "vibration", "units": "raw"},
                {"name": "water_leak", "units": "boolean"},
                {"name": "off_peak", "units": "boolean"},
                {"name": "alert_active", "units": "boolean"}
            ],
            "values": window_values,  # Multiple time points
            "label": label,
            "hmac_key": "",
            "protected": False,
            "metadata": {
                "window_start": window_data[0]["timestamp"],
                "window_end": window_data[-1]["timestamp"],
                "window_size_seconds": window_size,
                "season": window_data[0].get("season", "unknown")
            }
        }
        
        windows.append(edge_sample)
    
    # Create complete dataset
    dataset = {
        "version": 1,
        "has_data": True,
        "project_id": "iot_relay_timeseries",
        "project_name": "ESP32 IoT Time-Series Control",
        "project_description": "Time-series edge AI for IoT relay control with environmental pattern recognition",
        "created_at": datetime.now().isoformat(),
        "sample_rate_hz": 1,
        "window_size_ms": window_size * 1000,
        "total_samples": len(windows),
        "sensors": [
            {"name": "temperature", "units": "°C", "type": "float"},
            {"name": "humidity", "units": "%", "type": "float"},
            {"name": "vibration", "units": "raw", "type": "integer"},
            {"name": "water_leak", "units": "boolean", "type": "boolean"},
            {"name": "off_peak", "units": "boolean", "type": "boolean"},
            {"name": "alert_active", "units": "boolean", "type": "boolean"}
        ],
        "labels": ["relay_on", "relay_off"],
        "data": windows
    }
    
    # Save to file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Time-series dataset saved to: {output_path}")
    print(f"   Total windows: {len(windows)}")
    print(f"   Window size: {window_size} seconds")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return dataset

def generate_csv_format(num_samples: int = 1000, output_file: str = "edge_impulse_data.csv") -> str:
    """
    Generate CSV format (alternative to JSON for Edge Impulse)
    """
    print(f"📋 Generating CSV format with {num_samples} samples...")
    
    # Import our existing data generation logic
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../model'))
    
    try:
        from main import generate_synthetic_data
        raw_data = generate_synthetic_data(num_samples)
    except ImportError:
        raw_data = generate_fallback_data(num_samples)
    
    # Create CSV content
    csv_lines = []
    
    # Header
    csv_lines.append("timestamp,temperature,humidity,vibration,water_leak,off_peak,alert_active,label")
    
    # Data rows
    for sample in raw_data:
        label = "relay_on" if sample["relay_on"] else "relay_off"
        line = f"{sample['timestamp']},{sample['temperature']},{sample['humidity']},{sample['vibration']},{int(sample['water_leak'])},{int(sample['off_peak'])},{int(sample['alert_active'])},{label}"
        csv_lines.append(line)
    
    # Save CSV
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        f.write('\\n'.join(csv_lines))
    
    print(f"✅ CSV dataset saved to: {output_path}")
    print(f"   Total rows: {len(csv_lines) - 1}")  # Minus header
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate Edge Impulse compatible data')
    parser.add_argument('--format', choices=['json', 'timeseries', 'csv'], default='json',
                        help='Output format (default: json)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples to generate (default: 1000)')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration in minutes for time-series data (default: 60)')
    parser.add_argument('--output', help='Output filename (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    print("🚀 Edge Impulse Data Generator for ESP32 IoT Project")
    print("="*60)
    
    output_file = ""  # Initialize to avoid unbound variable
    
    if args.format == 'json':
        output_file = args.output or "edge_impulse_dataset.json"
        generate_edge_impulse_dataset(args.samples, output_file)
        
    elif args.format == 'timeseries':
        output_file = args.output or "edge_impulse_timeseries.json"
        generate_time_series_data(args.duration, output_file)
        
    elif args.format == 'csv':
        output_file = args.output or "edge_impulse_data.csv"
        generate_csv_format(args.samples, output_file)
    
    print("\\n📖 Usage with Edge Impulse:")
    print("1. Go to https://studio.edgeimpulse.com/")
    print("2. Create a new project")
    print("3. Go to 'Data acquisition' > 'Upload data'")
    print(f"4. Upload your generated file: {output_file}")
    print("5. Configure as 'Time series' data")
    print("6. Set labels as classification target")
    print("\\n🎯 Your IoT relay control model is ready for Edge Impulse training!")

if __name__ == "__main__":
    main()
