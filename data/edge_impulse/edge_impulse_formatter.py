#!/usr/bin/env python3
"""
Edge Impulse Data Formatter for ESP32 IoT Project
Converts synthetic data to Edge Impulse Studio compatible format
"""

import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_edge_impulse_sample() -> Dict[str, Any]:
    """Generate a single sample in Edge Impulse format"""
    
    # Generate realistic sensor data
    temperature = random.uniform(15, 35)
    humidity = random.uniform(30, 90)
    vibration = random.randint(0, 1000)
    water_leak = random.choice([0, 1]) if random.random() < 0.05 else 0
    off_peak = 1 if random.randint(0, 23) in [22, 23, 0, 1, 2, 3, 4, 5] else 0
    alert_active = random.choice([0, 1]) if random.random() < 0.1 else 0
    
    # Decision logic (simplified)
    relay_on = 1 if (temperature > 28 or (off_peak and temperature > 25)) and not water_leak else 0
    
    # Create Edge Impulse format
    return {
        "protected": {
            "ver": "v1", 
            "alg": "HS256",
            "iat": int(time.time())
        },
        "signature": "",
        "payload": {
            "device_name": f"ESP32_IoT_{random.randint(1000, 9999)}",
            "device_type": "ESP32",
            "interval_ms": 1000,
            "sensors": [
                {"name": "temperature", "units": "degC"},
                {"name": "humidity", "units": "percent"},
                {"name": "vibration", "units": "raw"},
                {"name": "water_leak", "units": "boolean"},
                {"name": "off_peak", "units": "boolean"},  
                {"name": "alert_active", "units": "boolean"}
            ],
            "values": [[
                temperature,
                humidity,
                vibration,
                water_leak,
                off_peak,
                alert_active
            ]],
            "label": "relay_on" if relay_on else "relay_off"
        }
    }

def create_edge_impulse_dataset(num_samples: int = 100) -> None:
    """Create multiple JSON files for Edge Impulse upload"""
    
    print(f"🚀 Creating {num_samples} Edge Impulse samples...")
    
    # Create output directory
    output_dir = Path("edge_impulse_samples")
    output_dir.mkdir(exist_ok=True)
    
    samples_created = 0
    relay_on_count = 0
    relay_off_count = 0
    
    for i in range(num_samples):
        sample = generate_edge_impulse_sample()
        
        # Track labels
        if sample["payload"]["label"] == "relay_on":
            relay_on_count += 1
        else:
            relay_off_count += 1
        
        # Save individual file (Edge Impulse prefers individual files)
        filename = f"sample_{i:04d}_{sample['payload']['label']}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(sample, f, indent=2)
        
        samples_created += 1
        
        if (i + 1) % 50 == 0:
            print(f"   Generated {i + 1}/{num_samples} samples...")
    
    print(f"\n✅ Edge Impulse samples created:")
    print(f"   Output directory: {output_dir}")
    print(f"   Total samples: {samples_created}")
    print(f"   Relay ON: {relay_on_count}")
    print(f"   Relay OFF: {relay_off_count}")
    print(f"\n📖 Upload Instructions:")
    print(f"1. Go to https://studio.edgeimpulse.com/")
    print(f"2. Create new project")
    print(f"3. Go to 'Data acquisition' → 'Upload data'")
    print(f"4. Select multiple files from: {output_dir}")
    print(f"5. Choose 'Infer from filename' for labels")

def create_single_json_for_upload(num_samples: int = 100) -> None:
    """Create a single JSON file with all samples (alternative format)"""
    
    print(f"📦 Creating single file with {num_samples} samples...")
    
    all_samples = []
    relay_on_count = 0
    relay_off_count = 0
    
    for i in range(num_samples):
        sample = generate_edge_impulse_sample()
        
        # Flatten for single file format
        flattened = {
            "deviceName": sample["payload"]["device_name"],
            "deviceType": sample["payload"]["device_type"],
            "intervalMs": sample["payload"]["interval_ms"],
            "sensors": sample["payload"]["sensors"],
            "values": sample["payload"]["values"][0],  # Flatten single reading
            "label": sample["payload"]["label"],
            "timestamp": int(time.time() * 1000) + i * 1000  # Increment timestamp
        }
        
        all_samples.append(flattened)
        
        if sample["payload"]["label"] == "relay_on":
            relay_on_count += 1
        else:
            relay_off_count += 1
    
    # Create dataset container
    dataset = {
        "version": 1,
        "projectName": "ESP32 IoT Relay Control",
        "sampleRateHz": 1,
        "totalSamples": len(all_samples),
        "samples": all_samples
    }
    
    # Save to file
    output_file = Path("edge_impulse_upload.json")
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Single upload file created: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
    print(f"   Relay ON: {relay_on_count}")
    print(f"   Relay OFF: {relay_off_count}")

def main():
    """Main function"""
    print("🎯 Edge Impulse Data Formatter")
    print("=" * 50)
    
    # Create both formats
    print("\n📁 Creating individual files (recommended)...")
    create_edge_impulse_dataset(200)
    
    print("\n📦 Creating single upload file (alternative)...")
    create_single_json_for_upload(200)
    
    print("\n🎉 Ready for Edge Impulse upload!")

if __name__ == "__main__":
    main()