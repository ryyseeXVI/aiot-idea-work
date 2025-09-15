#!/usr/bin/env python3
"""
Edge Impulse Compatible Data Generator for ESP32 IoT
Creates CSV files with proper numeric labels and Edge Impulse format
"""

import csv
import random
import time
from pathlib import Path
from datetime import datetime

def generate_edge_impulse_csv(num_samples: int = 500, output_file: str = "edge_impulse_data.csv"):
    """Generate Edge Impulse compatible CSV data"""
    
    print(f"📊 Generating {num_samples} Edge Impulse samples...")
    
    # Headers for Edge Impulse (timestamp + features + label)
    headers = [
        "timestamp",
        "temperature", 
        "humidity",
        "vibration", 
        "water_leak",
        "off_peak",
        "alert_active",
        "label"  # 0=relay_off, 1=relay_on
    ]
    
    relay_on_count = 0
    relay_off_count = 0
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        base_time = int(time.time() * 1000)  # milliseconds
        
        for i in range(num_samples):
            # Generate realistic sensor data
            temperature = round(random.uniform(15, 35), 1)
            humidity = round(random.uniform(30, 90), 1)
            vibration = random.randint(0, 1000)
            water_leak = 1 if random.random() < 0.02 else 0  # 2% chance
            off_peak = 1 if random.randint(0, 23) in [22, 23, 0, 1, 2, 3, 4, 5] else 0
            alert_active = 1 if random.random() < 0.05 else 0  # 5% chance
            
            # Decision logic (same as before)
            if water_leak:
                relay_on = 0  # Safety: turn off if water detected
            elif temperature > 30:
                relay_on = 1  # Hot: turn on
            elif temperature > 25 and off_peak:
                relay_on = 1  # Warm + off-peak: turn on
            elif temperature < 20:
                relay_on = 0  # Cool: turn off
            else:
                relay_on = random.choice([0, 1])  # Random for edge cases
            
            # Count labels
            if relay_on:
                relay_on_count += 1
            else:
                relay_off_count += 1
            
            # Write row (timestamp in ms, features, numeric label)
            row = [
                base_time + (i * 1000),  # timestamp in ms
                temperature,
                humidity,
                vibration,
                water_leak,
                off_peak,
                alert_active,
                relay_on  # 0 or 1 (numeric label)
            ]
            writer.writerow(row)
        
        print(f"✅ Edge Impulse CSV saved to: {output_file}")
        print(f"   Total samples: {num_samples}")
        print(f"   Relay ON (1): {relay_on_count}")
        print(f"   Relay OFF (0): {relay_off_count}")
        print(f"   File size: {Path(output_file).stat().st_size / 1024:.1f} KB")

def create_edge_impulse_datasets():
    """Create optimized datasets for Edge Impulse"""
    
    print("🎯 Edge Impulse Dataset Generator")
    print("=" * 40)
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Training data (larger dataset)
    training_file = script_dir / "edge_impulse_training.csv"
    generate_edge_impulse_csv(1000, str(training_file))
    
    print()
    
    # Test data (smaller dataset)  
    test_file = script_dir / "edge_impulse_testing.csv"
    generate_edge_impulse_csv(200, str(test_file))
    
    print(f"\n🎯 Edge Impulse Upload Instructions:")
    print(f"1. Go to https://studio.edgeimpulse.com/")
    print(f"2. Create new project: 'ESP32 IoT Relay Control'")
    print(f"3. Go to 'Data acquisition' → 'Upload data'")
    print(f"4. Upload '{training_file.name}' → Set as 'Training'")
    print(f"5. Upload '{test_file.name}' → Set as 'Testing'")
    print(f"6. Configure:")
    print(f"   - Label column: 'label' (0=OFF, 1=ON)")
    print(f"   - Features: temperature, humidity, vibration, water_leak, off_peak, alert_active")
    print(f"   - Sample rate: 1 Hz")
    print(f"   - Project type: Classification")

def main():
    """Main function"""
    create_edge_impulse_datasets()
    print("\n✅ Edge Impulse datasets ready for upload!")

if __name__ == "__main__":
    main()