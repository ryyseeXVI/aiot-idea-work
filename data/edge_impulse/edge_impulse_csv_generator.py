#!/usr/bin/env python3
"""
Edge Impulse CSV Data Generator for ESP32 IoT Project
Creates CSV files that Edge Impulse can import easily
"""

import csv
import random
import time
from pathlib import Path
from datetime import datetime

def generate_csv_data(num_samples: int = 500, output_file: str = "iot_relay_data.csv"):
    """Generate CSV data for Edge Impulse"""
    
    print(f"📊 Generating {num_samples} samples as CSV...")
    
    headers = [
        "timestamp",
        "temperature", 
        "humidity",
        "vibration", 
        "water_leak",
        "off_peak",
        "alert_active",
        "label"
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
            water_leak = 1 if random.random() < 0.03 else 0  # 3% chance
            off_peak = 1 if random.randint(0, 23) in [22, 23, 0, 1, 2, 3, 4, 5] else 0
            alert_active = 1 if random.random() < 0.08 else 0  # 8% chance
            
            # Decision logic
            if water_leak:
                relay_on = 0  # Safety: turn off if water detected
            elif temperature > 30:
                relay_on = 1  # Hot: turn on
            elif temperature > 25 and off_peak:
                relay_on = 1  # Warm + off-peak: turn on
            elif temperature < 20:
                relay_on = 0  # Cold: turn off
            else:
                relay_on = random.choice([0, 1])  # Random for edge cases
            
            # Create numeric label (0 = off, 1 = on)
            label = 1 if relay_on else 0
            
            if relay_on:
                relay_on_count += 1
            else:
                relay_off_count += 1
            
            # Write row
            row = [
                base_time + (i * 1000),  # timestamp in ms
                temperature,
                humidity,
                vibration,
                water_leak,
                off_peak,
                alert_active,
                label
            ]
            writer.writerow(row)
        
        print(f"✅ CSV data saved to: {output_file}")
        print(f"   Total samples: {num_samples}")
        print(f"   Relay ON: {relay_on_count}")
        print(f"   Relay OFF: {relay_off_count}")
        print(f"   File size: {Path(output_file).stat().st_size / 1024:.1f} KB")

def create_multiple_csv_files():
    """Create separate CSV files for training and testing"""
    
    print("📁 Creating training and test datasets...")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Training data (80%)
    generate_csv_data(800, str(script_dir / "iot_training_data.csv"))
    
    # Test data (20%)  
    generate_csv_data(200, str(script_dir / "iot_test_data.csv"))
    
    print("\n🎯 Edge Impulse CSV Upload Instructions:")
    print("1. Go to https://studio.edgeimpulse.com/")
    print("2. Create new project: 'ESP32 IoT Relay Control'")
    print("3. Go to 'Data acquisition' → 'Upload data'")
    print("4. Upload 'iot_training_data.csv' → Set as 'Training'")
    print("5. Upload 'iot_test_data.csv' → Set as 'Testing'")
    print("6. Set 'label' column as the target")
    print("7. Configure other columns as features")

def main():
    """Main function"""
    print("🎯 Edge Impulse CSV Generator")
    print("=" * 40)
    
    create_multiple_csv_files()
    
    print("\n✅ CSV files ready for Edge Impulse!")

if __name__ == "__main__":
    main()