#!/usr/bin/env python3
"""
ESP32 Deployment Script
Prepares and uploads IoT code to ESP32 microcontroller
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
import time
import json

def find_esp32_port():
    """Find ESP32 USB port automatically"""
    import serial.tools.list_ports
    
    esp32_ports = []
    for port in serial.tools.list_ports.comports():
        # Common ESP32 vendor IDs
        if port.vid in [0x10C4, 0x1A86, 0x0403]:  # Silicon Labs, QinHeng, FTDI
            esp32_ports.append(port.device)
    
    return esp32_ports[0] if esp32_ports else None

def check_file_size(file_path):
    """Check if file size is suitable for ESP32"""
    size = os.path.getsize(file_path)
    if size > 100000:  # 100KB warning
        print(f"⚠️  Warning: {file_path} is {size/1024:.1f}KB (might be too large for ESP32)")
    else:
        print(f"✅ {file_path}: {size/1024:.1f}KB (suitable for ESP32)")
    return size

def prepare_esp32_files(output_dir="esp32_deploy"):
    """Prepare minimal file set for ESP32 deployment"""
    
    print("🔧 Preparing ESP32 deployment files...")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Core files needed on ESP32
    core_files = {
        "iot/embedded_ai_model.py": "ai_model.py",  # Rename for simplicity
        "iot/sensors.py": "sensors.py",
        "iot/actuators.py": "actuators.py", 
        "iot/energy.py": "energy.py",
        "iot/config_manager.py": "config_manager.py",
        "iot/config.json": "config.json",
    }
    
    # Create main controller file for ESP32
    esp32_main = """'''
ESP32 Main Controller
Minimal IoT controller for ESP32 with embedded AI
'''

import time
import gc
from ai_model import ai_predict
from sensors import read_temperature, read_humidity, read_vibration, detect_water_leak
# Network utilities
    print("🔒 ESP32 Controller Stopped")

if __name__ == "__main__":
    main()
"""

    # Copy and prepare files
    total_size = 0
    
    for src_file, dst_name in core_files.items():
        src_path = Path(src_file)
        dst_path = Path(output_dir) / dst_name
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            size = check_file_size(dst_path)
            total_size += size
        else:
            print(f"⚠️  Warning: {src_file} not found")
    
    # Create ESP32 main file
    main_path = Path(output_dir) / "main.py"
    with open(main_path, 'w') as f:
        f.write(esp32_main)
    
    size = check_file_size(main_path)
    total_size += size
    
    # Create boot.py for ESP32
    boot_py = """# ESP32 Boot Configuration
import network
import time

# Enable garbage collection
import gc
gc.enable()

# Connect to WiFi (optional)
def connect_wifi(ssid="your_wifi", password="your_password"):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print("Connecting to WiFi...")
        wlan.connect(ssid, password)
        timeout = 10
        while not wlan.isconnected() and timeout > 0:
            time.sleep(1)
            timeout -= 1
        
        if wlan.isconnected():
            print(f"WiFi connected: {wlan.ifconfig()}")
        else:
            print("WiFi connection failed")

# Uncomment to enable WiFi
# connect_wifi()

print("ESP32 Boot Complete")
"""
    
    boot_path = Path(output_dir) / "boot.py"
    with open(boot_path, 'w') as f:
        f.write(boot_py)
    
    total_size += check_file_size(boot_path)
    
    print(f"\n📦 ESP32 deployment prepared in '{output_dir}'")
    print(f"   Total size: {total_size/1024:.1f}KB")
    print(f"   ESP32 flash available: ~3.8MB")
    print(f"   Usage: {(total_size/1024)/3800*100:.1f}% of available space")
    
    return output_dir

def upload_to_esp32(deployment_dir, port=None):
    """Upload files to ESP32 using ampy or mpremote"""
    
    if port is None:
        port = find_esp32_port()
        if port is None:
            print("❌ No ESP32 found. Please connect ESP32 and try again.")
            return False
    
    print(f"📤 Uploading to ESP32 on {port}...")
    
    # Try mpremote first (newer tool)
    try:
        for file_path in Path(deployment_dir).glob("*.py"):
            cmd = f"mpremote connect {port} cp {file_path} :{file_path.name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Uploaded {file_path.name}")
            else:
                print(f"❌ Failed to upload {file_path.name}: {result.stderr}")
        
        # Upload config.json
        config_path = Path(deployment_dir) / "config.json"
        if config_path.exists():
            cmd = f"mpremote connect {port} cp {config_path} :config.json"
            subprocess.run(cmd, shell=True)
            print(f"✅ Uploaded config.json")
        
        return True
        
    except FileNotFoundError:
        print("⚠️  mpremote not found, trying ampy...")
        
        # Fallback to ampy
        try:
            for file_path in Path(deployment_dir).glob("*.py"):
                cmd = f"ampy --port {port} put {file_path} {file_path.name}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ Uploaded {file_path.name}")
                else:
                    print(f"❌ Failed to upload {file_path.name}")
            
            return True
            
        except FileNotFoundError:
            print("❌ Neither mpremote nor ampy found.")
            print("   Install with: pip install mpremote")
            print("   Or manually copy files to ESP32")
            return False

def monitor_esp32(port=None):
    """Monitor ESP32 serial output"""
    
    if port is None:
        port = find_esp32_port()
        if port is None:
            print("❌ No ESP32 found")
            return
    
    print(f"👁️  Monitoring ESP32 on {port} (Ctrl+C to stop)...")
    
    try:
        cmd = f"mpremote connect {port} repl"
        subprocess.run(cmd, shell=True)
    except FileNotFoundError:
        print("mpremote not found. Using alternative method...")
        try:
            import serial
            ser = serial.Serial(port, 115200, timeout=1)
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(line)
        except ImportError:
            print("❌ pyserial not installed. Install with: pip install pyserial")
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")

def main():
    parser = argparse.ArgumentParser(description='ESP32 IoT Deployment Tool')
    parser.add_argument('action', choices=['prepare', 'upload', 'deploy', 'monitor'], 
                        help='Action to perform')
    parser.add_argument('--port', help='ESP32 serial port (auto-detected if not specified)')
    parser.add_argument('--output-dir', default='esp32_deploy', help='Output directory for prepared files')
    
    args = parser.parse_args()
    
    if args.action == 'prepare':
        prepare_esp32_files(args.output_dir)
        
    elif args.action == 'upload':
        upload_to_esp32(args.output_dir, args.port)
        
    elif args.action == 'deploy':
        # Full deployment: prepare + upload
        deployment_dir = prepare_esp32_files(args.output_dir)
        upload_to_esp32(deployment_dir, args.port)
        print("\n🎉 Deployment complete! ESP32 should start automatically.")
        print("   Use './deploy_esp32.py monitor' to see output")
        
    elif args.action == 'monitor':
        monitor_esp32(args.port)

if __name__ == "__main__":
    main()
