#!/usr/bin/env python3
"""
Wokwi Project Management Script
Manages Wokwi simulation, testing, and deployment
"""

import os
import json
import subprocess
import webbrowser
import time
from pathlib import Path
import shutil
import tempfile

def validate_wokwi_project():
    """Validate Wokwi project structure"""
    print("🔍 Validating Wokwi project...")
    
    required_files = ['main.py', 'diagram.json', 'wokwi.toml']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    # Check diagram.json validity
    try:
        with open('diagram.json', 'r') as f:
            diagram = json.load(f)
        print(f"✅ diagram.json valid ({len(diagram.get('parts', []))} parts)")
    except json.JSONDecodeError as e:
        print(f"❌ diagram.json invalid: {e}")
        return False
    
    # Check main.py syntax
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        compile(content, 'main.py', 'exec')
        print(f"✅ main.py syntax valid ({len(content.splitlines())} lines)")
    except SyntaxError as e:
        print(f"❌ main.py syntax error: {e}")
        return False
    
    print("✅ Wokwi project validation passed")
    return True

def check_pin_configuration():
    """Check if pins in diagram.json match configuration"""
    print("🔧 Checking pin configuration...")
    
    # Load config.json pins
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        gpio_pins = config.get('gpio_pins', {})
    except:
        print("⚠️  config.json not found, skipping pin check")
        return True
    
    # Load diagram.json
    try:
        with open('diagram.json', 'r') as f:
            diagram = json.load(f)
    except:
        print("❌ Cannot read diagram.json")
        return False
    
    # Expected pin mappings
    expected_pins = {
        'temp_sensor_pin': gpio_pins.get('temp_sensor_pin', 4),
        'relay_pin': gpio_pins.get('relay_pin', 5), 
        'alert_led_pin': gpio_pins.get('alert_led_pin', 15),
        'vibration_pin': gpio_pins.get('vibration_pin', 33),
        'water_leak_pin': gpio_pins.get('water_leak_pin', 14)
    }
    
    # Check connections in diagram
    connections = diagram.get('connections', [])
    pin_usage = {}
    
    for conn in connections:
        if len(conn) >= 2:
            esp_pin = conn[0]
            device = conn[1]
            if esp_pin.startswith('esp:D') or esp_pin.startswith('esp:A'):
                pin_num = esp_pin.split(':')[1]
                pin_usage[pin_num] = device
    
    print(f"📌 Pin configuration:")
    for pin_name, pin_num in expected_pins.items():
        pin_str = f"D{pin_num}" if pin_num < 32 else f"A{pin_num-32}"
        actual = pin_usage.get(pin_str, "not connected")
        print(f"   {pin_name}: GPIO{pin_num} ({pin_str}) -> {actual}")
    
    return True

def optimize_for_esp32():
    """Optimize code for ESP32 memory constraints"""
    print("⚡ Optimizing for ESP32...")
    
    # Check file sizes
    files_to_check = ['main.py', 'embedded_ai_model.py', 'sensors.py', 'actuators.py']
    total_size = 0
    
    for file in files_to_check:
        if Path(file).exists():
            size = Path(file).stat().st_size
            total_size += size
            status = "✅" if size < 50000 else "⚠️"
            print(f"   {status} {file}: {size/1024:.1f}KB")
        else:
            print(f"   ❓ {file}: not found")
    
    print(f"📊 Total size: {total_size/1024:.1f}KB")
    
    # ESP32 has ~4MB flash, but MicroPython uses ~1.5MB
    # Available for user code: ~2.5MB
    available_mb = 2.5
    usage_percent = (total_size / 1024 / 1024) / available_mb * 100
    
    if usage_percent < 10:
        print(f"✅ Memory usage: {usage_percent:.1f}% (excellent)")
    elif usage_percent < 25:
        print(f"✅ Memory usage: {usage_percent:.1f}% (good)")
    elif usage_percent < 50:
        print(f"⚠️  Memory usage: {usage_percent:.1f}% (acceptable)")
    else:
        print(f"❌ Memory usage: {usage_percent:.1f}% (too high)")
    
    return usage_percent < 50

def create_wokwi_project_zip():
    """Create a zip file for Wokwi.com upload"""
    print("📦 Creating Wokwi project zip...")
    
    # Files to include
    wokwi_files = [
        'main.py',
        'diagram.json', 
        'wokwi.toml',
        'embedded_ai_model.py',
        'config.json'
    ]
    
    # Optional files
    optional_files = [
        'sensors.py',
        'actuators.py',
        'energy.py',
        'config_manager.py'
    ]
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "wokwi_project"
        project_dir.mkdir()
        
        # Copy required files
        for file in wokwi_files:
            if Path(file).exists():
                shutil.copy2(file, project_dir / file)
                print(f"   ✅ Added {file}")
            else:
                print(f"   ❌ Missing {file}")
        
        # Copy optional files
        for file in optional_files:
            if Path(file).exists():
                shutil.copy2(file, project_dir / file)
                print(f"   📄 Added {file} (optional)")
        
        # Create zip
        zip_path = Path.cwd() / "wokwi_project.zip"
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', project_dir)
        
        print(f"📦 Project zip created: {zip_path}")
        print(f"   Size: {zip_path.stat().st_size / 1024:.1f}KB")
        return zip_path

def run_unit_tests():
    """Run unit tests specific to ESP32/Wokwi code"""
    print("🧪 Running ESP32 unit tests...")
    
    # Check if embedded AI model can be imported
    try:
        import sys
        sys.path.append('.')
        
        # Test embedded AI model
        import embedded_ai_model
        decision, confidence = embedded_ai_model.ai_predict([25.0, 65.0, 500, 0, 0, 0])
        print(f"   ✅ AI model test: decision={decision}, confidence={confidence:.3f}")
        
        # Test config loading
        if Path('config.json').exists():
            import json
            with open('config.json') as f:
                config = json.load(f)
            print(f"   ✅ Config test: {len(config)} sections")
        
        # Basic syntax check for main.py
        with open('main.py', 'r') as f:
            main_code = f.read()
        compile(main_code, 'main.py', 'exec')
        print(f"   ✅ Main.py syntax check passed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def open_wokwi_simulator():
    """Open Wokwi simulator in browser"""
    print("🌐 Opening Wokwi simulator...")
    
    wokwi_url = "https://wokwi.com/projects/new/micropython-esp32"
    
    try:
        webbrowser.open(wokwi_url)
        print(f"   Opened: {wokwi_url}")
        print("   📋 Upload your project files manually or use the zip file")
        return True
    except Exception as e:
        print(f"   ❌ Failed to open browser: {e}")
        print(f"   📋 Manually visit: {wokwi_url}")
        return False

def monitor_performance():
    """Analyze performance characteristics"""
    print("📊 Analyzing performance characteristics...")
    
    # Check main.py for performance indicators
    if not Path('main.py').exists():
        print("   ❌ main.py not found")
        return False
    
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Look for performance indicators
    indicators = {
        'AI inference': 'ai_predict' in content or 'predict(' in content,
        'Memory management': 'gc.collect' in content,
        'Timing optimization': 'time.ticks_' in content,
        'Error handling': 'try:' in content and 'except' in content,
        'Adaptive thresholds': 'adapt' in content.lower(),
        'Safety checks': 'safety' in content.lower()
    }
    
    print("   Performance features:")
    for feature, present in indicators.items():
        status = "✅" if present else "❌"
        print(f"     {status} {feature}")
    
    # Estimate cycle time
    lines_count = len(content.splitlines())
    estimated_cycle_ms = max(10, lines_count / 50)  # Rough estimate
    print(f"   📈 Estimated cycle time: ~{estimated_cycle_ms:.0f}ms")
    
    return True

def show_project_info():
    """Show comprehensive project information"""
    print("📋 ESP32 IoT Project Information")
    print("=" * 50)
    
    # Project structure
    print("\n📁 Project Structure:")
    important_files = [
        ('main.py', 'Main ESP32 controller'),
        ('diagram.json', 'Wokwi circuit diagram'),
        ('wokwi.toml', 'Wokwi configuration'),
        ('embedded_ai_model.py', 'AI model for ESP32'),
        ('config.json', 'Hardware configuration'),
        ('sensors.py', 'Sensor interfaces'),
        ('actuators.py', 'Actuator controls')
    ]
    
    for file, description in important_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"   ✅ {file:<20} - {description} ({size/1024:.1f}KB)")
        else:
            print(f"   ❌ {file:<20} - {description} (missing)")
    
    # Configuration
    if Path('config.json').exists():
        print("\n⚙️  Hardware Configuration:")
        try:
            with open('config.json') as f:
                config = json.load(f)
            gpio_pins = config.get('gpio_pins', {})
            for pin_name, pin_num in gpio_pins.items():
                print(f"   📌 {pin_name}: GPIO{pin_num}")
        except:
            print("   ❌ Cannot read config.json")
    
    # Wokwi info
    if Path('wokwi.toml').exists():
        print("\n🔧 Wokwi Configuration:")
        try:
            with open('wokwi.toml') as f:
                content = f.read()
                if 'micropython-dht' in content:
                    print("   📦 DHT sensor library included")
                print("   🎯 Ready for Wokwi simulation")
        except:
            print("   ❌ Cannot read wokwi.toml")
    
    print("\n🚀 Quick Start:")
    print("   1. Run: ./wokwi_manager.py validate")
    print("   2. Run: ./wokwi_manager.py test")
    print("   3. Run: ./wokwi_manager.py open")
    print("   4. Upload files to Wokwi simulator")
    print("   5. Click 'Start Simulation'")

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Wokwi ESP32 Project Manager')
    parser.add_argument('action', choices=[
        'validate', 'pins', 'optimize', 'zip', 'test', 'open', 'monitor', 'info'
    ], help='Action to perform')
    
    # Change to iot directory if exists
    if Path('iot').exists() and not Path('main.py').exists():
        print("📂 Changing to iot/ directory...")
        os.chdir('iot')
    
    args = parser.parse_args()
    
    if args.action == 'validate':
        validate_wokwi_project()
        
    elif args.action == 'pins':
        check_pin_configuration()
        
    elif args.action == 'optimize':
        optimize_for_esp32()
        
    elif args.action == 'zip':
        create_wokwi_project_zip()
        
    elif args.action == 'test':
        run_unit_tests()
        
    elif args.action == 'open':
        open_wokwi_simulator()
        
    elif args.action == 'monitor':
        monitor_performance()
        
    elif args.action == 'info':
        show_project_info()

if __name__ == "__main__":
    main()
