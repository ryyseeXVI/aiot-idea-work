"""
Configuration Management Module
Handles loading and accessing configuration values from config.json
"""

import json
import os

# Global configuration dictionary
_config = None

def load_config(config_file="config.json"):
    """Load configuration from JSON file"""
    global _config
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), config_file)
        with open(config_path, 'r') as f:
            _config = json.load(f)
        print(f"✅ Configuration loaded from {config_file}")
        return True
    except FileNotFoundError:
        print(f"❌ Config file {config_file} not found, using defaults")
        _config = get_default_config()
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        _config = get_default_config()
        return False

def get_default_config():
    """Return default configuration if file loading fails"""
    return {
        "energy": {
            "peak_rate_per_kwh": 0.15,
            "off_peak_rate_per_kwh": 0.08,
            "off_peak_start_hour": 22,
            "off_peak_end_hour": 6
        },
        "power": {
            "base_system_watts": 50,
            "cooling_system_watts": 200,
            "temp_power_factor": 5,
            "vibration_power_factor": 0.1
        },
        "thresholds": {
            "temp_warning": 28.0,
            "vibration_warning": 700,
            "energy_optimization_temp": 25.0
        },
        "gpio_pins": {
            "temp_sensor_pin": 4,
            "water_leak_pin": 14,
            "vibration_pin": 33,
            "relay_pin": 5,
            "alert_led_pin": 15
        },
        "data_collection": {
            "sensor_history_size": 10,
            "sampling_interval_seconds": 10,
            "log_file": "sensor_data.json"
        }
    }

def get_config(section=None, key=None):
    """Get configuration value(s)"""
    global _config
    
    # Load config if not already loaded
    if _config is None:
        load_config()
    
    # Ensure _config is not None after loading
    if _config is None:
        _config = get_default_config()
    
    if section is None:
        return _config
    
    if section not in _config:
        raise KeyError(f"Configuration section '{section}' not found")
    
    if key is None:
        return _config[section]
    
    if key not in _config[section]:
        raise KeyError(f"Configuration key '{key}' not found in section '{section}'")
    
    return _config[section][key]

def update_config(section, key, value):
    """Update configuration value at runtime"""
    global _config
    
    if _config is None:
        load_config()
    
    # Ensure _config is not None
    if _config is None:
        _config = get_default_config()
    
    if section not in _config:
        _config[section] = {}
    
    _config[section][key] = value
    print(f"🔧 Updated config: {section}.{key} = {value}")

def save_config(config_file="config.json"):
    """Save current configuration back to file"""
    global _config
    
    if _config is None:
        print("❌ No configuration loaded to save")
        return False
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), config_file)
        with open(config_path, 'w') as f:
            json.dump(_config, f, indent=2)
        print(f"💾 Configuration saved to {config_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to save config: {e}")
        return False

# Convenience functions for common config access
def get_energy_rates():
    """Get energy pricing rates"""
    energy_config = get_config('energy')
    return {
        'peak_rate': energy_config['peak_rate_per_kwh'],
        'off_peak_rate': energy_config['off_peak_rate_per_kwh']
    }

def get_off_peak_hours():
    """Get off-peak time range"""
    energy_config = get_config('energy')
    return {
        'start': energy_config['off_peak_start_hour'],
        'end': energy_config['off_peak_end_hour']
    }

def get_thresholds():
    """Get all warning thresholds"""
    return get_config('thresholds')

def get_gpio_pins():
    """Get GPIO pin assignments"""
    return get_config('gpio_pins')

def get_power_settings():
    """Get power calculation settings"""
    return get_config('power')

def get_data_settings():
    """Get data collection settings"""
    return get_config('data_collection')
