"""
Unit tests for IoT configuration management
Tests configuration loading and access functionality
"""

import unittest
import sys
import os
import tempfile
import json
from unittest.mock import patch

# Add the iot module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'iot'))

import config_manager
from config_manager import load_config, get_config, get_default_config, update_config, save_config


class TestConfigManager(unittest.TestCase):
    """Test cases for configuration manager"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Reset global config
        config_manager._config = None
    
    def tearDown(self):
        """Clean up after tests"""
        config_manager._config = None
    
    def test_default_config_structure(self):
        """Test that default config has required structure"""
        default = get_default_config()
        
        # Check main sections exist
        self.assertIn('energy', default)
        self.assertIn('power', default)
        self.assertIn('thresholds', default)
        self.assertIn('gpio_pins', default)
        self.assertIn('data_collection', default)
        
        # Check energy section
        energy = default['energy']
        self.assertIn('peak_rate_per_kwh', energy)
        self.assertIn('off_peak_rate_per_kwh', energy)
        self.assertIn('off_peak_start_hour', energy)
        self.assertIn('off_peak_end_hour', energy)
        
        # Check thresholds section
        thresholds = default['thresholds']
        self.assertIn('temp_warning', thresholds)
        self.assertIn('vibration_warning', thresholds)
        
        # Check GPIO pins section
        gpio = default['gpio_pins']
        self.assertIn('temp_sensor_pin', gpio)
        self.assertIn('relay_pin', gpio)
        self.assertIn('alert_led_pin', gpio)
    
    def test_config_loading_missing_file(self):
        """Test config loading when file doesn't exist"""
        # Load non-existent file
        result = load_config("nonexistent.json")
        self.assertFalse(result)
        
        # Should fall back to default config
        config = get_config()
        self.assertEqual(config, get_default_config())
    
    def test_config_loading_valid_file(self):
        """Test config loading with valid file"""
        test_config = {
            "energy": {"peak_rate_per_kwh": 0.20},
            "thresholds": {"temp_warning": 30.0}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_file = f.name
        
        try:
            result = load_config(temp_file)
            self.assertTrue(result)
            
            config = get_config()
            self.assertEqual(config['energy']['peak_rate_per_kwh'], 0.20)
            self.assertEqual(config['thresholds']['temp_warning'], 30.0)
        finally:
            os.unlink(temp_file)
    
    def test_config_loading_invalid_json(self):
        """Test config loading with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            temp_file = f.name
        
        try:
            result = load_config(temp_file)
            self.assertFalse(result)
            
            # Should fall back to default config
            config = get_config()
            self.assertEqual(config, get_default_config())
        finally:
            os.unlink(temp_file)
    
    def test_get_config_sections(self):
        """Test getting specific config sections"""
        # Load default config
        load_config("nonexistent.json")  # Will load defaults
        
        # Test getting entire config
        full_config = get_config()
        self.assertIsInstance(full_config, dict)
        
        # Test getting specific section
        energy_config = get_config('energy')
        self.assertIn('peak_rate_per_kwh', energy_config)
        
        # Test getting specific key
        peak_rate = get_config('energy', 'peak_rate_per_kwh')
        self.assertIsInstance(peak_rate, (int, float))
    
    def test_get_config_invalid_section(self):
        """Test getting invalid config section"""
        load_config("nonexistent.json")
        
        with self.assertRaises(KeyError):
            get_config('nonexistent_section')
    
    def test_get_config_invalid_key(self):
        """Test getting invalid config key"""
        load_config("nonexistent.json")
        
        with self.assertRaises(KeyError):
            get_config('energy', 'nonexistent_key')
    
    def test_update_config(self):
        """Test updating config values"""
        load_config("nonexistent.json")
        
        # Update existing value
        update_config('energy', 'peak_rate_per_kwh', 0.25)
        updated_value = get_config('energy', 'peak_rate_per_kwh')
        self.assertEqual(updated_value, 0.25)
        
        # Update non-existing section
        update_config('new_section', 'new_key', 'new_value')
        new_value = get_config('new_section', 'new_key')
        self.assertEqual(new_value, 'new_value')
    
    def test_save_config(self):
        """Test saving config to file"""
        load_config("nonexistent.json")
        update_config('test', 'value', 123)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            result = save_config(temp_file)
            self.assertTrue(result)
            
            # Verify file was saved correctly
            with open(temp_file, 'r') as f:
                saved_config = json.load(f)
            
            self.assertEqual(saved_config['test']['value'], 123)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_convenience_functions(self):
        """Test convenience functions for common config access"""
        load_config("nonexistent.json")
        
        # Test get_energy_rates
        rates = config_manager.get_energy_rates()
        self.assertIn('peak_rate', rates)
        self.assertIn('off_peak_rate', rates)
        
        # Test get_off_peak_hours
        hours = config_manager.get_off_peak_hours()
        self.assertIn('start', hours)
        self.assertIn('end', hours)
        
        # Test get_thresholds
        thresholds = config_manager.get_thresholds()
        self.assertIn('temp_warning', thresholds)
        
        # Test get_gpio_pins
        pins = config_manager.get_gpio_pins()
        self.assertIn('relay_pin', pins)
        
        # Test get_power_settings
        power = config_manager.get_power_settings()
        self.assertIn('base_system_watts', power)


class TestConfigManagerIntegration(unittest.TestCase):
    """Integration tests for configuration manager"""
    
    def setUp(self):
        """Set up test fixtures"""
        config_manager._config = None
    
    def tearDown(self):
        """Clean up after tests"""
        config_manager._config = None
    
    def test_full_config_cycle(self):
        """Test complete configuration management cycle"""
        # Start with defaults
        load_config("nonexistent.json")
        original_temp = get_config('thresholds', 'temp_warning')
        
        # Update configuration
        new_temp = original_temp + 5.0
        update_config('thresholds', 'temp_warning', new_temp)
        
        # Verify update
        updated_temp = get_config('thresholds', 'temp_warning')
        self.assertEqual(updated_temp, new_temp)
        
        # Save and reload
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            save_config(temp_file)
            config_manager._config = None  # Reset
            load_config(temp_file)
            
            # Verify persistence
            reloaded_temp = get_config('thresholds', 'temp_warning')
            self.assertEqual(reloaded_temp, new_temp)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_config_validation(self):
        """Test that config values are reasonable"""
        load_config("nonexistent.json")
        
        # Energy rates should be positive
        rates = config_manager.get_energy_rates()
        self.assertGreater(rates['peak_rate'], 0)
        self.assertGreater(rates['off_peak_rate'], 0)
        
        # Off-peak hours should be valid
        hours = config_manager.get_off_peak_hours()
        self.assertGreaterEqual(hours['start'], 0)
        self.assertLess(hours['start'], 24)
        self.assertGreaterEqual(hours['end'], 0)
        self.assertLess(hours['end'], 24)
        
        # Thresholds should be reasonable
        thresholds = config_manager.get_thresholds()
        self.assertGreater(thresholds['temp_warning'], 10)
        self.assertLess(thresholds['temp_warning'], 50)
        self.assertGreater(thresholds['vibration_warning'], 0)
        
        # GPIO pins should be valid
        pins = config_manager.get_gpio_pins()
        for pin_name, pin_value in pins.items():
            self.assertIsInstance(pin_value, int)
            self.assertGreaterEqual(pin_value, 0)
            self.assertLess(pin_value, 40)  # Typical ESP32 pin range


if __name__ == '__main__':
    unittest.main()
