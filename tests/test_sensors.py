"""
Unit tests for IoT sensor modules
Tests sensor reading functionality without hardware dependencies
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the iot module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'iot'))

# Mock hardware modules before importing
sys.modules['machine'] = MagicMock()
sys.modules['dht'] = MagicMock()

import sensors
from sensors import read_temperature, read_humidity, read_vibration, detect_water_leak


class TestSensors(unittest.TestCase):
    """Test cases for sensor module"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Ensure we're in simulation mode for testing
        sensors.MICROPYTHON_MODE = False
    
    def test_read_temperature_simulation(self):
        """Test temperature reading in simulation mode"""
        temp = read_temperature()
        
        self.assertIsNotNone(temp)
        self.assertIsInstance(temp, float)
        self.assertGreaterEqual(temp, 15.0)
        self.assertLessEqual(temp, 35.0)
    
    def test_read_humidity_simulation(self):
        """Test humidity reading in simulation mode"""
        humidity = read_humidity()
        
        self.assertIsNotNone(humidity)
        self.assertIsInstance(humidity, float)
        self.assertGreaterEqual(humidity, 0.0)
        self.assertLessEqual(humidity, 100.0)
    
    def test_read_vibration_simulation(self):
        """Test vibration reading in simulation mode"""
        vibration = read_vibration()
        
        self.assertIsNotNone(vibration)
        self.assertIsInstance(vibration, int)
        self.assertGreaterEqual(vibration, 300)
        self.assertLessEqual(vibration, 1000)
    
    def test_detect_water_leak_simulation(self):
        """Test water leak detection in simulation mode"""
        leak = detect_water_leak()
        
        self.assertIsInstance(leak, bool)
    
    def test_sensor_consistency(self):
        """Test that sensors return consistent types"""
        for _ in range(10):  # Test multiple readings
            temp = read_temperature()
            humidity = read_humidity()
            vibration = read_vibration()
            leak = detect_water_leak()
            
            # Type checks
            self.assertIsInstance(temp, float)
            self.assertIsInstance(humidity, float)
            self.assertIsInstance(vibration, int)
            self.assertIsInstance(leak, bool)
            
            # Range checks
            self.assertGreaterEqual(temp, 10.0)
            self.assertLessEqual(temp, 50.0)
            self.assertGreaterEqual(humidity, 0.0)
            self.assertLessEqual(humidity, 100.0)
            self.assertGreaterEqual(vibration, 200)
            self.assertLessEqual(vibration, 1200)
    
    @patch('sensors.dht_sensor')
    def test_temperature_hardware_error_handling(self, mock_dht):
        """Test temperature sensor error handling"""
        # Temporarily enable hardware mode
        sensors.MICROPYTHON_MODE = True
        
        # Mock sensor error
        mock_dht.measure.side_effect = Exception("Sensor error")
        
        temp = read_temperature()
        self.assertIsNone(temp)
        
        # Reset to simulation mode
        sensors.MICROPYTHON_MODE = False
    
    def test_multiple_readings_variation(self):
        """Test that multiple readings show realistic variation"""
        temperatures = [read_temperature() for _ in range(20)]
        humidities = [read_humidity() for _ in range(20)]
        vibrations = [read_vibration() for _ in range(20)]
        
        # Check for variation (not all identical)
        temp_range = max(temperatures) - min(temperatures)
        humidity_range = max(humidities) - min(humidities)
        vibration_range = max(vibrations) - min(vibrations)
        
        self.assertGreater(temp_range, 0.5)  # Some temperature variation
        self.assertGreater(humidity_range, 1.0)  # Some humidity variation
        self.assertGreater(vibration_range, 10)  # Some vibration variation


class TestSensorIntegration(unittest.TestCase):
    """Integration tests for sensor subsystem"""
    
    def test_all_sensors_readable(self):
        """Test that all sensors can be read without errors"""
        try:
            temp = read_temperature()
            humidity = read_humidity()
            vibration = read_vibration()
            leak = detect_water_leak()
            
            # All readings should be valid
            self.assertIsNotNone(temp)
            self.assertIsNotNone(humidity)
            self.assertIsNotNone(vibration)
            self.assertIsNotNone(leak)
            
        except Exception as e:
            self.fail(f"Sensor reading failed: {e}")
    
    def test_sensor_data_structure(self):
        """Test that sensor data can be structured for IoT processing"""
        sensor_data = {
            "temperature": read_temperature(),
            "humidity": read_humidity(),
            "vibration": read_vibration(),
            "water_leak": detect_water_leak(),
            "timestamp": 1234567890.0
        }
        
        # Validate data structure
        self.assertIn("temperature", sensor_data)
        self.assertIn("humidity", sensor_data)
        self.assertIn("vibration", sensor_data)
        self.assertIn("water_leak", sensor_data)
        self.assertIn("timestamp", sensor_data)
        
        # Validate data types
        self.assertIsInstance(sensor_data["temperature"], float)
        self.assertIsInstance(sensor_data["humidity"], float)
        self.assertIsInstance(sensor_data["vibration"], int)
        self.assertIsInstance(sensor_data["water_leak"], bool)
        self.assertIsInstance(sensor_data["timestamp"], float)


if __name__ == '__main__':
    unittest.main()
