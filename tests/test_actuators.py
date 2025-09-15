"""
Unit tests for IoT actuator modules
Tests actuator control functionality without hardware dependencies
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the iot module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'iot'))

# Mock hardware modules before importing
sys.modules['machine'] = MagicMock()

import actuators
from actuators import set_relay, set_alert_led, get_relay_state, get_alert_led_state


class TestActuators(unittest.TestCase):
    """Test cases for actuator module"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Ensure we're in simulation mode for testing
        actuators.MICROPYTHON_MODE = False
        # Reset states
        actuators._relay_state = False
        actuators._alert_led_state = False
    
    def test_relay_control(self):
        """Test relay on/off control"""
        # Test turning relay on
        set_relay(True)
        self.assertTrue(get_relay_state())
        
        # Test turning relay off
        set_relay(False)
        self.assertFalse(get_relay_state())
    
    def test_alert_led_control(self):
        """Test alert LED on/off control"""
        # Test turning LED on
        set_alert_led(True)
        self.assertTrue(get_alert_led_state())
        
        # Test turning LED off
        set_alert_led(False)
        self.assertFalse(get_alert_led_state())
    
    def test_relay_state_persistence(self):
        """Test that relay state persists between calls"""
        set_relay(True)
        self.assertTrue(get_relay_state())
        self.assertTrue(get_relay_state())  # Should still be true
        
        set_relay(False)
        self.assertFalse(get_relay_state())
        self.assertFalse(get_relay_state())  # Should still be false
    
    def test_alert_led_state_persistence(self):
        """Test that alert LED state persists between calls"""
        set_alert_led(True)
        self.assertTrue(get_alert_led_state())
        self.assertTrue(get_alert_led_state())  # Should still be true
        
        set_alert_led(False)
        self.assertFalse(get_alert_led_state())
        self.assertFalse(get_alert_led_state())  # Should still be false
    
    def test_type_conversion(self):
        """Test that actuators handle different input types"""
        # Test integer inputs
        set_relay(1)
        self.assertTrue(get_relay_state())
        
        set_relay(0)
        self.assertFalse(get_relay_state())
        
        # Test string inputs (truthy/falsy)
        set_alert_led("true")
        self.assertTrue(get_alert_led_state())
        
        set_alert_led("")
        self.assertFalse(get_alert_led_state())
    
    def test_simultaneous_control(self):
        """Test controlling both actuators simultaneously"""
        # Both off
        set_relay(False)
        set_alert_led(False)
        self.assertFalse(get_relay_state())
        self.assertFalse(get_alert_led_state())
        
        # Both on
        set_relay(True)
        set_alert_led(True)
        self.assertTrue(get_relay_state())
        self.assertTrue(get_alert_led_state())
        
        # Mixed states
        set_relay(True)
        set_alert_led(False)
        self.assertTrue(get_relay_state())
        self.assertFalse(get_alert_led_state())
        
        set_relay(False)
        set_alert_led(True)
        self.assertFalse(get_relay_state())
        self.assertTrue(get_alert_led_state())
    
    @patch('actuators.relay')
    def test_hardware_relay_control(self, mock_relay):
        """Test hardware relay control when in MicroPython mode"""
        # Temporarily enable hardware mode
        actuators.MICROPYTHON_MODE = True
        
        set_relay(True)
        mock_relay.value.assert_called_with(1)
        
        set_relay(False)
        mock_relay.value.assert_called_with(0)
        
        # Reset to simulation mode
        actuators.MICROPYTHON_MODE = False
    
    @patch('actuators.alert_led')
    def test_hardware_led_control(self, mock_led):
        """Test hardware LED control when in MicroPython mode"""
        # Temporarily enable hardware mode
        actuators.MICROPYTHON_MODE = True
        
        set_alert_led(True)
        mock_led.value.assert_called_with(1)
        
        set_alert_led(False)
        mock_led.value.assert_called_with(0)
        
        # Reset to simulation mode
        actuators.MICROPYTHON_MODE = False


class TestActuatorIntegration(unittest.TestCase):
    """Integration tests for actuator subsystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        actuators.MICROPYTHON_MODE = False
        actuators._relay_state = False
        actuators._alert_led_state = False
    
    def test_complete_control_cycle(self):
        """Test a complete control cycle"""
        # Start with everything off
        self.assertFalse(get_relay_state())
        self.assertFalse(get_alert_led_state())
        
        # Simulate normal operation
        set_relay(True)  # Turn on cooling
        self.assertTrue(get_relay_state())
        
        # Alert condition
        set_alert_led(True)
        self.assertTrue(get_alert_led_state())
        
        # Clear alert but keep cooling
        set_alert_led(False)
        self.assertFalse(get_alert_led_state())
        self.assertTrue(get_relay_state())  # Should still be on
        
        # Turn off cooling
        set_relay(False)
        self.assertFalse(get_relay_state())
        self.assertFalse(get_alert_led_state())
    
    def test_safety_shutdown_simulation(self):
        """Simulate emergency shutdown scenario"""
        # Normal operation
        set_relay(True)
        set_alert_led(False)
        
        # Emergency: turn off relay, turn on alert
        set_relay(False)  # Emergency shutdown
        set_alert_led(True)  # Alert active
        
        self.assertFalse(get_relay_state())
        self.assertTrue(get_alert_led_state())
    
    def test_actuator_state_dict(self):
        """Test creating a state dictionary for monitoring"""
        set_relay(True)
        set_alert_led(False)
        
        state = {
            "relay_on": get_relay_state(),
            "alert_active": get_alert_led_state()
        }
        
        self.assertEqual(state["relay_on"], True)
        self.assertEqual(state["alert_active"], False)
        
        # Change states
        set_relay(False)
        set_alert_led(True)
        
        state = {
            "relay_on": get_relay_state(),
            "alert_active": get_alert_led_state()
        }
        
        self.assertEqual(state["relay_on"], False)
        self.assertEqual(state["alert_active"], True)


if __name__ == '__main__':
    unittest.main()
