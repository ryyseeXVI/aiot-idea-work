"""
Actuator Control Module
Manages relay and buzzer control for ESP32 IoT system
"""

import time
from typing import Optional
from config_manager import get_gpio_pins

from config_manager import get_gpio_pins

# Try to import MicroPython modules
try:
    from machine import Pin
    MICROPYTHON_MODE = True
except ImportError:
    # Fallback to simulation mode
    MICROPYTHON_MODE = False

# Get GPIO pins from config
gpio_config = get_gpio_pins()
RELAY_PIN = gpio_config['relay_pin']
ALERT_LED_PIN = gpio_config['alert_led_pin']

# State tracking for simulation mode
_relay_state = False
_alert_led_state = False

if MICROPYTHON_MODE:
    # Real hardware setup
    relay = Pin(RELAY_PIN, Pin.OUT)
    alert_led = Pin(ALERT_LED_PIN, Pin.OUT)

def set_relay(state: bool) -> bool:
    """
    Control relay state.
    
    Args:
        state: True to turn ON, False to turn OFF
        
    Returns:
        Actual relay state after operation
    """
    global relay_state
    if HARDWARE_AVAILABLE:
        try:
            relay.value(1 if state else 0)
            relay_state = state
            print(f"Relay {'ON' if state else 'OFF'}")
            return state
        except Exception as e:
            print(f"Relay control error: {e}")
            return relay_state  # Return current state on error
    else:
        # Simulation mode
        relay_state = state
        print(f"[SIM] Relay {'ON' if state else 'OFF'}")
        return state

def set_alert_led(state):
    """Set alert LED state (True=ON, False=OFF)"""
    global _alert_led_state
    _alert_led_state = bool(state)
    
    if MICROPYTHON_MODE:
        alert_led.value(1 if state else 0)
    else:
        # Simulation mode - just track state
        print(f"💡 ALERT LED: {'ON' if state else 'OFF'}")

def get_relay_state() -> bool:
    """
    Get current relay state.
    
    Returns:
        Current relay state (True=ON, False=OFF)
    """

def get_alert_led_state():
    """Get current alert LED state"""
    if MICROPYTHON_MODE:
        return alert_led.value() == 1
    else:
        return _alert_led_state
