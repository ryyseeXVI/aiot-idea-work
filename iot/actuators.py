"""
Actuator Control Module
Supports both real hardware (MicroPython) and simulation mode (CPython)
"""

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

def set_relay(state):
    """Set relay state (True=ON, False=OFF)"""
    global _relay_state
    _relay_state = bool(state)
    
    if MICROPYTHON_MODE:
        relay.value(1 if state else 0)
    else:
        # Simulation mode - just track state
        print(f"🔌 RELAY: {'ON' if state else 'OFF'}")

def set_alert_led(state):
    """Set alert LED state (True=ON, False=OFF)"""
    global _alert_led_state
    _alert_led_state = bool(state)
    
    if MICROPYTHON_MODE:
        alert_led.value(1 if state else 0)
    else:
        # Simulation mode - just track state
        print(f"💡 ALERT LED: {'ON' if state else 'OFF'}")

def get_relay_state():
    """Get current relay state"""
    if MICROPYTHON_MODE:
        return relay.value() == 1
    else:
        return _relay_state

def get_alert_led_state():
    """Get current alert LED state"""
    if MICROPYTHON_MODE:
        return alert_led.value() == 1
    else:
        return _alert_led_state
