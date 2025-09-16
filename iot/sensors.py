"""
Sensor Interface Module
Supports both real hardware (MicroPython) and simulation mode (CPython)
"""

import time
import random
from typing import Optional, Union
from config_manager import get_gpio_pins

# Try to import MicroPython modules
try:
    import dht
    from machine import Pin, ADC
    MICROPYTHON_MODE = True
except ImportError:
    # Fallback to simulation mode for development
    MICROPYTHON_MODE = False
    print("📡 Running in simulation mode (no MicroPython hardware)")

# Get GPIO pins from config
gpio_config = get_gpio_pins()
TEMP_SENSOR_PIN = gpio_config['temp_sensor_pin']
WATER_LEAK_PIN = gpio_config['water_leak_pin']
VIBRATION_PIN = gpio_config['vibration_pin']

if MICROPYTHON_MODE:
    # Real hardware setup
    dht_sensor = dht.DHT22(Pin(TEMP_SENSOR_PIN))
    water_leak_sensor = Pin(WATER_LEAK_PIN, Pin.IN, Pin.PULL_UP)
    vibration_adc = ADC(Pin(VIBRATION_PIN))
    vibration_adc.atten(ADC.ATTN_11DB)
    vibration_adc.width(ADC.WIDTH_12BIT)

def get_temperature() -> float:
    """
    Get temperature reading from DHT22 sensor.
    
    Returns:
        Temperature in Celsius (real hardware) or simulated value (testing mode)
    """
    if HARDWARE_AVAILABLE:
        try:
            dht_sensor.measure()
            return float(dht_sensor.temperature())
        except Exception as e:
            print(f"Temperature sensor error: {e}")
            return 25.0  # Default fallback temperature
    else:
        # Simulation mode - return realistic temperature
        return 20.0 + random.uniform(-5, 15)

def read_humidity():
    """Read humidity sensor"""
    if MICROPYTHON_MODE:
        try:
            return dht_sensor.humidity()
        except Exception as e:
            print(f"Error reading humidity: {e}")
            return None
    else:
        # Simulation: realistic humidity
        base_humidity = 60.0 + 15 * (time.time() % 1800) / 1800
        noise = random.uniform(-5, 5)
        return round(max(0, min(100, base_humidity + noise)), 1)

def read_vibration():
    """Read vibration sensor (ADC value)"""
    if MICROPYTHON_MODE:
        try:
            return vibration_adc.read()
        except Exception as e:
            print(f"Error reading vibration: {e}")
            return 500  # Safe default
    else:
        # Simulation: vibration with occasional spikes
        base_vibration = 450
        if random.random() < 0.1:  # 10% chance of vibration spike
            return random.randint(700, 900)
        else:
            return base_vibration + random.randint(-50, 150)

def detect_water_leak():
    """Detect water leak (boolean)"""
    if MICROPYTHON_MODE:
        try:
            return water_leak_sensor.value() == 0
        except Exception as e:
            print(f"Error reading water leak sensor: {e}")
            return False
    else:
        # Simulation: occasional water leak events
        return random.random() < 0.05  # 5% chance of water leak
