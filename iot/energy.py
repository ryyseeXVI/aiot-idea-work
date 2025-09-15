"""
Energy Management Module
Supports both real hardware (MicroPython) and simulation mode (CPython)
"""

import time
from actuators import set_relay
from config_manager import get_off_peak_hours, get_thresholds

# Try to import MicroPython RTC
try:
    from machine import RTC
    MICROPYTHON_MODE = True
    rtc = RTC()
except ImportError:
    # Fallback to simulation mode
    MICROPYTHON_MODE = False

def is_off_peak():
    """Check if current time is during off-peak hours"""
    if MICROPYTHON_MODE:
        hour = rtc.datetime()[4]  # MicroPython RTC format
    else:
        # Simulation mode - use system time
        hour = time.localtime().tm_hour
    
    # Get off-peak hours from config
    peak_hours = get_off_peak_hours()
    off_peak_start = peak_hours['start']
    off_peak_end = peak_hours['end']
    
    return hour >= off_peak_start or hour < off_peak_end

def optimize_energy_usage(temp):
    """Optimize energy usage based on temperature and time"""
    # Get threshold from config
    thresholds = get_thresholds()
    energy_temp_threshold = thresholds['energy_optimization_temp']
    
    if is_off_peak():
        # During off-peak hours, be more liberal with energy use
        set_relay(True)
    else:
        # During peak hours, only use energy if necessary
        if temp is not None and temp > energy_temp_threshold:
            set_relay(True)
        else:
            set_relay(False)
