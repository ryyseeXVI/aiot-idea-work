"""
ESP32 Adaptive Edge AI Controller for Wokwi Simulation
Ultra-lightweight MicroPython implementation with embedded AI
"""

import time
import math
import gc
from machine import Pin, ADC
import dht

# Embedded AI Model (ultra-lightweight for ESP32)
class EmbeddedAI:
    def __init__(self):
        # Feature normalization constants
        self.feature_means = [25.448, 62.336, 600.06, 0.5, 0.52, 0.5]
        self.feature_stds = [2.5292087, 10.758769, 116.43014, 0.5, 0.49959984, 0.5]
        
        # Ultra-compressed model weights (quantized INT8 scaled)
        self.model = {
            'l0_w': [
                [102, 43, 50, -100, -17, -8], [-100, 6, -119, -45, -1, -51],
                [-51, 77, 67, -79, -58, -3], [39, 36, 12, -38, 67, 32],
                [84, 9, -40, -12, -112, 7], [96, 42, 30, 30, -57, -101],
                [29, 109, 53, -115, -25, 44], [-103, 10, -63, -1, 61, -48],
                [23, -10, 126, 127, 51, 20], [-92, -63, -63, -21, 101, -70],
                [19, 81, -29, 109, 113, -31], [-23, -69, -36, -106, -102, 29],
                [-24, 23, -77, 41, -79, -63], [-25, 22, 20, -35, -52, -34],
                [67, -24, 78, 61, 118, 90], [-21, -98, 58, 12, 90, -90]
            ],
            'l0_s': 0.004024933,
            'l0_b': [-0.277, 0.193, -0.319, -0.323, 0.361, 0.249, -0.306, 0.237, -0.009, 0.234, 0.090, 0.291, 0.071, 0.425, -0.135, -0.086],
            
            'l1_w': [
                [64, -9, 6, 3, 87, -4, 10, -24, -36, -52, -65, 5, -39, 50, -29, 44],
                [33, 12, 17, -12, 71, 51, 39, -20, -53, -28, -4, 67, -28, 36, 60, 56],
                [-60, 44, 47, 49, -64, 27, -86, 62, 25, 70, -33, -10, -65, -39, 79, 1],
                [-43, 7, 18, 35, -46, 78, -43, 92, 111, 10, 78, 27, 28, -27, 62, 8],
                [65, -58, -18, 23, 38, -1, -3, 33, -32, 32, 17, 77, 12, -4, 33, -56],
                [-6, 32, -66, 51, 1, 17, -5, -20, 37, 91, 39, -29, -53, -45, 12, 12],
                [111, -3, -23, 58, -10, 63, -2, 19, -78, -31, -36, 77, -19, 22, 22, 17],
                [-5, -32, -21, -2, 71, 50, 97, -7, 10, -43, -35, 40, 26, 52, 17, -16]
            ],
            'l1_s': 0.002643115,
            'l1_b': [0.072, -0.104, 0.173, -0.026, -0.025, 0.204, -0.013, -0.050],
            
            'l2_w': [[120, 56, -43, -112, 21, -127, 51, 66]],
            'l2_s': 0.002400716,
            'l2_b': [0.137]
        }
    
    def relu(self, x):
        return max(0, x)
    
    def sigmoid(self, x):
        if x > 10: return 1.0
        if x < -10: return 0.0
        return 1.0 / (1.0 + math.exp(-x))
    
    def predict(self, features):
        """Ultra-fast AI inference for ESP32"""
        # Normalize features
        normalized = [(f - m) / s for f, m, s in zip(features, self.feature_means, self.feature_stds)]
        
        # Layer 0: 6 -> 16
        h1 = []
        for i in range(16):
            val = self.model['l0_b'][i]
            for j in range(6):
                val += normalized[j] * self.model['l0_w'][i][j] * self.model['l0_s']
            h1.append(self.relu(val))
        
        # Layer 1: 16 -> 8  
        h2 = []
        for i in range(8):
            val = self.model['l1_b'][i]
            for j in range(16):
                val += h1[j] * self.model['l1_w'][i][j] * self.model['l1_s']
            h2.append(self.relu(val))
        
        # Layer 2: 8 -> 1
        val = self.model['l2_b'][0]
        for j in range(8):
            val += h2[j] * self.model['l2_w'][0][j] * self.model['l2_s']
        
        confidence = self.sigmoid(val)
        decision = confidence > 0.5
        return decision, confidence

# Adaptive Threshold Engine (simplified for ESP32)
class AdaptiveThresholds:
    def __init__(self):
        self.temp_warning = 28.0
        self.vibration_warning = 700
        self.cooling_aggressiveness = 0.5
        self.confidence_threshold = 0.7
        self.power_efficiency = False
        
        # Seasonal adjustments
        self.season_offsets = {
            'winter': {'temp': -2.0, 'cooling': -0.2, 'power': True},
            'spring': {'temp': -1.0, 'cooling': -0.1, 'power': False},
            'summer': {'temp': 2.0, 'cooling': 0.3, 'power': False},
            'fall': {'temp': -1.0, 'cooling': -0.1, 'power': True}
        }
    
    def get_season(self):
        """Simple season detection based on time of year"""
        month = time.localtime()[1]  # Month (1-12)
        if month in [12, 1, 2]: return 'winter'
        elif month in [3, 4, 5]: return 'spring'
        elif month in [6, 7, 8]: return 'summer'
        else: return 'fall'
    
    def get_time_period(self):
        """Get time of day period"""
        hour = time.localtime()[3]  # Hour (0-23)
        if 6 <= hour < 12: return 'morning'
        elif 12 <= hour < 18: return 'afternoon'
        elif 18 <= hour < 22: return 'evening'
        else: return 'night'
    
    def adapt(self, sensor_data, performance_score=0.8):
        """Adapt thresholds based on context and performance"""
        season = self.get_season()
        time_period = self.get_time_period()
        
        # Apply seasonal adjustments
        if season in self.season_offsets:
            offset = self.season_offsets[season]
            self.temp_warning = 28.0 + offset['temp']
            self.cooling_aggressiveness = max(0.1, min(0.9, 0.5 + offset['cooling']))
            self.power_efficiency = offset['power']
        
        # Time-based adjustments
        if time_period == 'night':
            self.temp_warning += 1.0  # More tolerant at night
            self.power_efficiency = True
        elif time_period == 'afternoon':
            self.temp_warning -= 0.5  # More aggressive in afternoon heat
        
        # Performance-based adaptation
        if performance_score < 0.6:
            self.temp_warning -= 0.5  # Be more aggressive if performance is poor
            self.cooling_aggressiveness += 0.1
        
        # Clamp values
        self.temp_warning = max(20.0, min(35.0, self.temp_warning))
        self.cooling_aggressiveness = max(0.1, min(0.9, self.cooling_aggressiveness))
    
    def get_thresholds(self):
        return {
            'temp_warning': self.temp_warning,
            'vibration_warning': self.vibration_warning,
            'cooling_aggressiveness': self.cooling_aggressiveness,
            'confidence_threshold': self.confidence_threshold,
            'power_efficiency': self.power_efficiency
        }

# Hardware Configuration
class HardwareConfig:
    # GPIO Pins for Wokwi ESP32 (matching config.json and diagram.json)
    DHT_PIN = 4          # DHT22 temperature/humidity sensor
    RELAY_PIN = 5        # Relay module control
    LED_PIN = 15         # Status LED (red)
    VIBRATION_PIN = 33   # Potentiometer simulating vibration sensor (A6)
    WATER_LEAK_PIN = 14  # Push button simulating water leak sensor
    
    def __init__(self):
        # Initialize hardware
        self.dht_sensor = dht.DHT22(Pin(self.DHT_PIN))
        self.relay = Pin(self.RELAY_PIN, Pin.OUT)
        self.led = Pin(self.LED_PIN, Pin.OUT)
        
        # Vibration sensor (potentiometer on ADC pin 32)
        self.vibration_adc = ADC(Pin(self.VIBRATION_PIN))
        self.vibration_adc.atten(ADC.ATTN_11DB)
        self.vibration_adc.width(ADC.WIDTH_12BIT)
        
        # Water leak sensor (button - pressed = leak detected)
        self.water_leak = Pin(self.WATER_LEAK_PIN, Pin.IN, Pin.PULL_UP)
        
        # Initial state
        self.relay.off()
        self.led.off()
        
        print("🚀 ESP32 Hardware initialized")
        print(f"   DHT22 on pin {self.DHT_PIN}")
        print(f"   Relay on pin {self.RELAY_PIN}")
        print(f"   LED on pin {self.LED_PIN}")
        print(f"   Vibration ADC on pin {self.VIBRATION_PIN}")
        print(f"   Water leak button on pin {self.WATER_LEAK_PIN}")

# Main Controller Class
class ESP32AdaptiveController:
    def __init__(self):
        self.hardware = HardwareConfig()
        self.ai_model = EmbeddedAI()
        self.adaptive_thresholds = AdaptiveThresholds()
        
        # System state
        self.relay_state = False
        self.led_state = False
        self.cycle_count = 0
        self.relay_changes = 0
        self.last_adaptation = 0
        self.safety_overrides = 0
        
        # Performance tracking
        self.total_cycle_time = 0
        self.performance_scores = []
        
        print("🧠 ESP32 Adaptive AI Controller ready")
        print(f"   AI Model: Embedded (Ultra-lightweight)")
        print(f"   Memory: {gc.mem_free()} bytes free")
    
    def read_sensors(self):
        """Read all sensors with error handling"""
        try:
            # Temperature and humidity (DHT22)
            self.hardware.dht_sensor.measure()
            temp = self.hardware.dht_sensor.temperature()
            humidity = self.hardware.dht_sensor.humidity()
            
            # Vibration (ADC)
            vibration = self.hardware.vibration_adc.read()
            
            # Water leak (digital)
            water_leak = self.hardware.water_leak.value() == 0
            
            # Additional features
            off_peak = (time.localtime()[3] >= 22 or time.localtime()[3] <= 6)
            alert_active = self.led_state
            
            return {
                'temperature': temp,
                'humidity': humidity,
                'vibration': vibration,
                'water_leak': water_leak,
                'off_peak': off_peak,
                'alert_active': alert_active
            }
            
        except Exception as e:
            print(f"Sensor error: {e}")
            return None
    
    def safety_check(self, sensor_data):
        """Critical safety checks"""
        # Water leak emergency
        if sensor_data['water_leak']:
            print("🚨 SAFETY: Water leak detected!")
            self.safety_overrides += 1
            return False, True  # relay_off, led_on
        
        # Extreme temperature
        if sensor_data['temperature'] > 40.0:
            print(f"🚨 SAFETY: Extreme temperature {sensor_data['temperature']}°C")
            self.safety_overrides += 1
            return False, True
        
        # Extreme vibration
        if sensor_data['vibration'] > 3000:
            print(f"🚨 SAFETY: Extreme vibration {sensor_data['vibration']}")
            self.safety_overrides += 1
            return False, True
        
        return None, None  # No safety override
    
    def make_decision(self, sensor_data, thresholds):
        """Make AI decision with adaptive thresholds"""
        try:
            # Prepare features for AI
            features = [
                sensor_data['temperature'],
                sensor_data['humidity'],
                sensor_data['vibration'],
                float(sensor_data['water_leak']),
                float(sensor_data['off_peak']),
                float(sensor_data['alert_active'])
            ]
            
            # AI inference
            ai_decision, confidence = self.ai_model.predict(features)
            
            # Apply adaptive confidence threshold
            final_decision = ai_decision and (confidence > thresholds['confidence_threshold'])
            
            # Temperature override for hot conditions
            if sensor_data['temperature'] > thresholds['temp_warning']:
                final_decision = True
                confidence = max(confidence, 0.8)
            
            # LED decision (alert when low confidence or problems)
            led_decision = (
                confidence < thresholds['confidence_threshold'] or
                sensor_data['vibration'] > thresholds['vibration_warning'] or
                self.relay_changes > 5
            )
            
            return final_decision, led_decision, confidence, "AI"
            
        except Exception as e:
            print(f"AI error: {e}")
            # Fallback to simple rules
            relay_on = sensor_data['temperature'] > thresholds['temp_warning']
            led_on = sensor_data['vibration'] > thresholds['vibration_warning']
            return relay_on, led_on, 0.6, "RULES"
    
    def update_actuators(self, relay_state, led_state):
        """Update hardware outputs"""
        # Track state changes
        if relay_state != self.relay_state:
            self.relay_changes += 1
            print(f"🔌 Relay: {'ON' if relay_state else 'OFF'}")
        
        if led_state != self.led_state:
            print(f"💡 LED: {'ON' if led_state else 'OFF'}")
        
        # Update hardware
        if relay_state:
            self.hardware.relay.on()
        else:
            self.hardware.relay.off()
            
        if led_state:
            self.hardware.led.on()
        else:
            self.hardware.led.off()
        
        # Update state
        self.relay_state = relay_state
        self.led_state = led_state
    
    def calculate_performance(self, sensor_data, thresholds):
        """Calculate simple performance score"""
        comfort_score = 1.0
        
        # Temperature comfort
        target_temp = thresholds['temp_warning'] - 2.0
        temp_diff = abs(sensor_data['temperature'] - target_temp)
        comfort_score *= max(0.0, 1.0 - temp_diff / 10.0)
        
        # Vibration comfort
        if sensor_data['vibration'] > thresholds['vibration_warning']:
            comfort_score *= 0.7
        
        # Water leak penalty
        if sensor_data['water_leak']:
            comfort_score *= 0.3
        
        return max(0.0, min(1.0, comfort_score))
    
    def process_cycle(self):
        """Process one control cycle"""
        cycle_start = time.ticks_ms()
        self.cycle_count += 1
        
        try:
            # Read sensors
            sensor_data = self.read_sensors()
            if not sensor_data:
                return True  # Continue despite sensor error
            
            # Get current thresholds
            thresholds = self.adaptive_thresholds.get_thresholds()
            
            # Adapt thresholds periodically
            current_time = time.time()
            if current_time - self.last_adaptation > 60:  # Every minute
                performance = self.calculate_performance(sensor_data, thresholds)
                self.adaptive_thresholds.adapt(sensor_data, performance)
                self.last_adaptation = current_time
                print(f"🔄 Adapted (T:{thresholds['temp_warning']:.1f}°C, C:{thresholds['cooling_aggressiveness']:.2f})")
            
            # Safety checks first
            safety_relay, safety_led = self.safety_check(sensor_data)
            
            if safety_relay is not None:
                # Safety override
                relay_decision = safety_relay
                led_decision = safety_led
                confidence = 1.0
                decision_source = "SAFETY"
            else:
                # Normal AI decision
                relay_decision, led_decision, confidence, decision_source = self.make_decision(sensor_data, thresholds)
            
            # Update actuators
            self.update_actuators(relay_decision, led_decision)
            
            # Performance tracking
            cycle_time = time.ticks_diff(time.ticks_ms(), cycle_start)
            self.total_cycle_time += cycle_time
            
            # Periodic status
            if self.cycle_count % 30 == 0:
                avg_cycle = self.total_cycle_time / self.cycle_count
                print(f"📊 Cycle {self.cycle_count}: {sensor_data['temperature']:.1f}°C, {sensor_data['vibration']}, {decision_source}, {avg_cycle:.1f}ms")
                print(f"   Memory: {gc.mem_free()} bytes, Changes: {self.relay_changes}, Safety: {self.safety_overrides}")
                
                # Collect garbage to free memory
                gc.collect()
            
            return True
            
        except Exception as e:
            print(f"Cycle error: {e}")
            return True  # Continue despite error
    
    def run(self, duration_seconds=None):
        """Main control loop"""
        print("🚀 Starting ESP32 Adaptive AI Control")
        print(f"   Duration: {duration_seconds if duration_seconds else 'Infinite'} seconds")
        
        start_time = time.time()
        
        try:
            while True:
                # Check duration
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    break
                
                # Process one cycle
                if not self.process_cycle():
                    break
                
                # Sleep between cycles
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("🛑 Stopped by user")
        except Exception as e:
            print(f"Runtime error: {e}")
        finally:
            # Cleanup
            self.hardware.relay.off()
            self.hardware.led.off()
            
            # Final stats
            if self.cycle_count > 0:
                avg_cycle = self.total_cycle_time / self.cycle_count
                uptime = time.time() - start_time
                print(f"\n📈 Final Statistics:")
                print(f"   Cycles: {self.cycle_count}")
                print(f"   Avg cycle: {avg_cycle:.1f}ms")
                print(f"   Uptime: {uptime:.1f}s")
                print(f"   Relay changes: {self.relay_changes}")
                print(f"   Safety overrides: {self.safety_overrides}")
                print(f"   Memory: {gc.mem_free()} bytes")
            
            print("🔒 ESP32 Controller shutdown")

# Main execution
if __name__ == "__main__":
    print("🌟 ESP32 Adaptive Edge AI IoT System")
    print("   Ultra-lightweight MicroPython implementation")
    print("   Integrated AI model running directly on chip")
    
    # Initialize controller
    controller = ESP32AdaptiveController()
    
    # Run for demo (adjust as needed)
    controller.run(duration_seconds=300)  # 5 minutes demo
