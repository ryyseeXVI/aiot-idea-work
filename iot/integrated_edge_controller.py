"""
Integrated Edge AI IoT Controller
Fully integrated AI model running directly on the IoT device with adaptive intelligence
"""

import time
import json
import sys
import logging
import traceback
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

# Local imports
from adaptive_thresholds import (
    AdaptiveThresholdEngine, 
    calculate_comfort_score, 
    calculate_energy_efficiency, 
    calculate_stability_score
)

# Import embedded AI model (ultra-lightweight)
try:
    import embedded_ai_model
    EMBEDDED_AI_AVAILABLE = True
    print("✅ Embedded AI model loaded")
except ImportError:
    print("⚠️ Embedded AI model not available, running in rule-based mode")
    EMBEDDED_AI_AVAILABLE = False

# PyTorch available only for development (not on ESP32)
TORCH_AVAILABLE = False

# Import IoT modules
try:
    from sensors import read_temperature, read_humidity, read_vibration, detect_water_leak
    from energy import is_off_peak
    from actuators import set_relay, set_alert_led, get_relay_state, get_alert_led_state
    from config_manager import load_config
    HARDWARE_AVAILABLE = True
except ImportError:
    print("⚠️ Hardware modules not available, using simulation mode")
    HARDWARE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SystemState:
    """Current system state for performance tracking"""
    relay_on: bool = False
    alert_active: bool = False
    relay_changes_count: int = 0
    last_relay_change: float = 0.0
    total_runtime_minutes: float = 0.0
    power_consumption: float = 0.0
    comfort_violations: int = 0
    safety_overrides: int = 0

class IntegratedEdgeAIController:
    """
    Fully integrated AI IoT controller that runs everything in one process
    Designed for true edge deployment with adaptive intelligence
    """
    
    def __init__(self, model_path: str = "../model/relay_model.pth", 
                 scaler_path: str = "../model/scaler.pkl",
                 config_path: str = "config.json"):
        
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config_path = config_path
        
        # Initialize adaptive threshold engine
        self.adaptive_engine = AdaptiveThresholdEngine(config_path)
        
        # Initialize embedded AI model (primary)
        self.embedded_ai_available = EMBEDDED_AI_AVAILABLE
        
        # Fallback PyTorch model (development only)
        self.ai_model = None
        self.scaler = None
        self.pytorch_available = False
        if TORCH_AVAILABLE:
            self._initialize_pytorch_model()
        
        # System state tracking
        self.system_state = SystemState()
        self.sensor_history: List[Dict] = []
        self.performance_history: List[Dict] = []
        
        # Performance metrics
        self.cycle_count = 0
        self.total_processing_time = 0.0
        self.last_adaptation_time = 0.0
        
        # Safety and control parameters
        self.safety_temp_limit = 40.0
        self.safety_vibration_limit = 1000
        self.max_relay_changes_per_minute = 6
        
        # Load base configuration
        if HARDWARE_AVAILABLE:
            try:
                from config_manager import load_config
                load_config()
            except ImportError:
                logger.warning("Config manager not available")
        
        logger.info("🚀 Integrated Edge AI Controller initialized")
        logger.info(f"   Embedded AI: {'Available' if self.embedded_ai_available else 'Not available'}")
        logger.info(f"   PyTorch AI: {'Available' if self.pytorch_available else 'Not available'}")
        logger.info(f"   Adaptive Thresholds: Enabled")
        logger.info(f"   Hardware: {'Real' if HARDWARE_AVAILABLE else 'Simulation'}")
        
    def _initialize_pytorch_model(self):
        """PyTorch model not available on ESP32 - kept for development reference only"""
        logger.info("PyTorch model disabled for ESP32 deployment")
        self.pytorch_available = False
    
    def _read_sensors(self) -> Optional[Dict[str, Any]]:
        """Read all sensors with error handling"""
        try:
            if HARDWARE_AVAILABLE:
                # Real hardware sensors
                from sensors import read_temperature, read_humidity, read_vibration, detect_water_leak
                from energy import is_off_peak
                from actuators import get_alert_led_state
                
                temp = read_temperature()
                humidity = read_humidity()
                vibration = read_vibration()
                water_leak = detect_water_leak()
                off_peak = is_off_peak()
                alert_active = get_alert_led_state()
            else:
                # Simulation mode
                import random
                temp = random.uniform(18, 32)
                humidity = random.uniform(40, 80)
                vibration = random.randint(300, 900)
                water_leak = random.random() < 0.05  # 5% chance
                off_peak = random.choice([True, False])
                alert_active = self.system_state.alert_active
            
            return {
                "timestamp": time.time(),
                "temperature": float(temp) if temp is not None else 25.0,
                "humidity": float(humidity) if humidity is not None else 60.0,
                "vibration": int(vibration),
                "water_leak": bool(water_leak),
                "off_peak": bool(off_peak),
                "alert_active": bool(alert_active)
            }
            
        except Exception as e:
            logger.error(f"Sensor reading error: {e}")
            return None
    
    def _safety_check(self, sensor_data: Dict) -> Optional[bool]:
        """Critical safety checks - returns relay state or None if no override needed"""
        
        # Water leak - immediate shutdown
        if sensor_data.get("water_leak", False):
            logger.warning("🚨 SAFETY: Water leak detected - forcing relay OFF")
            self.system_state.safety_overrides += 1
            return False
        
        # Extreme temperature
        if sensor_data.get("temperature", 25) > self.safety_temp_limit:
            logger.warning(f"🚨 SAFETY: Extreme temperature - forcing relay OFF")
            self.system_state.safety_overrides += 1
            return False
        
        # Extreme vibration
        if sensor_data.get("vibration", 500) > self.safety_vibration_limit:
            logger.warning(f"🚨 SAFETY: Extreme vibration - forcing relay OFF")
            self.system_state.safety_overrides += 1
            return False
        
        # Prevent relay oscillation
        current_time = time.time()
        if (current_time - self.system_state.last_relay_change < 30 and 
            self.system_state.relay_changes_count > self.max_relay_changes_per_minute):
            logger.warning("🚨 SAFETY: Preventing relay oscillation")
            return self.system_state.relay_on  # Keep current state
        
        return None  # No safety override needed
    
    def _ai_decision(self, sensor_data: Dict, adaptive_thresholds: Dict) -> tuple[bool, float]:
        """Get AI decision with adaptive threshold integration - Embedded AI first"""
        
        # Try embedded AI model first (preferred for edge deployment)
        if self.embedded_ai_available:
            try:
                # Import embedded AI locally to avoid global import issues
                import embedded_ai_model
                
                # Prepare features for embedded AI model
                features = [
                    sensor_data["temperature"],
                    sensor_data["humidity"],
                    sensor_data["vibration"],
                    float(sensor_data["water_leak"]),
                    float(sensor_data["off_peak"]),
                    float(sensor_data["alert_active"])
                ]
                
                # Use embedded AI inference
                predicted_state, raw_confidence = embedded_ai_model.ai_predict(features)
                
                # Apply adaptive confidence threshold
                confidence_threshold = adaptive_thresholds.get("confidence_threshold", 0.7)
                
                # Adjust confidence based on adaptive thresholds
                if sensor_data["temperature"] > adaptive_thresholds.get("temp_warning", 28):
                    # Force more aggressive cooling in hot conditions
                    predicted_state = True
                    raw_confidence = max(raw_confidence, 0.8)
                
                # Override decision based on confidence threshold
                final_decision = predicted_state and (raw_confidence > confidence_threshold)
                
                return final_decision, raw_confidence
                
            except Exception as e:
                logger.error(f"Embedded AI inference error: {e}")
                # Fall through to PyTorch or rule-based
        
        # Try PyTorch model as fallback (development/testing - not available on ESP32)
        if False:  # Disabled for ESP32 deployment
            pass
        
        # Rule-based fallback
        return self._rule_based_decision(sensor_data, adaptive_thresholds)
    
    def _rule_based_decision(self, sensor_data: Dict, adaptive_thresholds: Dict) -> tuple[bool, float]:
        """Rule-based fallback decision with adaptive thresholds"""
        temp = sensor_data["temperature"]
        vibration = sensor_data["vibration"]
        off_peak = sensor_data["off_peak"]
        
        # Use adaptive thresholds
        temp_threshold = adaptive_thresholds.get("temp_warning", 28.0)
        vibration_threshold = adaptive_thresholds.get("vibration_warning", 700)
        cooling_aggressiveness = adaptive_thresholds.get("cooling_aggressiveness", 0.5)
        
        # Adaptive rule-based logic
        relay_on = False
        confidence = 0.6  # Base confidence for rules
        
        # Temperature-based decision with adaptive aggressiveness
        if temp > temp_threshold:
            relay_on = True
            confidence = 0.8
        elif temp > temp_threshold - (2.0 * cooling_aggressiveness):
            relay_on = True
            confidence = 0.7
        
        # Vibration consideration
        if vibration > vibration_threshold:
            confidence *= 0.8  # Less confident with high vibration
        
        # Energy optimization
        if off_peak and adaptive_thresholds.get("power_efficiency_mode", False):
            if not relay_on:  # Don't override cooling needs
                confidence += 0.1
        elif not off_peak:  # Peak hours
            confidence *= 0.9  # Slightly less aggressive
        
        return relay_on, confidence
    
    def _update_actuators(self, relay_state: bool, alert_state: bool):
        """Update physical actuators and track state changes"""
        try:
            current_time = time.time()
            
            # Track relay state changes
            if relay_state != self.system_state.relay_on:
                self.system_state.relay_changes_count += 1
                self.system_state.last_relay_change = current_time
                
                # Reset counter every minute
                if current_time - self.system_state.last_relay_change > 60:
                    self.system_state.relay_changes_count = 0
            
            # Update hardware
            if HARDWARE_AVAILABLE:
                from actuators import set_relay, set_alert_led
                set_relay(relay_state)
                set_alert_led(alert_state)
            else:
                # Simulation logging
                if relay_state != self.system_state.relay_on:
                    print(f"🔌 RELAY: {'ON' if relay_state else 'OFF'}")
                if alert_state != self.system_state.alert_active:
                    print(f"💡 ALERT LED: {'ON' if alert_state else 'OFF'}")
            
            # Update system state
            self.system_state.relay_on = relay_state
            self.system_state.alert_active = alert_state
            
        except Exception as e:
            logger.error(f"Actuator update error: {e}")
    
    def _calculate_performance_metrics(self, sensor_data: Dict, adaptive_thresholds: Dict) -> Dict[str, float]:
        """Calculate current performance metrics for adaptive learning"""
        
        # Comfort score
        comfort_score = calculate_comfort_score(sensor_data, adaptive_thresholds)
        
        # Energy efficiency (simplified calculation)
        power_consumption = self._estimate_power_consumption(sensor_data)
        runtime = self.system_state.total_runtime_minutes
        efficiency_score = calculate_energy_efficiency(power_consumption, runtime, sensor_data["temperature"])
        
        # System stability
        recent_vibrations = [s.get("vibration", 500) for s in self.sensor_history[-10:]]
        stability_score = calculate_stability_score(recent_vibrations, self.system_state.relay_changes_count)
        
        return {
            "comfort_score": comfort_score,
            "energy_efficiency": efficiency_score,
            "stability_score": stability_score,
            "power_consumption": power_consumption
        }
    
    def _estimate_power_consumption(self, sensor_data: Dict) -> float:
        """Estimate current power consumption"""
        base_power = 50  # Base system power
        cooling_power = 200 if self.system_state.relay_on else 0
        
        # Temperature-dependent power factor
        temp_factor = max(0, (sensor_data["temperature"] - 20) * 5)
        
        return base_power + cooling_power + temp_factor
    
    def _update_learning_data(self, sensor_data: Dict, performance_metrics: Dict):
        """Update data for continuous learning"""
        # Update sensor history
        self.adaptive_engine.update_sensor_history(sensor_data)
        self.sensor_history.append(sensor_data)
        
        # Update performance feedback
        self.adaptive_engine.update_performance_feedback(
            performance_metrics["comfort_score"],
            performance_metrics["energy_efficiency"],
            performance_metrics["stability_score"]
        )
        
        # Keep history manageable
        if len(self.sensor_history) > 100:
            self.sensor_history = self.sensor_history[-50:]
    
    def process_single_cycle(self) -> bool:
        """Process one complete control cycle"""
        cycle_start = time.perf_counter()
        
        try:
            self.cycle_count += 1
            
            # 1. Read sensors
            sensor_data = self._read_sensors()
            if not sensor_data:
                return True  # Continue despite sensor error
            
            # 2. Get adaptive thresholds
            current_time = time.time()
            if current_time - self.last_adaptation_time > 60:  # Adapt every minute
                performance_metrics = self._calculate_performance_metrics(sensor_data, 
                    self.adaptive_engine.get_current_thresholds())
                
                # Trigger threshold adaptation
                self.adaptive_engine.adapt_thresholds(sensor_data, performance_metrics)
                self.last_adaptation_time = current_time
            
            adaptive_thresholds = self.adaptive_engine.get_current_thresholds()
            
            # 3. Safety checks (highest priority)
            safety_override = self._safety_check(sensor_data)
            
            if safety_override is not None:
                # Safety system forces specific state
                relay_decision = safety_override
                alert_decision = True  # Always alert during safety override
                confidence = 1.0
                decision_source = "SAFETY"
            else:
                # 4. AI/Rule-based decision
                relay_decision, confidence = self._ai_decision(sensor_data, adaptive_thresholds)
                
                # 5. Alert logic
                alert_decision = (
                    confidence < adaptive_thresholds.get("confidence_threshold", 0.7) or
                    sensor_data["vibration"] > adaptive_thresholds.get("vibration_warning", 700) or
                    self.system_state.relay_changes_count > 3
                )
                
                decision_source = "EMBEDDED_AI" if self.embedded_ai_available else ("PYTORCH_AI" if self.pytorch_available else "RULES")
            
            # 6. Update actuators
            self._update_actuators(relay_decision, alert_decision)
            
            # 7. Calculate and update performance metrics
            performance_metrics = self._calculate_performance_metrics(sensor_data, adaptive_thresholds)
            self._update_learning_data(sensor_data, performance_metrics)
            
            # 8. Update runtime tracking
            if self.system_state.relay_on:
                self.system_state.total_runtime_minutes += 1/60  # Assume 1-second cycles
            
            # 9. Performance logging
            cycle_time = (time.perf_counter() - cycle_start) * 1000
            self.total_processing_time += cycle_time
            
            if self.cycle_count % 30 == 0:  # Log every 30 cycles
                self._log_system_status(sensor_data, adaptive_thresholds, performance_metrics, 
                                      decision_source, confidence, cycle_time)
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Cycle interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Cycle processing error: {e}")
            logger.error(traceback.format_exc())
            return True  # Continue despite error
    
    def _log_system_status(self, sensor_data: Dict, thresholds: Dict, performance: Dict,
                          decision_source: str, confidence: float, cycle_time: float):
        """Log comprehensive system status"""
        print(f"\n📊 System Status (Cycle {self.cycle_count}):")
        print(f"   🌡️  Temp: {sensor_data['temperature']:.1f}°C (threshold: {thresholds['temp_warning']:.1f}°C)")
        print(f"   💨 Humidity: {sensor_data['humidity']:.1f}%")
        print(f"   📳 Vibration: {sensor_data['vibration']} (threshold: {thresholds['vibration_warning']})")
        print(f"   🔌 Relay: {'ON' if self.system_state.relay_on else 'OFF'} ({decision_source}, conf: {confidence:.2f})")
        print(f"   ⚡ Power: {performance['power_consumption']:.0f}W")
        print(f"   🎯 Comfort: {performance['comfort_score']:.2f}, Efficiency: {performance['energy_efficiency']:.2f}")
        print(f"   🕒 Season: {thresholds['season']}, {thresholds['time_of_day']}")
        print(f"   ⏱️  Cycle time: {cycle_time:.2f}ms")
        print(f"   🔄 Relay changes: {self.system_state.relay_changes_count}")
        
    def run_continuous_control(self, duration_seconds: Optional[float] = None):
        """Run continuous intelligent control loop"""
        logger.info("🚀 Starting Integrated Edge AI Control System")
        logger.info(f"   Adaptive thresholds: {self.adaptive_engine.get_current_thresholds()}")
        
        start_time = time.time()
        next_cycle_time = start_time
        cycle_interval = 1.0  # 1 second per cycle
        
        try:
            while True:
                current_time = time.time()
                
                # Check duration limit
                if duration_seconds and (current_time - start_time) >= duration_seconds:
                    logger.info(f"⏰ Duration limit reached ({duration_seconds}s)")
                    break
                
                # Wait for next cycle time
                if current_time < next_cycle_time:
                    time.sleep(next_cycle_time - current_time)
                
                # Process one cycle
                if not self.process_single_cycle():
                    break
                
                # Schedule next cycle
                next_cycle_time += cycle_interval
                
        except KeyboardInterrupt:
            logger.info("🛑 Control system stopped by user")
        except Exception as e:
            logger.error(f"Control system error: {e}")
            logger.error(traceback.format_exc())
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup and save state"""
        try:
            # Set actuators to safe state
            self._update_actuators(False, False)
            
            # Save adaptive threshold history
            self.adaptive_engine.save_threshold_history()
            
            # Print final statistics
            self._print_final_statistics()
            
            logger.info("🔒 System safely shutdown")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def _print_final_statistics(self):
        """Print comprehensive final statistics"""
        if self.cycle_count == 0:
            return
            
        avg_cycle_time = self.total_processing_time / self.cycle_count
        uptime = time.time() - (self.cycle_count * 1.0)  # Approximate
        
        print(f"\n📈 Final System Statistics:")
        print(f"   Total cycles: {self.cycle_count}")
        print(f"   Average cycle time: {avg_cycle_time:.2f}ms")
        print(f"   System uptime: {uptime/60:.1f} minutes")
        print(f"   Relay runtime: {self.system_state.total_runtime_minutes:.1f} minutes")
        print(f"   Relay changes: {self.system_state.relay_changes_count}")
        print(f"   Safety overrides: {self.system_state.safety_overrides}")
        print(f"   AI decisions: {'Embedded AI' if self.embedded_ai_available else ('PyTorch AI' if self.pytorch_available else 'Rule-based only')}")
        
        # Adaptation statistics
        adaptation_stats = self.adaptive_engine.get_adaptation_stats()
        print(f"   Threshold adaptations: {adaptation_stats['total_adaptations']}")
        print(f"   Average comfort score: {adaptation_stats['avg_comfort_score']:.2f}")
        print(f"   Average efficiency score: {adaptation_stats['avg_efficiency_score']:.2f}")
        print(f"   Average stability score: {adaptation_stats['avg_stability_score']:.2f}")

# Command line interface
def main():
    """Main function with command line options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Edge AI IoT Controller')
    parser.add_argument('--duration', type=float, help='Run duration in seconds')
    parser.add_argument('--model-path', default='../model/relay_model.pth', help='Path to AI model')
    parser.add_argument('--scaler-path', default='../model/scaler.pkl', help='Path to feature scaler')
    parser.add_argument('--config-path', default='config.json', help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run controller
    controller = IntegratedEdgeAIController(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        config_path=args.config_path
    )
    
    controller.run_continuous_control(args.duration)

if __name__ == "__main__":
    main()
