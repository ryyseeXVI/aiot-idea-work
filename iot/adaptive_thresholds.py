"""
Adaptive Threshold Engine for IoT Edge AI
Automatically adjusts system parameters based on environmental patterns, seasons, and usage
"""

import json
import time
import math
import statistics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

@dataclass
class ThresholdProfile:
    """Adaptive threshold profile for different conditions"""
    temp_warning: float
    temp_comfort: float
    vibration_warning: int
    vibration_normal: int
    energy_optimization_temp: float
    cooling_aggressiveness: float  # 0.0 to 1.0
    power_efficiency_mode: bool
    confidence_threshold: float

@dataclass
class EnvironmentalContext:
    """Current environmental and temporal context"""
    season: str  # spring, summer, fall, winter
    time_of_day: str  # morning, afternoon, evening, night
    hour: int
    day_of_week: int  # 0=Monday
    is_weekend: bool
    is_off_peak: bool
    outdoor_temp_estimate: float
    recent_weather_pattern: str  # hot, mild, cold, variable

class AdaptiveThresholdEngine:
    """
    Intelligent threshold management system that learns and adapts
    """
    
    def __init__(self, config_path: str = "config.json", history_path: str = "threshold_history.json"):
        self.config_path = config_path
        self.history_path = history_path
        
        # Load base configuration
        self.base_config = self._load_config()
        
        # Historical data for learning
        self.sensor_history: List[Dict] = []
        self.performance_history: List[Dict] = []
        self.threshold_history: List[Dict] = []
        
        # Current adaptive state
        self.current_profile = self._get_default_profile()
        self.learning_rate = 0.05
        self.adaptation_interval = 300  # 5 minutes
        self.last_adaptation = 0
        
        # Seasonal profiles
        self.seasonal_profiles = self._initialize_seasonal_profiles()
        
        # Performance tracking
        self.comfort_score_history = []
        self.energy_efficiency_history = []
        self.system_stability_history = []
        
        # Load historical data
        self._load_threshold_history()
        
        print("🧠 Adaptive Threshold Engine initialized")
        print(f"   Base temp warning: {self.current_profile.temp_warning}°C")
        print(f"   Current season: {self._get_season()}")
        
    def _load_config(self) -> Dict:
        """Load base configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration if file doesn't exist"""
        return {
            "thresholds": {
                "temp_warning": 28.0,
                "vibration_warning": 700,
                "energy_optimization_temp": 25.0
            }
        }
    
    def _get_default_profile(self) -> ThresholdProfile:
        """Get default threshold profile"""
        base = self.base_config.get("thresholds", {})
        return ThresholdProfile(
            temp_warning=base.get("temp_warning", 28.0),
            temp_comfort=base.get("temp_comfort", 24.0),
            vibration_warning=base.get("vibration_warning", 700),
            vibration_normal=400,
            energy_optimization_temp=base.get("energy_optimization_temp", 25.0),
            cooling_aggressiveness=0.5,
            power_efficiency_mode=False,
            confidence_threshold=0.7
        )
    
    def _initialize_seasonal_profiles(self) -> Dict[str, ThresholdProfile]:
        """Initialize season-specific threshold profiles"""
        base = self._get_default_profile()
        
        return {
            "winter": ThresholdProfile(
                temp_warning=base.temp_warning - 2.0,  # 26°C
                temp_comfort=base.temp_comfort - 1.0,   # 23°C
                vibration_warning=base.vibration_warning + 100,  # More tolerance
                vibration_normal=base.vibration_normal,
                energy_optimization_temp=base.energy_optimization_temp - 2.0,
                cooling_aggressiveness=0.3,  # Less aggressive
                power_efficiency_mode=True,  # Save energy in winter
                confidence_threshold=0.6     # More flexible
            ),
            "spring": ThresholdProfile(
                temp_warning=base.temp_warning,
                temp_comfort=base.temp_comfort,
                vibration_warning=base.vibration_warning,
                vibration_normal=base.vibration_normal,
                energy_optimization_temp=base.energy_optimization_temp,
                cooling_aggressiveness=0.5,
                power_efficiency_mode=False,
                confidence_threshold=0.7
            ),
            "summer": ThresholdProfile(
                temp_warning=base.temp_warning + 2.0,  # 30°C
                temp_comfort=base.temp_comfort + 1.0,   # 25°C
                vibration_warning=base.vibration_warning - 50,  # Less tolerance
                vibration_normal=base.vibration_normal,
                energy_optimization_temp=base.energy_optimization_temp + 1.0,
                cooling_aggressiveness=0.8,  # More aggressive
                power_efficiency_mode=False,
                confidence_threshold=0.8     # Higher confidence needed
            ),
            "fall": ThresholdProfile(
                temp_warning=base.temp_warning - 1.0,  # 27°C
                temp_comfort=base.temp_comfort,
                vibration_warning=base.vibration_warning + 50,
                vibration_normal=base.vibration_normal,
                energy_optimization_temp=base.energy_optimization_temp - 1.0,
                cooling_aggressiveness=0.4,
                power_efficiency_mode=True,
                confidence_threshold=0.65
            )
        }
    
    def _get_environmental_context(self) -> EnvironmentalContext:
        """Analyze current environmental and temporal context"""
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        
        # Determine season
        month = now.month
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:
            season = "fall"
        
        # Time of day
        if 6 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 18:
            time_of_day = "afternoon"
        elif 18 <= hour < 22:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        # Estimate outdoor temperature based on season and time
        base_temp = {"winter": 5, "spring": 15, "summer": 25, "fall": 15}[season]
        daily_variation = 5 * math.sin((hour - 6) * math.pi / 12)  # Peak at 2 PM
        outdoor_temp = base_temp + daily_variation
        
        # Weather pattern estimation from recent sensor data
        weather_pattern = self._estimate_weather_pattern()
        
        return EnvironmentalContext(
            season=season,
            time_of_day=time_of_day,
            hour=hour,
            day_of_week=day_of_week,
            is_weekend=day_of_week >= 5,
            is_off_peak=self._is_off_peak_time(hour),
            outdoor_temp_estimate=outdoor_temp,
            recent_weather_pattern=weather_pattern
        )
    
    def _get_season(self) -> str:
        """Get current season"""
        return self._get_environmental_context().season
    
    def _is_off_peak_time(self, hour: int) -> bool:
        """Check if current time is off-peak"""
        energy_config = self.base_config.get("energy", {})
        start_hour = energy_config.get("off_peak_start_hour", 22)
        end_hour = energy_config.get("off_peak_end_hour", 6)
        
        if start_hour > end_hour:  # Crosses midnight
            return hour >= start_hour or hour < end_hour
        else:
            return start_hour <= hour < end_hour
    
    def _estimate_weather_pattern(self) -> str:
        """Estimate recent weather pattern from sensor history"""
        if len(self.sensor_history) < 10:
            return "mild"
        
        recent_temps = [s.get("temperature", 25) for s in self.sensor_history[-20:]]
        avg_temp = statistics.mean(recent_temps)
        temp_variance = statistics.variance(recent_temps) if len(recent_temps) > 1 else 0
        
        if avg_temp > 28:
            return "hot"
        elif avg_temp < 20:
            return "cold"
        elif temp_variance > 4:
            return "variable"
        else:
            return "mild"
    
    def adapt_thresholds(self, sensor_data: Dict, system_performance: Dict) -> ThresholdProfile:
        """
        Main adaptation function - adjusts thresholds based on context and performance
        """
        current_time = time.time()
        
        # Only adapt periodically to avoid oscillation
        if current_time - self.last_adaptation < self.adaptation_interval:
            return self.current_profile
        
        # Get environmental context
        context = self._get_environmental_context()
        
        # Start with seasonal base profile
        adapted_profile = self._copy_profile(self.seasonal_profiles[context.season])
        
        # Apply time-of-day adjustments
        adapted_profile = self._apply_time_adjustments(adapted_profile, context)
        
        # Apply performance-based learning
        adapted_profile = self._apply_performance_learning(adapted_profile, system_performance)
        
        # Apply environmental adaptation
        adapted_profile = self._apply_environmental_adaptation(adapted_profile, sensor_data, context)
        
        # Apply energy optimization
        adapted_profile = self._apply_energy_optimization(adapted_profile, context)
        
        # Validate and constrain thresholds
        adapted_profile = self._validate_thresholds(adapted_profile)
        
        # Update current profile
        old_profile = self.current_profile
        self.current_profile = adapted_profile
        self.last_adaptation = current_time
        
        # Log adaptation if significant change
        if self._profile_changed_significantly(old_profile, adapted_profile):
            self._log_adaptation(old_profile, adapted_profile, context)
        
        # Store in history
        self.threshold_history.append({
            "timestamp": current_time,
            "profile": asdict(adapted_profile),
            "context": asdict(context),
            "reason": "periodic_adaptation"
        })
        
        return adapted_profile
    
    def _copy_profile(self, profile: ThresholdProfile) -> ThresholdProfile:
        """Create a copy of threshold profile"""
        return ThresholdProfile(**asdict(profile))
    
    def _apply_time_adjustments(self, profile: ThresholdProfile, context: EnvironmentalContext) -> ThresholdProfile:
        """Apply time-of-day specific adjustments"""
        
        # Night time - more conservative
        if context.time_of_day == "night":
            profile.temp_warning -= 1.0
            profile.cooling_aggressiveness *= 0.7
            profile.power_efficiency_mode = True
            profile.confidence_threshold -= 0.1
        
        # Morning - gradual warmup
        elif context.time_of_day == "morning":
            profile.temp_comfort += 0.5
            profile.cooling_aggressiveness *= 0.8
        
        # Afternoon - peak performance
        elif context.time_of_day == "afternoon":
            profile.cooling_aggressiveness *= 1.1
            profile.confidence_threshold += 0.05
        
        # Weekend adjustments
        if context.is_weekend:
            profile.temp_comfort += 1.0  # More relaxed comfort
            profile.power_efficiency_mode = True
        
        # Off-peak energy optimization
        if context.is_off_peak:
            profile.energy_optimization_temp -= 1.0
            profile.power_efficiency_mode = True
        
        return profile
    
    def _apply_performance_learning(self, profile: ThresholdProfile, performance: Dict) -> ThresholdProfile:
        """Adapt based on recent system performance"""
        
        # Learn from comfort feedback
        comfort_score = performance.get("comfort_score", 0.7)
        if comfort_score < 0.6:  # Poor comfort
            profile.temp_warning -= 0.5
            profile.cooling_aggressiveness += 0.1
        elif comfort_score > 0.9:  # Excellent comfort
            profile.temp_warning += 0.2
            profile.cooling_aggressiveness -= 0.05
        
        # Learn from energy efficiency
        efficiency_score = performance.get("energy_efficiency", 0.7)
        if efficiency_score < 0.5:  # Poor efficiency
            profile.energy_optimization_temp += 0.5
            profile.power_efficiency_mode = True
            profile.cooling_aggressiveness -= 0.1
        
        # Learn from system stability
        stability_score = performance.get("stability_score", 0.8)
        if stability_score < 0.7:  # Unstable
            profile.confidence_threshold += 0.05
            profile.vibration_warning -= 50
        
        return profile
    
    def _apply_environmental_adaptation(self, profile: ThresholdProfile, 
                                      sensor_data: Dict, context: EnvironmentalContext) -> ThresholdProfile:
        """Adapt to current environmental conditions"""
        
        current_temp = sensor_data.get("temperature", 25)
        current_humidity = sensor_data.get("humidity", 60)
        current_vibration = sensor_data.get("vibration", 500)
        
        # Temperature adaptation
        if current_temp > profile.temp_warning:
            # System is running hot, be more aggressive
            profile.cooling_aggressiveness += 0.1
            profile.temp_warning += 0.5  # Raise threshold slightly to avoid oscillation
        
        # Humidity adaptation (affects perceived temperature)
        if current_humidity > 80:
            profile.temp_comfort -= 1.0  # Feel warmer at high humidity
            profile.cooling_aggressiveness += 0.05
        elif current_humidity < 40:
            profile.temp_comfort += 0.5  # Feel cooler at low humidity
        
        # Vibration pattern adaptation
        recent_vibrations = [s.get("vibration", 500) for s in self.sensor_history[-10:]]
        if recent_vibrations:
            avg_vibration = statistics.mean(recent_vibrations)
            if avg_vibration > profile.vibration_normal * 1.5:
                profile.vibration_warning += 50  # Adapt to noisy environment
            elif avg_vibration < profile.vibration_normal * 0.8:
                profile.vibration_warning -= 20  # More sensitive in quiet environment
        
        # Weather pattern adaptation
        if context.recent_weather_pattern == "hot":
            profile.temp_warning += 1.0
            profile.cooling_aggressiveness += 0.1
        elif context.recent_weather_pattern == "cold":
            profile.temp_warning -= 1.0
            profile.power_efficiency_mode = True
        elif context.recent_weather_pattern == "variable":
            profile.confidence_threshold -= 0.05  # More flexible
        
        return profile
    
    def _apply_energy_optimization(self, profile: ThresholdProfile, context: EnvironmentalContext) -> ThresholdProfile:
        """Apply energy optimization strategies"""
        
        # Peak hours - energy conservation
        if not context.is_off_peak:
            profile.energy_optimization_temp += 1.0
            profile.cooling_aggressiveness *= 0.8
            profile.power_efficiency_mode = True
        
        # Summer peak demand management
        if context.season == "summer" and context.time_of_day == "afternoon":
            profile.energy_optimization_temp += 1.5
            profile.cooling_aggressiveness *= 0.7
        
        # Winter heating efficiency
        if context.season == "winter":
            profile.power_efficiency_mode = True
            profile.cooling_aggressiveness *= 0.5
        
        return profile
    
    def _validate_thresholds(self, profile: ThresholdProfile) -> ThresholdProfile:
        """Ensure thresholds are within safe and reasonable bounds"""
        
        # Temperature bounds
        profile.temp_warning = max(20.0, min(40.0, profile.temp_warning))
        profile.temp_comfort = max(18.0, min(30.0, profile.temp_comfort))
        profile.energy_optimization_temp = max(18.0, min(35.0, profile.energy_optimization_temp))
        
        # Ensure logical ordering
        if profile.temp_comfort > profile.temp_warning:
            profile.temp_comfort = profile.temp_warning - 1.0
        
        # Vibration bounds
        profile.vibration_warning = max(300, min(1200, profile.vibration_warning))
        profile.vibration_normal = max(200, min(800, profile.vibration_normal))
        
        # Cooling aggressiveness bounds
        profile.cooling_aggressiveness = max(0.1, min(1.0, profile.cooling_aggressiveness))
        
        # Confidence bounds
        profile.confidence_threshold = max(0.3, min(0.9, profile.confidence_threshold))
        
        return profile
    
    def _profile_changed_significantly(self, old: ThresholdProfile, new: ThresholdProfile) -> bool:
        """Check if profile changed significantly enough to log"""
        temp_change = abs(old.temp_warning - new.temp_warning)
        vibration_change = abs(old.vibration_warning - new.vibration_warning)
        aggressiveness_change = abs(old.cooling_aggressiveness - new.cooling_aggressiveness)
        
        return (temp_change > 0.5 or 
                vibration_change > 50 or 
                aggressiveness_change > 0.1)
    
    def _log_adaptation(self, old: ThresholdProfile, new: ThresholdProfile, context: EnvironmentalContext):
        """Log significant threshold adaptations"""
        print(f"🔄 Threshold Adaptation ({context.season}, {context.time_of_day}):")
        print(f"   Temp warning: {old.temp_warning:.1f}°C → {new.temp_warning:.1f}°C")
        print(f"   Vibration warning: {old.vibration_warning} → {new.vibration_warning}")
        print(f"   Cooling aggressiveness: {old.cooling_aggressiveness:.2f} → {new.cooling_aggressiveness:.2f}")
        print(f"   Power efficiency: {old.power_efficiency_mode} → {new.power_efficiency_mode}")
    
    def update_sensor_history(self, sensor_data: Dict):
        """Update sensor history for learning"""
        sensor_data["timestamp"] = time.time()
        self.sensor_history.append(sensor_data)
        
        # Keep history manageable
        if len(self.sensor_history) > 1000:
            self.sensor_history = self.sensor_history[-500:]
    
    def update_performance_feedback(self, comfort_score: float, energy_efficiency: float, stability_score: float):
        """Update performance feedback for learning"""
        feedback = {
            "timestamp": time.time(),
            "comfort_score": comfort_score,
            "energy_efficiency": energy_efficiency,
            "stability_score": stability_score
        }
        self.performance_history.append(feedback)
        
        # Store for trend analysis
        self.comfort_score_history.append(comfort_score)
        self.energy_efficiency_history.append(energy_efficiency)
        self.system_stability_history.append(stability_score)
        
        # Keep manageable
        for history in [self.comfort_score_history, self.energy_efficiency_history, self.system_stability_history]:
            if len(history) > 100:
                history[:] = history[-50:]
    
    def get_current_thresholds(self) -> Dict[str, Any]:
        """Get current threshold values for the system"""
        profile = self.current_profile
        context = self._get_environmental_context()
        
        return {
            "temp_warning": profile.temp_warning,
            "temp_comfort": profile.temp_comfort,
            "vibration_warning": profile.vibration_warning,
            "vibration_normal": profile.vibration_normal,
            "energy_optimization_temp": profile.energy_optimization_temp,
            "cooling_aggressiveness": profile.cooling_aggressiveness,
            "power_efficiency_mode": profile.power_efficiency_mode,
            "confidence_threshold": profile.confidence_threshold,
            "season": context.season,
            "time_of_day": context.time_of_day,
            "is_off_peak": context.is_off_peak
        }
    
    def _load_threshold_history(self):
        """Load historical threshold data"""
        try:
            if Path(self.history_path).exists():
                with open(self.history_path, 'r') as f:
                    data = json.load(f)
                    self.threshold_history = data.get("threshold_history", [])
                    self.performance_history = data.get("performance_history", [])
                    # Restore recent learning state
                    recent_performance = self.performance_history[-20:] if self.performance_history else []
                    for perf in recent_performance:
                        self.comfort_score_history.append(perf.get("comfort_score", 0.7))
                        self.energy_efficiency_history.append(perf.get("energy_efficiency", 0.7))
                        self.system_stability_history.append(perf.get("stability_score", 0.8))
        except Exception as e:
            print(f"Warning: Could not load threshold history: {e}")
    
    def save_threshold_history(self):
        """Save threshold history to disk"""
        try:
            data = {
                "threshold_history": self.threshold_history[-100:],  # Keep last 100
                "performance_history": self.performance_history[-100:],
                "last_save": time.time()
            }
            with open(self.history_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save threshold history: {e}")
    
    def get_adaptation_stats(self) -> Dict:
        """Get statistics about threshold adaptations"""
        if not self.threshold_history:
            return {"total_adaptations": 0}
        
        recent_adaptations = [a for a in self.threshold_history 
                            if time.time() - a["timestamp"] < 86400]  # Last 24 hours
        
        seasons_adapted = set(a["context"]["season"] for a in recent_adaptations)
        
        return {
            "total_adaptations": len(self.threshold_history),
            "recent_adaptations_24h": len(recent_adaptations),
            "seasons_adapted_today": list(seasons_adapted),
            "current_season": self._get_season(),
            "avg_comfort_score": statistics.mean(self.comfort_score_history) if self.comfort_score_history else 0.7,
            "avg_efficiency_score": statistics.mean(self.energy_efficiency_history) if self.energy_efficiency_history else 0.7,
            "avg_stability_score": statistics.mean(self.system_stability_history) if self.system_stability_history else 0.8
        }

# Performance feedback calculation helpers
def calculate_comfort_score(sensor_data: Dict, thresholds: Dict) -> float:
    """Calculate comfort score based on how well system maintains comfort"""
    temp = sensor_data.get("temperature", 25)
    target_temp = thresholds.get("temp_comfort", 24)
    temp_diff = abs(temp - target_temp)
    
    # Score from 0 to 1, perfect at target, degrading with distance
    comfort_score = max(0, 1 - (temp_diff / 5.0))
    return comfort_score

def calculate_energy_efficiency(power_consumption: float, relay_runtime: float, temp_achieved: float) -> float:
    """Calculate energy efficiency score"""
    if relay_runtime == 0:
        return 1.0
    
    # Efficiency based on power used vs temperature control achieved
    base_efficiency = 1.0 / (1.0 + power_consumption / 1000.0)  # Penalize high power
    temp_efficiency = max(0.5, 1.0 - abs(temp_achieved - 24.0) / 10.0)  # Reward good temp control
    
    return (base_efficiency + temp_efficiency) / 2.0

def calculate_stability_score(vibration_history: List[float], relay_changes: int) -> float:
    """Calculate system stability score"""
    if not vibration_history:
        return 0.8
    
    # Stability based on vibration variance and relay oscillation
    vibration_variance = statistics.variance(vibration_history) if len(vibration_history) > 1 else 0
    vibration_stability = max(0, 1 - vibration_variance / 10000)  # Normalize variance
    
    # Penalize frequent relay changes (oscillation)
    relay_stability = max(0.3, 1 - relay_changes / 10.0)
    
    return (vibration_stability + relay_stability) / 2.0
