# ESP32 IoT Relay Control with Edge AI

🚀 **Intelligent IoT system with embedded AI for ESP32 microcontroller**

## 📁 Project Structure

```
aiot-idea-work/
├── 📂 iot/                    # ESP32 Production Code
│   ├── main.py               # ESP32 MicroPython controller
│   ├── sensors.py            # Hardware sensor interfaces
│   ├── actuators.py          # Hardware control outputs
│   ├── embedded_ai_model.py  # Ultra-lightweight AI (6.3KB)
│   ├── embedded_ai_converter.py  # PyTorch→ESP32 conversion
│   ├── config_manager.py     # Configuration management
│   ├── energy.py             # Off-peak energy optimization
│   ├── adaptive_thresholds.py # Dynamic threshold engine
│   └── integrated_edge_controller.py # Advanced controller
│
├── 📂 model/                  # AI Training Pipeline
│   └── main.py               # Complete PyTorch training system
│
├── 📂 tests/                  # Comprehensive Testing
│   ├── test_sensors.py       # Sensor testing
│   ├── test_actuators.py     # Actuator testing
│   ├── test_config_manager.py # Config testing
│   └── test_ai_model.py      # AI model testing
│
├── 📂 scripts/               # Utility Scripts
│   ├── deploy_esp32.py       # ESP32 deployment automation
│   └── wokwi_manager.py      # Wokwi simulation utilities
│
├── 📂 data/                  # Data & Training Sets
│   └── edge_impulse/         # Edge Impulse datasets
│       ├── iot_training_data.csv
│       ├── iot_test_data.csv
│       ├── edge_impulse_formatter.py
│       ├── edge_impulse_csv_generator.py
│       └── edge_impulse_samples/
│
├── 📂 docs/                  # Documentation
│   ├── ESP32_DEPLOYMENT_GUIDE.md
│   └── README.md
│
└── 📂 config/                # Configuration Files
    ├── requirements.txt
    └── pyproject.toml
```

## 🎯 Quick Start

### 1. **Install Dependencies**
```bash
pip install -r config/requirements.txt
```

### 2. **Train AI Model**
```bash
python model/main.py
```

### 3. **Run Tests**
```bash
python -m pytest tests/
```

### 4. **Deploy to ESP32**
```bash
python scripts/deploy_esp32.py
```

### 5. **Wokwi Simulation**
```bash
python scripts/wokwi_manager.py
```

## 🧠 Edge Impulse Integration

Ready-to-use datasets for Edge Impulse training:
- **Training Data:** `data/edge_impulse/edge_impulse_training.csv` (1000 samples)
- **Test Data:** `data/edge_impulse/edge_impulse_testing.csv` (200 samples)

Upload to https://studio.edgeimpulse.com/ for cloud-based ML training.

## 🔧 Hardware Configuration

- **ESP32 DevKit** - Main microcontroller
- **DHT22** (GPIO4) - Temperature & humidity sensor
- **Relay Module** (GPIO5) - Control output
- **LED** (GPIO15) - Status indicator
- **Vibration Sensor** (GPIO33) - Motion detection
- **Water Leak Sensor** (GPIO14) - Safety monitoring

## 📊 Features

✅ **Ultra-lightweight AI** (6.3KB on ESP32)  
✅ **Safety-first design** (water leak protection)  
✅ **Energy optimization** (off-peak scheduling)  
✅ **Adaptive thresholds** (seasonal adjustment)  
✅ **Comprehensive testing** (100% core coverage)  
✅ **Wokwi simulation** (test without hardware)  
✅ **Edge Impulse ready** (cloud ML training)  

## 📖 Documentation

- **[Deployment Guide](docs/ESP32_DEPLOYMENT_GUIDE.md)** - Complete ESP32 setup
- **[Detailed README](docs/README.md)** - Technical specifications

## 🚀 Production Ready

This system is optimized for real-world IoT deployment with enterprise-grade reliability and efficiency.

---
**Built with ❤️ for ESP32 IoT applications**