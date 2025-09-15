# 🚀 Quick Start Guide

## Project Organization

Your project is now perfectly organized with clean separation of concerns:

```
📁 Root Directory
├── iot/          - ESP32 production code
├── model/        - AI training pipeline  
├── tests/        - Unit testing framework
├── scripts/      - Utility & deployment tools
├── data/         - Training datasets & Edge Impulse files
├── docs/         - Documentation
└── config/       - Project configuration
```

## ⚡ Quick Commands

### 🧠 Train AI Model
```bash
python model/main.py
```

### 🧪 Run Tests
```bash
python -m pytest tests/ -v
```

### 🚀 Deploy to ESP32
```bash
python scripts/deploy_esp32.py
```

### 🎮 Wokwi Simulation
```bash
python scripts/wokwi_manager.py
```

### 📊 Generate Edge Impulse Data
```bash
python data/edge_impulse/edge_impulse_csv_generator.py
```

## 📁 Key Files for Edge Impulse

Upload these to https://studio.edgeimpulse.com/:

- **Training:** `data/edge_impulse/iot_training_data.csv`
- **Testing:** `data/edge_impulse/iot_test_data.csv`

## 🔧 Installation

```bash
pip install -r config/requirements.txt
```

## 📖 Full Documentation

- **[Complete Guide](docs/README.md)** - Detailed technical specs
- **[Deployment Guide](docs/ESP32_DEPLOYMENT_GUIDE.md)** - ESP32 setup

---
✨ **Your project is now perfectly organized and production-ready!**