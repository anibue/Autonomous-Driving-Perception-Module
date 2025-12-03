[简体中文](README_ZN.md) | English

# Autonomous Driving Perception Module

A **multi-task learning** project for autonomous driving perception, featuring **lane line detection** and **traffic sign recognition** using a shared encoder architecture.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![ROS2](https://img.shields.io/badge/ROS2-Humble-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Multi-Task UNet Architecture**: Shared encoder with separate decoder heads for lane segmentation and traffic sign classification
- **Joint Training**: Simultaneous optimization of both tasks with configurable loss weighting
- **ONNX Export**: Production-ready model export for deployment
- **ROS2 Integration**: Real-time perception node for autonomous driving applications
- **Containerized**: Full Docker support for training and inference
- **Comprehensive Testing**: pytest-based test suite

## 🏗️ Architecture Overview

```
                    ┌──────────────────┐
                    │   Input Image    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Shared Encoder  │
                    │  (CNN Backbone)  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
     ┌────────▼─────────┐         ┌────────▼─────────┐
     │ Segmentation     │         │ Classification   │
     │ Decoder (UNet)   │         │ Head (FC)        │
     └────────┬─────────┘         └────────┬─────────┘
              │                             │
     ┌────────▼─────────┐         ┌────────▼─────────┐
     │  Lane Mask       │         │  Sign Class      │
     └──────────────────┘         └──────────────────┘
```

## 📋 Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU training)
- ROS2 Humble (for ROS2 integration)

## 🚀 Installation

### Option 1: Python Virtual Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/Autonomous-Driving-Perception-Module.git
cd Autonomous-Driving-Perception-Module

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker

```bash
# Build the Docker image
docker build -t adpm:latest .

# Run with GPU support
docker run --gpus all -it -v $(pwd):/workspace adpm:latest

# Run without GPU
docker run -it -v $(pwd):/workspace adpm:latest
```

## ⚡ Quickstart

### 1. Prepare Configuration

Edit the configuration file to set your dataset paths:

```bash
cp configs/config_example.yaml configs/my_config.yaml
# Edit configs/my_config.yaml with your dataset paths
```

### 2. Generate Dummy Data (Optional)

For testing the pipeline without real data:

```bash
python scripts/prepare_dummy_data.py
```

### 3. Train the Model

```bash
python -m src.training.train --config configs/config_example.yaml
```

### 4. Export to ONNX

```bash
python -m src.inference.export_onnx --config configs/config_example.yaml --checkpoint checkpoints/best_model.pth
```

### 5. Run ONNX Inference

```bash
python -m src.inference.infer_onnx --config configs/config_example.yaml --image path/to/image.jpg --output output/
```

## 🤖 ROS2 Integration

### Prerequisites

- ROS2 Humble or Foxy installed
- ONNX model exported

### Running the ROS2 Node

```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Run the perception node
ros2 run src.ros2_integration lane_sign_node --ros-args \
    -p onnx_model_path:=outputs/model.onnx \
    -p config_path:=configs/config_example.yaml
```

### Using Launch File

```bash
ros2 launch src.ros2_integration lane_sign.launch.py \
    onnx_model_path:=outputs/model.onnx \
    config_path:=configs/config_example.yaml
```

### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Input camera image |
| `/perception/lane_mask` | `sensor_msgs/Image` | Lane segmentation mask |
| `/perception/sign_label` | `std_msgs/String` | Recognized traffic sign |

## 🧪 Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model_forward.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📁 Repository Structure

```
Autonomous-Driving-Perception-Module/
├── README.md                   # English documentation
├── README_ZN.md               # Chinese documentation
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── pyproject.toml            # Project metadata and tooling
├── .gitignore                # Git ignore rules
├── configs/
│   └── config_example.yaml   # Example configuration
├── docs/
│   └── technical_report.md   # Technical documentation
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py        # Dataset classes
│   │   └── transforms.py     # Data augmentation
│   ├── models/
│   │   ├── __init__.py
│   │   └── multitask_unet.py # Multi-task model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py          # Training script
│   │   ├── losses.py         # Loss functions
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── utils.py          # Training utilities
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── export_onnx.py    # ONNX export
│   │   └── infer_onnx.py     # ONNX inference
│   └── ros2_integration/
│       ├── __init__.py
│       ├── lane_sign_node.py # ROS2 node
│       └── launch/
│           └── lane_sign.launch.py
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_model_forward.py
│   ├── test_onnx_export.py
│   └── test_ros2_node_import.py
└── scripts/
    ├── prepare_dummy_data.py
    ├── run_training_example.sh
    └── run_training_example.bat
```

## 📊 Supported Datasets

This project is designed to be dataset-agnostic. Example compatible datasets:

- **Lane Detection**: TuSimple, CULane, BDD100K
- **Traffic Sign Recognition**: GTSRB, TT100K

See the configuration file for dataset path setup.

## 📄 License

This project is open-source. Please choose an appropriate license (MIT, Apache 2.0, etc.) based on your needs.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP8 guidelines
- Write comments in Simplified Chinese
- Include type hints in function signatures
- Add tests for new features

## 📬 Contact

For questions and support, please open an issue on GitHub.

---

**⭐ Star this repository if you find it helpful!**
