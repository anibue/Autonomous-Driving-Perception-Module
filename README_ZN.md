# 自动驾驶感知模块

一个用于自动驾驶感知的**多任务学习**项目，采用共享编码器架构实现**车道线检测**和**交通标志识别**。

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![ROS2](https://img.shields.io/badge/ROS2-Humble-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 功能特点

- **多任务UNet架构**：共享编码器，配备独立的解码器头用于车道线分割和交通标志分类
- **联合训练**：同时优化两个任务，支持可配置的损失权重
- **ONNX导出**：生产级模型导出，便于部署
- **ROS2集成**：用于自动驾驶应用的实时感知节点
- **容器化支持**：完整的Docker支持，可用于训练和推理
- **全面测试**：基于pytest的测试套件

## 🏗️ 架构概览

```
                    ┌──────────────────┐
                    │     输入图像      │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │    共享编码器     │
                    │  (CNN骨干网络)    │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
     ┌────────▼─────────┐         ┌────────▼─────────┐
     │    分割解码器     │         │    分类头        │
     │    (UNet)        │         │    (全连接)       │
     └────────┬─────────┘         └────────┬─────────┘
              │                             │
     ┌────────▼─────────┐         ┌────────▼─────────┐
     │    车道线掩码     │         │    标志类别       │
     └──────────────────┘         └──────────────────┘
```

## 📋 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA（可选，用于GPU训练）
- ROS2 Humble（用于ROS2集成）

## 🚀 安装步骤

### 方式一：Python虚拟环境

```bash
# 克隆仓库
git clone https://github.com/yourusername/Autonomous-Driving-Perception-Module.git
cd Autonomous-Driving-Perception-Module

# 创建并激活虚拟环境
python -m venv venv
# Windows系统:
venv\Scripts\activate
# Linux/Mac系统:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 方式二：Docker

```bash
# 构建Docker镜像
docker build -t adpm:latest .

# 使用GPU运行
docker run --gpus all -it -v $(pwd):/workspace adpm:latest

# 不使用GPU运行
docker run -it -v $(pwd):/workspace adpm:latest
```

## ⚡ 快速开始

### 1. 准备配置文件

编辑配置文件，设置数据集路径：

```bash
cp configs/config_example.yaml configs/my_config.yaml
# 编辑 configs/my_config.yaml，设置您的数据集路径
```

### 2. 生成测试数据（可选）

无需真实数据即可测试整个流程：

```bash
python scripts/prepare_dummy_data.py
```

### 3. 训练模型

```bash
python -m src.training.train --config configs/config_example.yaml
```

### 4. 导出ONNX模型

```bash
python -m src.inference.export_onnx --config configs/config_example.yaml --checkpoint checkpoints/best_model.pth
```

### 5. 运行ONNX推理

```bash
python -m src.inference.infer_onnx --config configs/config_example.yaml --image path/to/image.jpg --output output/
```

## 🤖 ROS2集成

### 前置条件

- 已安装ROS2 Humble或Foxy
- 已导出ONNX模型

### 运行ROS2节点

```bash
# 初始化ROS2环境
source /opt/ros/humble/setup.bash

# 运行感知节点
ros2 run src.ros2_integration lane_sign_node --ros-args \
    -p onnx_model_path:=outputs/model.onnx \
    -p config_path:=configs/config_example.yaml
```

### 使用Launch文件

```bash
ros2 launch src.ros2_integration lane_sign.launch.py \
    onnx_model_path:=outputs/model.onnx \
    config_path:=configs/config_example.yaml
```

### 话题说明

| 话题 | 类型 | 描述 |
|------|------|------|
| `/camera/image_raw` | `sensor_msgs/Image` | 输入相机图像 |
| `/perception/lane_mask` | `sensor_msgs/Image` | 车道线分割掩码 |
| `/perception/sign_label` | `std_msgs/String` | 识别的交通标志 |

## 🧪 测试

使用pytest运行测试套件：

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_model_forward.py -v

# 运行并生成覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

## 📁 目录结构

```
Autonomous-Driving-Perception-Module/
├── README.md                   # 英文文档
├── README_ZN.md               # 中文文档
├── requirements.txt           # Python依赖
├── Dockerfile                 # 容器配置
├── pyproject.toml            # 项目元数据和工具配置
├── .gitignore                # Git忽略规则
├── configs/
│   └── config_example.yaml   # 示例配置
├── docs/
│   └── technical_report.md   # 技术文档
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py        # 数据集类
│   │   └── transforms.py     # 数据增强
│   ├── models/
│   │   ├── __init__.py
│   │   └── multitask_unet.py # 多任务模型
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py          # 训练脚本
│   │   ├── losses.py         # 损失函数
│   │   ├── metrics.py        # 评估指标
│   │   └── utils.py          # 训练工具
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── export_onnx.py    # ONNX导出
│   │   └── infer_onnx.py     # ONNX推理
│   └── ros2_integration/
│       ├── __init__.py
│       ├── lane_sign_node.py # ROS2节点
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

## 📊 支持的数据集

本项目设计为数据集无关，以下是兼容的示例数据集：

- **车道线检测**：TuSimple、CULane、BDD100K
- **交通标志识别**：GTSRB、TT100K

详见配置文件了解数据集路径设置。

### 数据集准备说明

#### 车道线检测数据集（以TuSimple为例）

1. 下载TuSimple数据集
2. 组织目录结构：
   ```
   data/
   ├── lane/
   │   ├── images/        # 原始图像
   │   └── masks/         # 分割掩码
   ```
3. 在配置文件中设置相应路径

#### 交通标志数据集（以GTSRB为例）

1. 下载GTSRB数据集
2. 组织目录结构：
   ```
   data/
   ├── sign/
   │   ├── images/        # 标志图像
   │   └── labels.csv     # 标签文件
   ```
3. 在配置文件中设置相应路径

## 📄 许可证

本项目为开源项目。请根据您的需求选择合适的许可证（MIT、Apache 2.0等）。

## 🤝 贡献指南

欢迎贡献！请按以下步骤操作：

1. Fork本仓库
2. 创建功能分支（`git checkout -b feature/amazing-feature`）
3. 提交更改（`git commit -m 'Add amazing feature'`）
4. 推送到分支（`git push origin feature/amazing-feature`）
5. 创建Pull Request

### 代码规范

- 遵循PEP8规范
- 使用简体中文编写注释
- 在函数签名中包含类型提示
- 为新功能添加测试

## 📬 联系方式

如有问题和需要支持，请在GitHub上提交Issue。

---

**⭐ 如果这个项目对您有帮助，请给个Star！**
