# =============================================================================
# Dockerfile for Autonomous Driving Perception Module
# 支持训练和ROS2推理的Docker镜像
# =============================================================================

# 使用ROS2 Humble作为基础镜像（包含Python 3.10）
FROM ros:humble-ros-base

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV WORKSPACE=/workspace

# 安装系统依赖
# 安装OpenCV所需的系统库和其他工具
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR ${WORKSPACE}

# 复制依赖文件
COPY requirements.txt ${WORKSPACE}/

# 安装Python依赖
# 注意：ROS2环境中的Python包通过pip安装
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . ${WORKSPACE}/

# 创建必要的目录
RUN mkdir -p ${WORKSPACE}/checkpoints \
    ${WORKSPACE}/logs \
    ${WORKSPACE}/outputs \
    ${WORKSPACE}/data

# 设置Python路径
ENV PYTHONPATH="${WORKSPACE}:${PYTHONPATH}"

# 设置ROS2环境
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# 默认命令：显示帮助信息
CMD ["bash", "-c", "echo '=== Autonomous Driving Perception Module ===' && \
    echo '' && \
    echo 'Available commands:' && \
    echo '' && \
    echo '1. Training:' && \
    echo '   python3 -m src.training.train --config configs/config_example.yaml' && \
    echo '' && \
    echo '2. Export ONNX:' && \
    echo '   python3 -m src.inference.export_onnx --config configs/config_example.yaml --checkpoint checkpoints/best_model.pth' && \
    echo '' && \
    echo '3. ONNX Inference:' && \
    echo '   python3 -m src.inference.infer_onnx --config configs/config_example.yaml --image <image_path> --output outputs/' && \
    echo '' && \
    echo '4. ROS2 Node:' && \
    echo '   source /opt/ros/humble/setup.bash && ros2 run src.ros2_integration lane_sign_node' && \
    echo '' && \
    echo '5. Run Tests:' && \
    echo '   pytest tests/ -v' && \
    echo '' && \
    bash"]

# =============================================================================
# 使用示例：
# 
# 构建镜像：
#   docker build -t adpm:latest .
#
# 运行训练（GPU支持）：
#   docker run --gpus all -it -v $(pwd)/data:/workspace/data -v $(pwd)/checkpoints:/workspace/checkpoints adpm:latest \
#       python3 -m src.training.train --config configs/config_example.yaml
#
# 运行ROS2节点：
#   docker run --gpus all -it --network host adpm:latest \
#       bash -c "source /opt/ros/humble/setup.bash && ros2 run src.ros2_integration lane_sign_node"
#
# 交互式shell：
#   docker run --gpus all -it -v $(pwd):/workspace adpm:latest bash
# =============================================================================
