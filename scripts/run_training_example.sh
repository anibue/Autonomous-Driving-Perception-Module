#!/bin/bash

# Training Example Script for Linux/Mac
# Linux/Mac下的训练示例脚本

set -e  # 遇到错误立即退出

echo "=================================="
echo "自动驾驶感知模块 - 训练示例"
echo "=================================="

# 1. 检查数据集
echo ""
echo "[1/4] 检查数据集..."
if [ ! -d "./data/lane" ] || [ ! -d "./data/sign" ]; then
    echo "数据集不存在，正在生成虚拟数据集..."
    python scripts/prepare_dummy_data.py \
        --output_dir ./data \
        --num_lane_samples 100 \
        --num_sign_samples 200
else
    echo "✓ 数据集已存在"
fi

# 2. 运行训练
echo ""
echo "[2/4] 开始训练..."
python src/training/train.py \
    --config configs/config_example.yaml \
    --epochs 5 \
    --batch_size 4

# 3. 导出ONNX模型
echo ""
echo "[3/4] 导出ONNX模型..."
python src/inference/export_onnx.py \
    --config configs/config_example.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --output outputs/models/model.onnx

# 4. 运行ONNX推理
echo ""
echo "[4/4] 运行ONNX推理测试..."
if [ -f "./data/lane/images/lane_0000.jpg" ]; then
    python src/inference/infer_onnx.py \
        --config configs/config_example.yaml \
        --model_path outputs/models/model.onnx \
        --image_path ./data/lane/images/lane_0000.jpg \
        --output_dir ./outputs/inference_results
    echo "✓ 推理结果已保存到 ./outputs/inference_results"
else
    echo "⊗ 测试图像不存在，跳过推理测试"
fi

echo ""
echo "=================================="
echo "训练流程完成！"
echo "=================================="
echo ""
echo "后续步骤："
echo "  1. 查看训练日志: outputs/logs/"
echo "  2. 查看模型检查点: outputs/checkpoints/"
echo "  3. 查看ONNX模型: outputs/models/model.onnx"
echo "  4. 查看推理结果: outputs/inference_results/"
echo ""
