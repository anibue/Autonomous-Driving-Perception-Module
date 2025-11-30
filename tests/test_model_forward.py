"""
Tests for model forward pass.

模型前向传播测试。
"""

import torch
import pytest

from src.models.multitask_unet import MultiTaskUNet, create_model


def test_model_creation():
    """测试模型创建。"""
    model = MultiTaskUNet(
        in_channels=3,
        encoder_channels=[64, 128, 256, 512],
        num_segmentation_classes=2,
        num_sign_classes=43,
        dropout_rate=0.3
    )
    
    assert model is not None, "模型创建失败"
    print("✓ 模型创建测试通过")


def test_model_forward():
    """测试模型前向传播。"""
    # 创建模型
    model = MultiTaskUNet(
        in_channels=3,
        encoder_channels=[32, 64, 128, 256],  # 使用较小的通道数加快测试
        num_segmentation_classes=2,
        num_sign_classes=10,
        dropout_rate=0.3
    )
    
    model.eval()
    
    # 创建随机输入
    batch_size = 2
    height = 128
    width = 256
    x = torch.randn(batch_size, 3, height, width)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(x)
    
    # 验证输出
    assert 'lane_logits' in outputs, "输出应包含lane_logits"
    assert 'sign_logits' in outputs, "输出应包含sign_logits"
    
    lane_logits = outputs['lane_logits']
    sign_logits = outputs['sign_logits']
    
    # 验证形状
    assert lane_logits.shape == (batch_size, 2, height, width), \
        f"车道线logits形状错误: {lane_logits.shape}"
    
    assert sign_logits.shape == (batch_size, 10), \
        f"标志logits形状错误: {sign_logits.shape}"
    
    print("✓ 模型前向传播测试通过")
    print(f"  输入形状: {x.shape}")
    print(f"  车道线logits形状: {lane_logits.shape}")
    print(f"  标志logits形状: {sign_logits.shape}")


def test_model_with_config():
    """测试从配置创建模型。"""
    config = {
        'model': {
            'encoder_channels': [32, 64, 128, 256],
            'num_segmentation_classes': 2,
            'num_sign_classes': 43,
            'dropout_rate': 0.3
        }
    }
    
    model = create_model(config)
    assert model is not None
    
    # 测试前向传播
    x = torch.randn(1, 3, 128, 256)
    with torch.no_grad():
        outputs = model(x)
    
    assert 'lane_logits' in outputs
    assert 'sign_logits' in outputs
    
    print("✓ 配置创建模型测试通过")


def test_model_predictions():
    """测试模型预测方法。"""
    model = MultiTaskUNet(
        in_channels=3,
        encoder_channels=[32, 64, 128, 256],
        num_segmentation_classes=2,
        num_sign_classes=10,
        dropout_rate=0.3
    )
    
    model.eval()
    x = torch.randn(2, 3, 128, 256)
    
    # 测试车道线预测
    with torch.no_grad():
        lane_pred = model.get_lane_prediction(x)
    
    assert lane_pred.shape == (2, 128, 256), \
        f"车道线预测形状错误: {lane_pred.shape}"
    
    # 测试交通标志预测
    with torch.no_grad():
        sign_pred = model.get_sign_prediction(x)
    
    assert sign_pred.shape == (2,), \
        f"标志预测形状错误: {sign_pred.shape}"
    
    print("✓ 模型预测方法测试通过")


if __name__ == '__main__':
    # 运行测试
    test_model_creation()
    test_model_forward()
    test_model_with_config()
    test_model_predictions()
    print("\n所有模型测试通过! ✓")
