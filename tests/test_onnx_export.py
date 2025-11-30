"""
Tests for ONNX export functionality.

ONNX导出功能测试。
"""

import os
import tempfile

import torch
import onnx
import pytest

from src.models.multitask_unet import MultiTaskUNet


def test_onnx_export():
    """测试ONNX导出。"""
    # 创建小型模型
    model = MultiTaskUNet(
        in_channels=3,
        encoder_channels=[16, 32, 64, 128],  # 小型模型
        num_segmentation_classes=2,
        num_sign_classes=10,
        dropout_rate=0.3
    )
    
    model.eval()
    
    # 创建临时文件
    temp_dir = tempfile.mkdtemp()
    onnx_path = os.path.join(temp_dir, 'test_model.onnx')
    
    try:
        # 导出ONNX
        dummy_input = torch.randn(1, 3, 128, 256)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            input_names=['image'],
            output_names=['lane_logits', 'sign_logits'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'lane_logits': {0: 'batch_size'},
                'sign_logits': {0: 'batch_size'}
            }
        )
        
        # 验证ONNX文件存在
        assert os.path.exists(onnx_path), "ONNX文件未生成"
        
        # 验证ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        print("✓ ONNX导出测试通过")
        print(f"  ONNX文件大小: {os.path.getsize(onnx_path) / 1024:.2f} KB")
    
    finally:
        # 清理
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        os.rmdir(temp_dir)


def test_onnx_model_inference():
    """测试ONNX模型推理（如果有onnxruntime）。"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("⊗ onnxruntime未安装,跳过ONNX推理测试")
        return
    
    # 创建并导出模型
    model = MultiTaskUNet(
        in_channels=3,
        encoder_channels=[16, 32, 64, 128],
        num_segmentation_classes=2,
        num_sign_classes=10,
        dropout_rate=0.3
    )
    
    model.eval()
    
    temp_dir = tempfile.mkdtemp()
    onnx_path = os.path.join(temp_dir, 'test_model.onnx')
    
    try:
        # 导出
        dummy_input = torch.randn(1, 3, 128, 256)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            input_names=['image'],
            output_names=['lane_logits', 'sign_logits']
        )
        
        # 创建ONNX Runtime会话
        session = ort.InferenceSession(onnx_path)
        
        # 运行推理
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: dummy_input.numpy()})
        
        # 验证输出
        assert len(outputs) == 2, "应有2个输出"
        assert outputs[0].shape == (1, 2, 128, 256), "车道线logits形状错误"
        assert outputs[1].shape == (1, 10), "标志logits形状错误"
        
        print("✓ ONNX模型推理测试通过")
        print(f"  车道线输出形状: {outputs[0].shape}")
        print(f"  标志输出形状: {outputs[1].shape}")
    
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        os.rmdir(temp_dir)


if __name__ == '__main__':
    # 运行测试
    test_onnx_export()
    test_onnx_model_inference()
    print("\n所有ONNX测试通过! ✓")
