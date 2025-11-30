"""
Export PyTorch model to ONNX format.

将PyTorch模型导出为ONNX格式,用于部署。
"""

import argparse
import os
import logging

import torch
import onnx

from src.models import create_model
from src.training.utils import load_config, setup_logger, load_checkpoint, get_device


def export_to_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    input_shape: tuple,
    opset_version: int = 14,
    dynamic_batch: bool = True
):
    """
    导出模型为ONNX格式。
    
    Args:
        model: PyTorch模型
        onnx_path: ONNX模型保存路径
        input_shape: 输入形状 (B, C, H, W)
        opset_version: ONNX opset版本
        dynamic_batch: 是否使用动态批次维度
    """
    # 确保模型处于评估模式
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(input_shape)
    
    # 移到模型所在设备
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)
    
    # 定义输入输出名称
    input_names = ['image']
    output_names = ['lane_logits', 'sign_logits']
    
    # 定义动态轴（如果启用）
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'image': {0: 'batch_size'},
            'lane_logits': {0: 'batch_size'},
            'sign_logits': {0: 'batch_size'}
        }
    
    # 导出为ONNX
    logging.info(f"导出模型到ONNX: {onnx_path}")
    logging.info(f"输入形状: {input_shape}")
    logging.info(f"Opset版本: {opset_version}")
    logging.info(f"动态批次: {dynamic_batch}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
    
    logging.info(f"✓ ONNX模型已保存到: {onnx_path}")
    
    # 验证ONNX模型
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logging.info("✓ ONNX模型验证通过")
    except Exception as e:
        logging.error(f"✗ ONNX模型验证失败: {e}")
        raise
    
    # 打印模型信息
    logging.info(f"ONNX模型信息:")
    logging.info(f"  输入: {onnx_model.graph.input}")
    logging.info(f"  输出: {onnx_model.graph.output}")


def main():
    """主导出函数。"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='导出模型为ONNX格式')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output', type=str, default=None, help='ONNX输出路径（可选）')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opset版本')
    parser.add_argument('--dynamic-batch', action='store_true', help='使用动态批次维度')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    setup_logger(
        log_dir=config['paths'].get('logs_dir', 'logs'),
        log_filename='export_onnx.log'
    )
    
    # 获取设备
    device = get_device(use_cuda=True)
    
    # 创建模型
    logging.info("创建模型...")
    model = create_model(config)
    model = model.to(device)
    
    # 加载检查点
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"检查点文件不存在: {args.checkpoint}")
    
    logging.info(f"加载检查点: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, device=device)
    
    # 确定输出路径
    if args.output:
        onnx_path = args.output
    else:
        onnx_config = config.get('onnx', {})
        onnx_path = onnx_config.get('onnx_output_path', 'outputs/multitask_model.onnx')
    
    # 确定输入形状
    training_config = config.get('training', {})
    img_height = training_config.get('img_height', 256)
    img_width = training_config.get('img_width', 512)
    input_shape = (1, 3, img_height, img_width)
    
    # 导出模型
    export_to_onnx(
        model=model,
        onnx_path=onnx_path,
        input_shape=input_shape,
        opset_version=args.opset,
        dynamic_batch=args.dynamic_batch or config.get('onnx', {}).get('dynamic_batch', True)
    )
    
    logging.info("导出完成!")


if __name__ == "__main__":
    main()
