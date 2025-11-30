"""
ONNX model inference script.

使用ONNX Runtime进行模型推理。
"""

import argparse
import os
import logging
from typing import Tuple, Dict

import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

from src.training.utils import load_config, setup_logger


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int],
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    预处理输入图像。
    
    将图像调整大小、归一化,并转换为模型输入格式。
    
    Args:
        image_path: 图像文件路径
        target_size: 目标尺寸 (height, width)
        mean: 归一化均值
        std: 归一化标准差
        
    Returns:
        预处理后的图像数组 [1, 3, H, W]
    """
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    
    # 调整大小
    image = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    # 转换为numpy数组
    image = np.array(image).astype(np.float32) / 255.0
    
    # 归一化
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    image = (image - mean) / std
    
    # 转换为CHW格式并添加批次维度
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = np.expand_dims(image, axis=0)  # CHW -> BCHW
    
    return image


def postprocess_segmentation(
    lane_logits: np.ndarray,
    original_size: Tuple[int, int]
) -> np.ndarray:
    """
    后处理分割输出。
    
    将logits转换为类别预测,并调整到原始图像尺寸。
    
    Args:
        lane_logits: 分割logits [1, C, H, W]
        original_size: 原始图像尺寸 (height, width)
        
    Returns:
        分割掩码 [H, W]
    """
    # 应用softmax获取概率
    # 对于numpy,手动实现softmax
    exp_logits = np.exp(lane_logits - np.max(lane_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # 获取预测类别
    pred_mask = np.argmax(probs, axis=1)[0]  # [H, W]
    
    # 调整到原始尺寸
    pred_mask = cv2.resize(
        pred_mask.astype(np.uint8),
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    return pred_mask


def postprocess_classification(
    sign_logits: np.ndarray
) -> Tuple[int, float]:
    """
    后处理分类输出。
    
    获取top-1预测类别和置信度。
    
    Args:
        sign_logits: 分类logits [1, num_classes]
        
    Returns:
        Tuple[预测类别, 置信度]
    """
    # 应用softmax获取概率
    exp_logits = np.exp(sign_logits - np.max(sign_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # 获取top-1预测
    pred_class = np.argmax(probs, axis=1)[0]
    confidence = probs[0, pred_class]
    
    return int(pred_class), float(confidence)


def visualize_results(
    image_path: str,
    lane_mask: np.ndarray,
    sign_class: int,
    sign_confidence: float,
    output_path: str
):
    """
    可视化推理结果。
    
    将分割掩码叠加到原始图像上,并添加分类结果文本。
    
    Args:
        image_path: 原始图像路径
        lane_mask: 车道线分割掩码 [H, W]
        sign_class: 预测的交通标志类别
        sign_confidence: 分类置信度
        output_path: 输出图像路径
    """
    # 读取原始图像
    image = cv2.imread(image_path)
    
    # 创建彩色掩码（车道线用绿色表示）
    colored_mask = np.zeros_like(image)
    colored_mask[lane_mask == 1] = [0, 255, 0]  # 绿色
    
    # 叠加掩码到原始图像
    alpha = 0.5
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    # 添加文本信息
    text = f"Sign Class: {sign_class}, Conf: {sign_confidence:.2f}"
    cv2.putText(
        overlay, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
    )
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)
    logging.info(f"可视化结果已保存到: {output_path}")


def run_onnx_inference(
    onnx_model_path: str,
    image_path: str,
    config: Dict,
    output_dir: str = "outputs"
):
    """
    运行ONNX推理。
    
    Args:
        onnx_model_path: ONNX模型路径
        image_path: 输入图像路径
        config: 配置字典
        output_dir: 输出目录
    """
    # 创建ONNX Runtime会话
    logging.info(f"加载ONNX模型: {onnx_model_path}")
    
    # 配置推理选项
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 选择执行提供者（CPU或CUDA）
    providers = ['CPUExecutionProvider']
    if config.get('ros2', {}).get('use_gpu', False):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # 创建推理会话
    session = ort.InferenceSession(
        onnx_model_path,
        sess_options=sess_options,
        providers=providers
    )
    
    logging.info(f"使用执行提供者: {session.get_providers()}")
    
    # 获取输入输出信息
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    logging.info(f"模型输入: {input_name}")
    logging.info(f"模型输出: {output_names}")
    
    # 读取原始图像尺寸
    original_image = Image.open(image_path)
    original_size = (original_image.height, original_image.width)
    
    # 预处理图像
    training_config = config.get('training', {})
    target_size = (
        training_config.get('img_height', 256),
        training_config.get('img_width', 512)
    )
    
    aug_config = config.get('augmentation', {}).get('val', {})
    norm_config = aug_config.get('normalize', {})
    mean = tuple(norm_config.get('mean', [0.485, 0.456, 0.406]))
    std = tuple(norm_config.get('std', [0.229, 0.224, 0.225]))
    
    logging.info("预处理图像...")
    input_data = preprocess_image(image_path, target_size, mean, std)
    
    # 运行推理
    logging.info("运行推理...")
    outputs = session.run(output_names, {input_name: input_data})
    
    lane_logits = outputs[0]  # [1, C, H, W]
    sign_logits = outputs[1]  # [1, num_classes]
    
    logging.info(f"车道线logits形状: {lane_logits.shape}")
    logging.info(f"标志logits形状: {sign_logits.shape}")
    
    # 后处理
    logging.info("后处理结果...")
    lane_mask = postprocess_segmentation(lane_logits, original_size)
    sign_class, sign_confidence = postprocess_classification(sign_logits)
    
    # 打印结果
    logging.info(f"\n推理结果:")
    logging.info(f"  交通标志类别: {sign_class}")
    logging.info(f"  分类置信度: {sign_confidence:.4f}")
    logging.info(f"  车道线像素数: {np.sum(lane_mask == 1)}")
    
    # 保存分割掩码
    mask_output_path = os.path.join(output_dir, "lane_mask.png")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(mask_output_path, lane_mask * 255)
    logging.info(f"分割掩码已保存到: {mask_output_path}")
    
    # 可视化结果
    vis_output_path = os.path.join(output_dir, "result_visualization.jpg")
    visualize_results(image_path, lane_mask, sign_class, sign_confidence, vis_output_path)


def main():
    """主推理函数。"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ONNX模型推理')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--model', type=str, required=True, help='ONNX模型路径')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, default='outputs/inference', help='输出目录')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    setup_logger(
        log_dir=args.output,
        log_filename='inference.log'
    )
    
    # 运行推理
    run_onnx_inference(args.model, args.image, config, args.output)
    
    logging.info("推理完成!")


if __name__ == "__main__":
    main()
