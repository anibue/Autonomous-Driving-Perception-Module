"""
Evaluation metrics for multi-task learning.

评估指标模块，用于计算模型性能指标。
"""

from typing import Optional

import torch
import torch.nn.functional as F


def calculate_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    计算IoU（Intersection over Union）指标。
    
    IoU是语义分割中最常用的评估指标，衡量预测和真实标签的重叠程度。
    
    公式：IoU = TP / (TP + FP + FN)
    
    Args:
        predictions: 预测logits [B, C, H, W] 或预测类别 [B, H, W]
        targets: 目标标签 [B, H, W]
        num_classes: 类别数
        ignore_index: 忽略的标签索引
        
    Returns:
        各类别的IoU [num_classes]
    """
    # 如果predictions是logits，转换为类别预测
    if predictions.dim() == 4:
        predictions = torch.argmax(predictions, dim=1)  # [B, H, W]
    
    # 初始化IoU数组
    ious = []
    
    # 对每个类别计算IoU
    for cls in range(num_classes):
        # 创建二值掩码
        pred_mask = (predictions == cls)
        target_mask = (targets == cls)
        
        # 排除ignore_index
        valid_mask = (targets != ignore_index)
        pred_mask = pred_mask & valid_mask
        target_mask = target_mask & valid_mask
        
        # 计算交集和并集
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        # 避免除零
        if union == 0:
            iou = torch.tensor(1.0) if intersection == 0 else torch.tensor(0.0)
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return torch.stack(ious)


def calculate_dice(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    计算Dice系数。
    
    Dice系数也称为F1分数，是另一个常用的分割评估指标。
    
    公式：Dice = 2 * TP / (2 * TP + FP + FN)
    
    Args:
        predictions: 预测logits [B, C, H, W] 或预测类别 [B, H, W]
        targets: 目标标签 [B, H, W]
        num_classes: 类别数
        ignore_index: 忽略的标签索引
        
    Returns:
        各类别的Dice系数 [num_classes]
    """
    # 如果predictions是logits，转换为类别预测
    if predictions.dim() == 4:
        predictions = torch.argmax(predictions, dim=1)  # [B, H, W]
    
    # 初始化Dice数组
    dice_scores = []
    
    # 对每个类别计算Dice
    for cls in range(num_classes):
        # 创建二值掩码
        pred_mask = (predictions == cls)
        target_mask = (targets == cls)
        
        # 排除ignore_index
        valid_mask = (targets != ignore_index)
        pred_mask = pred_mask & valid_mask
        target_mask = target_mask & valid_mask
        
        # 计算交集和基数
        intersection = (pred_mask & target_mask).sum().float()
        pred_sum = pred_mask.sum().float()
        target_sum = target_mask.sum().float()
        
        # 计算Dice系数
        if (pred_sum + target_sum) == 0:
            dice = torch.tensor(1.0) if intersection == 0 else torch.tensor(0.0)
        else:
            dice = (2.0 * intersection) / (pred_sum + target_sum)
        
        dice_scores.append(dice)
    
    return torch.stack(dice_scores)


def calculate_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    topk: int = 1
) -> float:
    """
    计算分类准确率。
    
    用于交通标志识别任务的评估。
    
    Args:
        predictions: 预测logits [B, num_classes]
        targets: 目标标签 [B]
        topk: Top-K准确率（默认Top-1）
        
    Returns:
        准确率（百分比）
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        
        # 获取top-k预测
        _, pred_indices = predictions.topk(topk, dim=1, largest=True, sorted=True)
        
        # 检查是否正确
        correct = pred_indices.eq(targets.view(-1, 1).expand_as(pred_indices))
        
        # 计算准确率
        correct_k = correct.sum().float()
        accuracy = (correct_k / batch_size) * 100.0
        
        return accuracy.item()


def calculate_pixel_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    计算像素准确率。
    
    像素准确率是最简单的分割指标，衡量正确分类的像素比例。
    
    Args:
        predictions: 预测logits [B, C, H, W] 或预测类别 [B, H, W]
        targets: 目标标签 [B, H, W]
        ignore_index: 忽略的标签索引
        
    Returns:
        像素准确率（百分比）
    """
    # 如果predictions是logits，转换为类别预测
    if predictions.dim() == 4:
        predictions = torch.argmax(predictions, dim=1)  # [B, H, W]
    
    # 创建有效像素掩码
    valid_mask = (targets != ignore_index)
    
    # 计算正确像素数
    correct = (predictions == targets) & valid_mask
    correct_pixels = correct.sum().float()
    
    # 计算总有效像素数
    total_pixels = valid_mask.sum().float()
    
    # 计算准确率
    if total_pixels == 0:
        return 0.0
    
    accuracy = (correct_pixels / total_pixels) * 100.0
    return accuracy.item()


def calculate_mean_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
    ignore_background: bool = False
) -> float:
    """
    计算平均IoU（mIoU）。
    
    mIoU是所有类别IoU的平均值，是分割任务的标准评估指标。
    
    Args:
        predictions: 预测logits [B, C, H, W] 或预测类别 [B, H, W]
        targets: 目标标签 [B, H, W]
        num_classes: 类别数
        ignore_index: 忽略的标签索引
        ignore_background: 是否忽略背景类（类别0）
        
    Returns:
        平均IoU（百分比）
    """
    # 计算各类别IoU
    ious = calculate_iou(predictions, targets, num_classes, ignore_index)
    
    # 选择要计算平均值的类别
    if ignore_background and num_classes > 1:
        ious = ious[1:]  # 排除背景类
    
    # 计算平均值
    mean_iou = ious.mean() * 100.0
    return mean_iou.item()


class MetricsTracker:
    """
    指标跟踪器。
    
    用于在训练过程中累积和平均多个批次的指标。
    
    Attributes:
        metrics: 指标累积字典
        counts: 各指标的计数
    """
    
    def __init__(self):
        """初始化指标跟踪器。"""
        self.metrics = {}
        self.counts = {}
    
    def update(self, metric_name: str, value: float, count: int = 1):
        """
        更新指标值。
        
        Args:
            metric_name: 指标名称
            value: 指标值
            count: 样本数量
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = 0.0
            self.counts[metric_name] = 0
        
        self.metrics[metric_name] += value * count
        self.counts[metric_name] += count
    
    def get_average(self, metric_name: str) -> Optional[float]:
        """
        获取指标的平均值。
        
        Args:
            metric_name: 指标名称
            
        Returns:
            平均值，如果指标不存在返回None
        """
        if metric_name not in self.metrics or self.counts[metric_name] == 0:
            return None
        
        return self.metrics[metric_name] / self.counts[metric_name]
    
    def get_all_averages(self) -> dict:
        """
        获取所有指标的平均值。
        
        Returns:
            包含所有指标平均值的字典
        """
        averages = {}
        for metric_name in self.metrics.keys():
            avg = self.get_average(metric_name)
            if avg is not None:
                averages[metric_name] = avg
        return averages
    
    def reset(self):
        """重置所有指标。"""
        self.metrics.clear()
        self.counts.clear()
    
    def __str__(self) -> str:
        """返回指标的字符串表示。"""
        averages = self.get_all_averages()
        return ", ".join([f"{k}: {v:.4f}" for k, v in averages.items()])
