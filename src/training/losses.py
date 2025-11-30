"""
Loss functions for multi-task learning.

多任务学习的损失函数，包括分割损失和分类损失。
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice损失函数。
    
    Dice损失常用于图像分割任务，尤其是在类别不平衡的情况下。
    
    公式：Dice = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    
    Attributes:
        smooth: 平滑项，防止除零错误
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        初始化Dice损失。
        
        Args:
            smooth: 平滑因子
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Dice损失。
        
        Args:
            predictions: 预测logits [B, C, H, W]
            targets: 目标标签 [B, H, W]（类别索引）
            
        Returns:
            Dice损失值（标量）
        """
        # 将logits转换为概率
        predictions = F.softmax(predictions, dim=1)  # [B, C, H, W]
        
        # 获取类别数
        num_classes = predictions.size(1)
        
        # 将目标转换为one-hot编码
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # 展平空间维度
        predictions = predictions.view(predictions.size(0), num_classes, -1)  # [B, C, H*W]
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), num_classes, -1)  # [B, C, H*W]
        
        # 计算交集和并集
        intersection = (predictions * targets_one_hot).sum(dim=2)  # [B, C]
        cardinality = predictions.sum(dim=2) + targets_one_hot.sum(dim=2)  # [B, C]
        
        # 计算Dice系数
        dice_coef = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)  # [B, C]
        
        # 对所有类别和批次取平均
        dice_loss = 1.0 - dice_coef.mean()
        
        return dice_loss


class SegmentationLoss(nn.Module):
    """
    分割任务的组合损失函数。
    
    结合交叉熵损失和Dice损失，以充分利用两者的优势：
    - 交叉熵：逐像素分类，收敛快
    - Dice损失：关注区域重叠，对类别不平衡鲁棒
    
    Attributes:
        bce_weight: 交叉熵损失权重
        dice_weight: Dice损失权重
        bce_loss: 交叉熵损失函数
        dice_loss: Dice损失函数
    """
    
    def __init__(
        self, 
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        ignore_index: int = -100
    ):
        """
        初始化分割损失。
        
        Args:
            bce_weight: 交叉熵损失的权重
            dice_weight: Dice损失的权重
            ignore_index: 忽略的标签索引（用于处理填充等）
        """
        super().__init__()
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        # 交叉熵损失
        self.bce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # Dice损失
        self.dice_loss = DiceLoss()
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算组合分割损失。
        
        Args:
            predictions: 预测logits [B, C, H, W]
            targets: 目标标签 [B, H, W]
            
        Returns:
            Tuple[总损失, 损失详情字典]:
            - 总损失：加权和
            - 损失详情：{'bce': BCE值, 'dice': Dice值}
        """
        # 计算BCE损失
        bce = self.bce_loss(predictions, targets)
        
        # 计算Dice损失
        dice = self.dice_loss(predictions, targets)
        
        # 加权组合
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        
        # 返回总损失和详情
        loss_details = {
            'bce': bce.item(),
            'dice': dice.item()
        }
        
        return total_loss, loss_details


class ClassificationLoss(nn.Module):
    """
    分类任务的交叉熵损失。
    
    用于交通标志识别任务。
    
    Attributes:
        ce_loss: 交叉熵损失函数
    """
    
    def __init__(self, label_smoothing: float = 0.0):
        """
        初始化分类损失。
        
        Args:
            label_smoothing: 标签平滑因子（0表示不使用）
        """
        super().__init__()
        
        # 交叉熵损失，支持标签平滑
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算分类损失。
        
        Args:
            predictions: 预测logits [B, num_classes]
            targets: 目标标签 [B]
            
        Returns:
            交叉熵损失值
        """
        return self.ce_loss(predictions, targets)


def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, Optional[torch.Tensor]],
    seg_loss_fn: SegmentationLoss,
    cls_loss_fn: ClassificationLoss,
    task_weights: Dict[str, float],
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算多任务学习的总损失。
    
    根据配置的权重组合车道线分割损失和交通标志分类损失。
    如果某个任务的目标为None，则跳过该任务的损失计算。
    
    Args:
        outputs: 模型输出字典
            - 'lane_logits': 车道线分割logits [B, C, H, W]
            - 'sign_logits': 交通标志分类logits [B, num_classes]
        targets: 目标标签字典
            - 'lane_mask': 车道线分割目标 [B, H, W] 或 None
            - 'sign_label': 交通标志分类目标 [B] 或 None
        seg_loss_fn: 分割损失函数
        cls_loss_fn: 分类损失函数
        task_weights: 任务权重字典
            - 'lane_segmentation': 分割任务权重
            - 'sign_classification': 分类任务权重
        device: 计算设备
        
    Returns:
        Tuple[总损失, 损失详情字典]:
        - 总损失：加权多任务损失
        - 损失详情：包含各任务损失值的字典
    """
    total_loss = torch.tensor(0.0, device=device)
    loss_details = {}
    
    # 计算车道线分割损失
    lane_mask = targets.get('lane_mask')
    if lane_mask is not None and outputs.get('lane_logits') is not None:
        lane_logits = outputs['lane_logits']
        seg_loss, seg_details = seg_loss_fn(lane_logits, lane_mask)
        
        # 加权
        weighted_seg_loss = task_weights.get('lane_segmentation', 1.0) * seg_loss
        total_loss = total_loss + weighted_seg_loss
        
        # 记录详情
        loss_details['lane_seg_total'] = seg_loss.item()
        loss_details['lane_seg_bce'] = seg_details['bce']
        loss_details['lane_seg_dice'] = seg_details['dice']
        loss_details['lane_seg_weighted'] = weighted_seg_loss.item()
    
    # 计算交通标志分类损失
    sign_label = targets.get('sign_label')
    if sign_label is not None and outputs.get('sign_logits') is not None:
        sign_logits = outputs['sign_logits']
        cls_loss = cls_loss_fn(sign_logits, sign_label)
        
        # 加权
        weighted_cls_loss = task_weights.get('sign_classification', 1.0) * cls_loss
        total_loss = total_loss + weighted_cls_loss
        
        # 记录详情
        loss_details['sign_cls_total'] = cls_loss.item()
        loss_details['sign_cls_weighted'] = weighted_cls_loss.item()
    
    # 记录总损失
    loss_details['total'] = total_loss.item()
    
    return total_loss, loss_details


def create_loss_functions(
    config: Dict
) -> Tuple[SegmentationLoss, ClassificationLoss]:
    """
    根据配置创建损失函数。
    
    Args:
        config: 配置字典
        
    Returns:
        Tuple[分割损失函数, 分类损失函数]
    """
    # 分割损失
    seg_loss_fn = SegmentationLoss(
        bce_weight=0.5,
        dice_weight=0.5
    )
    
    # 分类损失
    cls_loss_fn = ClassificationLoss(
        label_smoothing=0.0
    )
    
    return seg_loss_fn, cls_loss_fn
