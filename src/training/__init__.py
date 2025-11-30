"""
Training module for multi-task learning.

训练模块，包含训练脚本、损失函数、评估指标和工具函数。
"""

from .losses import compute_multitask_loss, DiceLoss, SegmentationLoss
from .metrics import calculate_iou, calculate_dice, calculate_accuracy
from .utils import set_seed, save_checkpoint, load_checkpoint, setup_logger

__all__ = [
    "compute_multitask_loss",
    "DiceLoss",
    "SegmentationLoss",
    "calculate_iou",
    "calculate_dice",
    "calculate_accuracy",
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
    "setup_logger"
]
