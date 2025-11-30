"""
Utility functions for training.

训练工具函数，包括种子设置、检查点管理、日志配置等。
"""

import os
import random
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """
    设置随机种子以确保可复现性。
    
    固定Python、NumPy和PyTorch的随机种子。
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 如果使用CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 为了完全的可复现性，需要设置这些选项
        # 注意：这可能会影响性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    """
    保存模型检查点。
    
    保存模型状态、优化器状态、训练轮数和指标。
    
    Args:
        model: 模型实例
        optimizer: 优化器实例
        epoch: 当前轮数
        metrics: 评估指标字典
        filepath: 检查点保存路径
        scheduler: 学习率调度器（可选）
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 准备检查点字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # 保存调度器状态
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 保存到文件
    torch.save(checkpoint, filepath)
    
    logging.info(f"检查点已保存到: {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    加载模型检查点。
    
    恢复模型状态、优化器状态和训练信息。
    
    Args:
        filepath: 检查点文件路径
        model: 模型实例
        optimizer: 优化器实例（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
        
    Returns:
        包含轮数和指标的字典
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"检查点文件不存在: {filepath}")
    
    # 加载检查点
    checkpoint = torch.load(filepath, map_location=device)
    
    # 恢复模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"模型状态已从 {filepath} 加载")
    
    # 恢复优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info("优化器状态已加载")
    
    # 恢复调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging.info("调度器状态已加载")
    
    # 返回训练信息
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {})
    }


def setup_logger(
    log_dir: str,
    log_filename: str = "training.log",
    level: str = "INFO"
) -> logging.Logger:
    """
    配置日志记录器。
    
    创建同时输出到控制台和文件的日志记录器。
    
    Args:
        log_dir: 日志目录
        log_filename: 日志文件名
        level: 日志级别（DEBUG, INFO, WARNING, ERROR）
        
    Returns:
        配置好的Logger实例
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件路径
    log_path = os.path.join(log_dir, log_filename)
    
    # 配置日志格式
    log_format = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    logging.info(f"日志记录器已配置，日志文件: {log_path}")
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件。
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    保存配置到YAML文件。
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logging.info(f"配置已保存到: {save_path}")


def create_directories(config: Dict[str, Any]):
    """
    创建训练所需的目录。
    
    根据配置创建输出、检查点、日志等目录。
    
    Args:
        config: 配置字典
    """
    paths = config.get('paths', {})
    
    # 需要创建的目录列表
    dirs_to_create = [
        paths.get('output_dir', 'outputs'),
        paths.get('checkpoints_dir', 'checkpoints'),
        paths.get('logs_dir', 'logs')
    ]
    
    # 创建目录
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"目录已创建: {dir_path}")


def get_device(use_cuda: bool = True) -> torch.device:
    """
    获取计算设备。
    
    Args:
        use_cuda: 是否尝试使用CUDA
        
    Returns:
        torch.device实例
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logging.info("使用CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型的可训练参数数量。
    
    Args:
        model: 模型实例
        
    Returns:
        可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    格式化时间（秒）为可读字符串。
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串（例如："1h 23m 45s"）
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class AverageMeter:
    """
    平均值计算器。
    
    用于跟踪和计算运行平均值，常用于损失和指标的统计。
    
    Attributes:
        val: 当前值
        avg: 平均值
        sum: 累积和
        count: 计数
    """
    
    def __init__(self):
        """初始化计算器。"""
        self.reset()
    
    def reset(self):
        """重置所有统计量。"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        更新统计量。
        
        Args:
            val: 新值
            n: 样本数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class EarlyStopping:
    """
    早停机制。
    
    监控验证指标，如果长时间没有改善则停止训练。
    
    Attributes:
        patience: 耐心值（无改善的轮数）
        min_delta: 最小改善阈值
        counter: 当前计数器
        best_score: 最佳分数
        early_stop: 是否应该停止
    """
    
    def __init__(
        self, 
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        """
        初始化早停机制。
        
        Args:
            patience: 无改善时的最大等待轮数
            min_delta: 视为改善的最小变化量
            mode: 监控模式，'max'（越大越好）或'min'（越小越好）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # 根据模式设置比较函数
        if mode == 'max':
            self.is_better = lambda new, best: new > best + min_delta
            self.best_score = float('-inf')
        else:
            self.is_better = lambda new, best: new < best - min_delta
            self.best_score = float('inf')
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停。
        
        Args:
            score: 当前验证分数
            
        Returns:
            是否应该停止训练
        """
        if self.is_better(score, self.best_score):
            # 有改善
            self.best_score = score
            self.counter = 0
        else:
            # 无改善
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info(f"早停触发！已经{self.patience}轮无改善。")
        
        return self.early_stop
