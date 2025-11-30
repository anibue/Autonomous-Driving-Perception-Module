"""
Main training script for multi-task learning.

主训练脚本,用于训练多任务UNet模型。
"""

import argparse
import os
import time
import logging
from typing import Dict, Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import create_dataloaders
from src.models import create_model
from src.training.losses import create_loss_functions, compute_multitask_loss
from src.training.metrics import (
    calculate_mean_iou,
    calculate_accuracy,
    MetricsTracker
)
from src.training.utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    setup_logger,
    load_config,
    save_config,
    create_directories,
    get_device,
    count_parameters,
    format_time,
    EarlyStopping
)


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    seg_loss_fn,
    cls_loss_fn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_weights: Dict[str, float],
    epoch: int,
    log_interval: int = 10
) -> Dict[str, float]:
    """
    训练一个epoch。
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        seg_loss_fn: 分割损失函数
        cls_loss_fn: 分类损失函数
        optimizer: 优化器
        device: 计算设备
        task_weights: 任务权重
        epoch: 当前轮数
        log_interval: 日志打印间隔
        
    Returns:
        包含平均损失和指标的字典
    """
    model.train()
    
    # 指标跟踪器
    metrics_tracker = MetricsTracker()
    
    # 进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, batch in enumerate(pbar):
        # 将数据移到设备
        images = batch['image'].to(device)
        lane_masks = batch['lane_mask'].to(device) if batch['lane_mask'] is not None else None
        sign_labels = batch['sign_label'].to(device) if batch['sign_label'] is not None else None
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        
        # 计算损失
        targets = {
            'lane_mask': lane_masks,
            'sign_label': sign_labels
        }
        
        loss, loss_details = compute_multitask_loss(
            outputs, targets, seg_loss_fn, cls_loss_fn, task_weights, device
        )
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 更新指标
        metrics_tracker.update('loss', loss.item(), images.size(0))
        for key, value in loss_details.items():
            metrics_tracker.update(f'train_{key}', value, images.size(0))
        
        # 计算车道线分割IoU（如果有）
        if lane_masks is not None and outputs.get('lane_logits') is not None:
            with torch.no_grad():
                miou = calculate_mean_iou(
                    outputs['lane_logits'],
                    lane_masks,
                    num_classes=outputs['lane_logits'].size(1)
                )
                metrics_tracker.update('train_lane_miou', miou, images.size(0))
        
        # 计算交通标志准确率（如果有）
        if sign_labels is not None and outputs.get('sign_logits') is not None:
            with torch.no_grad():
                acc = calculate_accuracy(outputs['sign_logits'], sign_labels)
                metrics_tracker.update('train_sign_acc', acc, images.size(0))
        
        # 更新进度条
        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix({
                'loss': f"{metrics_tracker.get_average('loss'):.4f}"
            })
    
    return metrics_tracker.get_all_averages()


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    seg_loss_fn,
    cls_loss_fn,
    device: torch.device,
    task_weights: Dict[str, float],
    epoch: int
) -> Dict[str, float]:
    """
    在验证集上评估模型。
    
    Args:
        model: 模型实例
        val_loader: 验证数据加载器
        seg_loss_fn: 分割损失函数
        cls_loss_fn: 分类损失函数
        device: 计算设备
        task_weights: 任务权重
        epoch: 当前轮数
        
    Returns:
        包含平均损失和指标的字典
    """
    model.eval()
    
    # 指标跟踪器
    metrics_tracker = MetricsTracker()
    
    # 进度条
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch in pbar:
            # 将数据移到设备
            images = batch['image'].to(device)
            lane_masks = batch['lane_mask'].to(device) if batch['lane_mask'] is not None else None
            sign_labels = batch['sign_label'].to(device) if batch['sign_label'] is not None else None
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            targets = {
                'lane_mask': lane_masks,
                'sign_label': sign_labels
            }
            
            loss, loss_details = compute_multitask_loss(
                outputs, targets, seg_loss_fn, cls_loss_fn, task_weights, device
            )
            
            # 更新指标
            metrics_tracker.update('val_loss', loss.item(), images.size(0))
            for key, value in loss_details.items():
                metrics_tracker.update(f'val_{key}', value, images.size(0))
            
            # 计算车道线分割IoU（如果有）
            if lane_masks is not None and outputs.get('lane_logits') is not None:
                miou = calculate_mean_iou(
                    outputs['lane_logits'],
                    lane_masks,
                    num_classes=outputs['lane_logits'].size(1)
                )
                metrics_tracker.update('val_lane_miou', miou, images.size(0))
            
            # 计算交通标志准确率（如果有）
            if sign_labels is not None and outputs.get('sign_logits') is not None:
                acc = calculate_accuracy(outputs['sign_logits'], sign_labels)
                metrics_tracker.update('val_sign_acc', acc, images.size(0))
    
    return metrics_tracker.get_all_averages()


def main():
    """主训练函数。"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练多任务UNet模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建必要的目录
    create_directories(config)
    
    # 设置日志
    log_config = config.get('logging', {})
    logger = setup_logger(
        log_dir=config['paths'].get('logs_dir', 'logs'),
        level=log_config.get('level', 'INFO')
    )
    
    # 保存配置副本
    save_config(config, os.path.join(config['paths']['output_dir'], 'config.yaml'))
    
    # 设置随机种子
    seed = config.get('training', {}).get('seed', 42)
    set_seed(seed)
    logging.info(f"随机种子设置为: {seed}")
    
    # 获取设备
    device = get_device(use_cuda=True)
    
    # 创建模型
    logging.info("创建模型...")
    model = create_model(config)
    model = model.to(device)
    
    # 打印模型信息
    num_params = count_parameters(model)
    logging.info(f"模型参数数量: {num_params:,}")
    
    # 创建数据加载器
    logging.info("创建数据加载器...")
    train_loader, val_loader = create_dataloaders(config)
    logging.info(f"训练样本数: {len(train_loader.dataset)}, 验证样本数: {len(val_loader.dataset)}")
    
    # 创建损失函数
    seg_loss_fn, cls_loss_fn = create_loss_functions(config)
    seg_loss_fn = seg_loss_fn.to(device)
    cls_loss_fn = cls_loss_fn.to(device)
    
    # 任务权重
    task_weights = config.get('training', {}).get('task_loss_weights', {
        'lane_segmentation': 1.0,
        'sign_classification': 0.5
    })
    logging.info(f"任务权重: {task_weights}")
    
    # 创建优化器
    training_config = config.get('training', {})
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.get('learning_rate', 0.001),
        weight_decay=training_config.get('weight_decay', 0.0001)
    )
    
    # 创建学习率调度器
    scheduler_config = training_config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'cosine')
    
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.get('num_epochs', 100),
            eta_min=scheduler_config.get('min_lr', 0.00001)
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    else:
        scheduler = None
    
    # 早停机制
    early_stop_config = training_config.get('early_stopping', {})
    early_stopping = None
    if early_stop_config.get('enabled', True):
        early_stopping = EarlyStopping(
            patience=early_stop_config.get('patience', 15),
            min_delta=early_stop_config.get('min_delta', 0.001),
            mode='max'
        )
    
    # 恢复训练（如果指定）
    start_epoch = 0
    best_metric = 0.0
    
    if args.resume and os.path.exists(args.resume):
        logging.info(f"从检查点恢复训练: {args.resume}")
        checkpoint_info = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = checkpoint_info['epoch'] + 1
        best_metric = checkpoint_info.get('metrics', {}).get('combined', 0.0)
    
    # 训练循环
    num_epochs = training_config.get('num_epochs', 100)
    checkpoints_dir = config['paths'].get('checkpoints_dir', 'checkpoints')
    save_interval = log_config.get('save_interval', 5)
    
    logging.info("开始训练...")
    total_start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # 训练一个epoch
        train_metrics = train_one_epoch(
            model, train_loader, seg_loss_fn, cls_loss_fn,
            optimizer, device, task_weights, epoch,
            log_interval=log_config.get('log_interval', 10)
        )
        
        # 验证
        val_metrics = validate(
            model, val_loader, seg_loss_fn, cls_loss_fn,
            device, task_weights, epoch
        )
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 计算组合指标（用于早停和最佳模型选择）
        combined_metric = 0.0
        metric_count = 0
        
        if 'val_lane_miou' in val_metrics:
            combined_metric += val_metrics['val_lane_miou']
            metric_count += 1
        
        if 'val_sign_acc' in val_metrics:
            combined_metric += val_metrics['val_sign_acc']
            metric_count += 1
        
        if metric_count > 0:
            combined_metric /= metric_count
        
        val_metrics['combined'] = combined_metric
        
        # 记录训练信息
        epoch_time = time.time() - epoch_start_time
        logging.info(f"\nEpoch {epoch}/{num_epochs-1} - 耗时: {format_time(epoch_time)}")
        logging.info(f"训练 - Loss: {train_metrics.get('loss', 0):.4f}")
        
        if 'train_lane_miou' in train_metrics:
            logging.info(f"训练 - Lane mIoU: {train_metrics['train_lane_miou']:.2f}%")
        
        if 'train_sign_acc' in train_metrics:
            logging.info(f"训练 - Sign Acc: {train_metrics['train_sign_acc']:.2f}%")
        
        logging.info(f"验证 - Loss: {val_metrics.get('val_loss', 0):.4f}")
        
        if 'val_lane_miou' in val_metrics:
            logging.info(f"验证 - Lane mIoU: {val_metrics['val_lane_miou']:.2f}%")
        
        if 'val_sign_acc' in val_metrics:
            logging.info(f"验证 - Sign Acc: {val_metrics['val_sign_acc']:.2f}%")
        
        logging.info(f"验证 - Combined Metric: {combined_metric:.2f}%")
        logging.info(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if combined_metric > best_metric:
            best_metric = combined_metric
            best_path = os.path.join(checkpoints_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, best_path, scheduler)
            logging.info(f"★ 新的最佳模型已保存! Combined Metric: {best_metric:.2f}%")
        
        # 定期保存检查点
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path, scheduler)
        
        # 早停检查
        if early_stopping is not None:
            if early_stopping(combined_metric):
                logging.info(f"早停触发，训练结束于第 {epoch} 轮")
                break
    
    # 训练完成
    total_time = time.time() - total_start_time
    logging.info(f"\n训练完成! 总耗时: {format_time(total_time)}")
    logging.info(f"最佳验证指标: {best_metric:.2f}%")
    
    # 保存最终模型
    final_path = os.path.join(checkpoints_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, num_epochs - 1, val_metrics, final_path, scheduler)
    logging.info(f"最终模型已保存到: {final_path}")


if __name__ == "__main__":
    main()
