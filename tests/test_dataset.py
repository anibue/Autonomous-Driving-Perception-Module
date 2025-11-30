"""
Tests for dataset module.

数据集模块测试。
"""

import os
import tempfile
import shutil

import numpy as np
import torch
from PIL import Image
import pytest

from src.data.dataset import MultiTaskDataset


def create_dummy_dataset(temp_dir):
    """
    创建临时测试数据集。
    
    生成一些简单的测试图像和掩码。
    
    Args:
        temp_dir: 临时目录路径
    """
    # 创建目录结构
    lane_images_dir = os.path.join(temp_dir, 'lane', 'images')
    lane_masks_dir = os.path.join(temp_dir, 'lane', 'masks')
    sign_images_dir = os.path.join(temp_dir, 'sign', 'class_0')
    
    os.makedirs(lane_images_dir, exist_ok=True)
    os.makedirs(lane_masks_dir, exist_ok=True)
    os.makedirs(sign_images_dir, exist_ok=True)
    
    # 生成5个车道线样本
    for i in range(5):
        # 生成随机RGB图像
        image = np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8)
        Image.fromarray(image).save(os.path.join(lane_images_dir, f'lane_{i}.jpg'))
        
        # 生成随机二值掩码
        mask = np.random.randint(0, 2, (128, 256), dtype=np.uint8)
        Image.fromarray(mask * 255).save(os.path.join(lane_masks_dir, f'lane_{i}.png'))
    
    # 生成5个交通标志样本
    for i in range(5):
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(image).save(os.path.join(sign_images_dir, f'sign_{i}.jpg'))
    
    return {
        'lane_images_dir': lane_images_dir,
        'lane_masks_dir': lane_masks_dir,
        'sign_images_dir': os.path.join(temp_dir, 'sign')
    }


def test_dataset_creation():
    """测试数据集创建。"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建虚拟数据
        paths = create_dummy_dataset(temp_dir)
        
        # 创建数据集实例
        dataset = MultiTaskDataset(
            lane_images_dir=paths['lane_images_dir'],
            lane_masks_dir=paths['lane_masks_dir'],
            sign_images_dir=paths['sign_images_dir'],
            sign_labels_file=None,
            transform=None,
            mode='joint'
        )
        
        # 验证数据集长度
        assert len(dataset) > 0, "数据集不应为空"
        
        # 验证样本加载
        sample = dataset[0]
        assert 'image' in sample, "样本应包含image键"
        assert 'lane_mask' in sample, "样本应包含lane_mask键"
        assert 'sign_label' in sample, "样本应包含sign_label键"
        
        # 验证数据类型和形状
        if sample['image'] is not None:
            assert isinstance(sample['image'], torch.Tensor), "image应为Tensor"
            assert sample['image'].ndim == 3, "image应为3维张量 [C, H, W]"
        
        if sample['lane_mask'] is not None:
            assert isinstance(sample['lane_mask'], torch.Tensor), "lane_mask应为Tensor"
        
        if sample['sign_label'] is not None:
            assert isinstance(sample['sign_label'], torch.Tensor), "sign_label应为Tensor"
        
        print(f"✓ 数据集创建测试通过")
        print(f"  数据集长度: {len(dataset)}")
        print(f"  样本键: {list(sample.keys())}")
    
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)


def test_dataset_lane_only():
    """测试仅车道线模式。"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        paths = create_dummy_dataset(temp_dir)
        
        dataset = MultiTaskDataset(
            lane_images_dir=paths['lane_images_dir'],
            lane_masks_dir=paths['lane_masks_dir'],
            sign_images_dir=None,
            sign_labels_file=None,
            mode='lane'
        )
        
        assert len(dataset) > 0
        sample = dataset[0]
        assert sample['image'] is not None
        assert sample['lane_mask'] is not None
        
        print(f"✓ 仅车道线模式测试通过")
    
    finally:
        shutil.rmtree(temp_dir)


def test_dataset_sign_only():
    """测试仅交通标志模式。"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        paths = create_dummy_dataset(temp_dir)
        
        dataset = MultiTaskDataset(
            lane_images_dir=None,
            lane_masks_dir=None,
            sign_images_dir=paths['sign_images_dir'],
            sign_labels_file=None,
            mode='sign'
        )
        
        assert len(dataset) > 0
        sample = dataset[0]
        assert sample['image'] is not None
        assert sample['sign_label'] is not None
        
        print(f"✓ 仅交通标志模式测试通过")
    
    finally:
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    # 运行测试
    test_dataset_creation()
    test_dataset_lane_only()
    test_dataset_sign_only()
    print("\n所有数据集测试通过! ✓")
