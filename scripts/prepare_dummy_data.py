"""
Prepare dummy dataset for testing and quick start.

准备用于测试和快速开始的虚拟数据集。

该脚本会生成一个小型的虚拟数据集，包含：
- 车道线检测数据：图像+分割掩码
- 交通标志识别数据：图像+类别标签
"""

import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import csv


def create_lane_sample(image_size=(256, 512), lane_width=10):
    """
    创建一个带车道线的图像和对应的掩码。
    
    Args:
        image_size: 图像大小 (高, 宽)
        lane_width: 车道线宽度
    
    Returns:
        image: RGB图像 (PIL Image)
        mask: 二值掩码 (PIL Image)
    """
    height, width = image_size
    
    # 创建背景图像（灰色路面）
    image = Image.new('RGB', (width, height), color=(80, 80, 80))
    draw_img = ImageDraw.Draw(image)
    
    # 创建掩码
    mask = Image.new('L', (width, height), color=0)
    draw_mask = ImageDraw.Draw(mask)
    
    # 添加2-3条随机车道线
    num_lanes = np.random.randint(2, 4)
    
    for i in range(num_lanes):
        # 随机车道线位置
        x_start = np.random.randint(width // 4, 3 * width // 4)
        x_end = x_start + np.random.randint(-50, 50)
        
        # 绘制车道线（黄色或白色）
        color = (255, 255, 0) if i == 0 else (255, 255, 255)
        
        # 图像上绘制
        draw_img.line([(x_start, height), (x_end, 0)], fill=color, width=lane_width)
        
        # 掩码上绘制
        draw_mask.line([(x_start, height), (x_end, 0)], fill=255, width=lane_width)
    
    # 添加一些噪声到图像
    img_array = np.array(image)
    noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(img_array)
    
    return image, mask


def create_sign_sample(image_size=(64, 64), sign_class=0, num_classes=43):
    """
    创建一个交通标志图像。
    
    Args:
        image_size: 图像大小
        sign_class: 标志类别
        num_classes: 总类别数
    
    Returns:
        image: RGB图像 (PIL Image)
    """
    width, height = image_size
    
    # 创建背景
    image = Image.new('RGB', (width, height), color=(200, 200, 200))
    draw = ImageDraw.Draw(image)
    
    # 根据类别绘制不同的形状
    center = (width // 2, height // 2)
    radius = min(width, height) // 3
    
    # 选择颜色
    colors = [
        (255, 0, 0),    # 红色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (0, 255, 0),    # 绿色
    ]
    color = colors[sign_class % len(colors)]
    
    # 绘制不同的形状
    if sign_class % 4 == 0:  # 圆形
        draw.ellipse(
            [center[0] - radius, center[1] - radius,
             center[0] + radius, center[1] + radius],
            fill=color, outline=(0, 0, 0), width=3
        )
    elif sign_class % 4 == 1:  # 三角形
        points = [
            (center[0], center[1] - radius),
            (center[0] - radius, center[1] + radius),
            (center[0] + radius, center[1] + radius)
        ]
        draw.polygon(points, fill=color, outline=(0, 0, 0))
    elif sign_class % 4 == 2:  # 矩形
        draw.rectangle(
            [center[0] - radius, center[1] - radius,
             center[0] + radius, center[1] + radius],
            fill=color, outline=(0, 0, 0), width=3
        )
    else:  # 八边形
        angles = np.linspace(0, 2 * np.pi, 9)
        points = [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) 
                  for a in angles]
        draw.polygon(points, fill=color, outline=(0, 0, 0))
    
    return image


def main():
    """主函数：生成虚拟数据集。"""
    parser = argparse.ArgumentParser(description='生成虚拟数据集用于测试')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data',
        help='输出目录路径'
    )
    parser.add_argument(
        '--num_lane_samples',
        type=int,
        default=100,
        help='车道线样本数量'
    )
    parser.add_argument(
        '--num_sign_samples',
        type=int,
        default=200,
        help='交通标志样本数量'
    )
    parser.add_argument(
        '--num_sign_classes',
        type=int,
        default=43,
        help='交通标志类别数量'
    )
    parser.add_argument(
        '--lane_image_size',
        type=int,
        nargs=2,
        default=[256, 512],
        help='车道线图像大小 (高 宽)'
    )
    parser.add_argument(
        '--sign_image_size',
        type=int,
        nargs=2,
        default=[64, 64],
        help='交通标志图像大小 (高 宽)'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 生成车道线数据
    print(f"正在生成 {args.num_lane_samples} 个车道线样本...")
    
    lane_images_dir = output_dir / 'lane' / 'images'
    lane_masks_dir = output_dir / 'lane' / 'masks'
    lane_images_dir.mkdir(parents=True, exist_ok=True)
    lane_masks_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(args.num_lane_samples):
        image, mask = create_lane_sample(tuple(args.lane_image_size))
        
        image.save(lane_images_dir / f'lane_{i:04d}.jpg')
        mask.save(lane_masks_dir / f'lane_{i:04d}.png')
        
        if (i + 1) % 20 == 0:
            print(f"  已生成 {i + 1}/{args.num_lane_samples} 个车道线样本")
    
    print(f"✓ 车道线数据生成完成")
    print(f"  图像目录: {lane_images_dir}")
    print(f"  掩码目录: {lane_masks_dir}")
    
    # 2. 生成交通标志数据
    print(f"\n正在生成 {args.num_sign_samples} 个交通标志样本...")
    
    sign_base_dir = output_dir / 'sign'
    sign_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建CSV标签文件
    csv_path = sign_base_dir / 'labels.csv'
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'class_id'])
        
        for i in range(args.num_sign_samples):
            # 随机分配类别
            class_id = np.random.randint(0, args.num_sign_classes)
            
            # 创建类别目录（文件夹结构方式）
            class_dir = sign_base_dir / f'class_{class_id}'
            class_dir.mkdir(exist_ok=True)
            
            # 生成图像
            image = create_sign_sample(
                tuple(args.sign_image_size),
                class_id,
                args.num_sign_classes
            )
            
            filename = f'sign_{i:04d}.jpg'
            image.save(class_dir / filename)
            
            # 写入CSV
            writer.writerow([f'class_{class_id}/{filename}', class_id])
            
            if (i + 1) % 50 == 0:
                print(f"  已生成 {i + 1}/{args.num_sign_samples} 个交通标志样本")
    
    print(f"✓ 交通标志数据生成完成")
    print(f"  数据目录: {sign_base_dir}")
    print(f"  标签文件: {csv_path}")
    print(f"  类别数量: {args.num_sign_classes}")
    
    # 3. 生成数据集统计信息
    print(f"\n数据集统计信息:")
    print(f"  总样本数: {args.num_lane_samples + args.num_sign_samples}")
    print(f"  车道线样本: {args.num_lane_samples}")
    print(f"  交通标志样本: {args.num_sign_samples}")
    print(f"  交通标志类别: {args.num_sign_classes}")
    print(f"\n数据集已保存到: {output_dir.absolute()}")
    
    # 4. 创建数据集划分建议
    print(f"\n建议的数据集划分:")
    print(f"  训练集: 70% ({int(args.num_lane_samples * 0.7)} 车道线, "
          f"{int(args.num_sign_samples * 0.7)} 标志)")
    print(f"  验证集: 15% ({int(args.num_lane_samples * 0.15)} 车道线, "
          f"{int(args.num_sign_samples * 0.15)} 标志)")
    print(f"  测试集: 15% ({int(args.num_lane_samples * 0.15)} 车道线, "
          f"{int(args.num_sign_samples * 0.15)} 标志)")


if __name__ == '__main__':
    main()
