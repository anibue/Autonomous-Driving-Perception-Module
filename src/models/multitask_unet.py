"""
Multi-Task UNet model for lane detection and traffic sign recognition.

多任务UNet模型，用于车道线检测（分割）和交通标志识别（分类）。
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    卷积块：两个3x3卷积 + BatchNorm + ReLU。
    
    这是UNet架构中的基本构建单元，每个卷积块包含：
    - Conv3x3 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm -> ReLU
    
    Attributes:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        初始化卷积块。
        
        Args:
            in_channels: 输入特征通道数
            out_channels: 输出特征通道数
        """
        super().__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            
        Returns:
            输出张量 [B, C_out, H, W]
        """
        # 第一个卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 第二个卷积
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


class EncoderBlock(nn.Module):
    """
    编码器块：卷积块 + 最大池化下采样。
    
    用于提取特征并降低空间分辨率，同时增加通道数。
    
    Attributes:
        conv_block: 卷积块
        pool: 最大池化层
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        初始化编码器块。
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super().__init__()
        
        # 卷积块用于特征提取
        self.conv_block = ConvBlock(in_channels, out_channels)
        
        # 最大池化用于下采样（尺寸减半）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            
        Returns:
            Tuple[特征图, 池化后的特征图]:
            - 特征图用于跳跃连接 [B, C_out, H, W]
            - 池化后的特征图传递到下一层 [B, C_out, H/2, W/2]
        """
        # 卷积块提取特征
        features = self.conv_block(x)
        
        # 池化下采样
        pooled = self.pool(features)
        
        return features, pooled


class DecoderBlock(nn.Module):
    """
    解码器块：上采样 + 跳跃连接拼接 + 卷积块。
    
    用于恢复空间分辨率，并融合编码器的跳跃连接特征。
    
    Attributes:
        up: 上采样层（转置卷积）
        conv_block: 卷积块
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        初始化解码器块。
        
        Args:
            in_channels: 输入通道数（上采样前）
            out_channels: 输出通道数
        """
        super().__init__()
        
        # 转置卷积用于上采样（尺寸加倍）
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=2, stride=2
        )
        
        # 卷积块处理拼接后的特征
        # 输入通道数是上采样输出 + 跳跃连接，所以是 out_channels * 2
        self.conv_block = ConvBlock(out_channels * 2, out_channels)
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip_features: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            skip_features: 编码器跳跃连接特征 [B, C_out, H*2, W*2]
            
        Returns:
            输出张量 [B, C_out, H*2, W*2]
        """
        # 上采样
        x = self.up(x)
        
        # 处理尺寸不匹配的情况（由于池化/上采样可能导致奇数尺寸）
        diff_h = skip_features.size(2) - x.size(2)
        diff_w = skip_features.size(3) - x.size(3)
        
        # 使用填充调整尺寸
        x = F.pad(x, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2
        ])
        
        # 拼接跳跃连接特征
        x = torch.cat([skip_features, x], dim=1)
        
        # 卷积块处理
        x = self.conv_block(x)
        
        return x


class Encoder(nn.Module):
    """
    共享编码器网络。
    
    由多个编码器块组成，逐层提取特征并降低分辨率。
    输出包括各层的跳跃连接特征和最后的瓶颈特征。
    
    架构示意：
    Input -> Block1 -> Block2 -> Block3 -> Block4 (Bottleneck)
    
    Attributes:
        encoder_blocks: 编码器块列表
        bottleneck: 瓶颈层
    """
    
    def __init__(
        self, 
        in_channels: int = 3, 
        encoder_channels: List[int] = [64, 128, 256, 512]
    ):
        """
        初始化编码器。
        
        Args:
            in_channels: 输入图像通道数（RGB为3）
            encoder_channels: 各层编码器通道数列表
        """
        super().__init__()
        
        self.encoder_channels = encoder_channels
        
        # 创建编码器块（除了最后一层）
        self.encoder_blocks = nn.ModuleList()
        
        # 第一层：输入通道 -> 第一个编码器通道
        prev_channels = in_channels
        for i, out_channels in enumerate(encoder_channels[:-1]):
            self.encoder_blocks.append(
                EncoderBlock(prev_channels, out_channels)
            )
            prev_channels = out_channels
        
        # 瓶颈层（最深层，不进行池化）
        self.bottleneck = ConvBlock(
            encoder_channels[-2], 
            encoder_channels[-1]
        )
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播。
        
        Args:
            x: 输入图像张量 [B, 3, H, W]
            
        Returns:
            Tuple[瓶颈特征, 跳跃连接特征列表]:
            - 瓶颈特征：最深层特征 [B, C_bottle, H/8, W/8]
            - 跳跃连接特征列表：各编码器层的输出，用于解码器
        """
        skip_features = []
        
        # 通过编码器块
        for encoder_block in self.encoder_blocks:
            features, x = encoder_block(x)
            skip_features.append(features)
        
        # 瓶颈层
        bottleneck_features = self.bottleneck(x)
        
        return bottleneck_features, skip_features


class SegmentationDecoder(nn.Module):
    """
    分割解码器网络（UNet风格）。
    
    通过上采样和跳跃连接恢复空间分辨率，生成分割掩码。
    
    架构示意：
    Bottleneck -> DecoderBlock1 -> DecoderBlock2 -> DecoderBlock3 -> Output
                       |                |                |
                    Skip3           Skip2            Skip1
    
    Attributes:
        decoder_blocks: 解码器块列表
        final_conv: 最终输出卷积
    """
    
    def __init__(
        self, 
        encoder_channels: List[int] = [64, 128, 256, 512],
        num_classes: int = 2
    ):
        """
        初始化分割解码器。
        
        Args:
            encoder_channels: 编码器通道数列表（需要与编码器匹配）
            num_classes: 分割类别数
        """
        super().__init__()
        
        # 解码器通道数（与编码器相反的顺序）
        decoder_channels = encoder_channels[::-1]
        
        # 创建解码器块
        self.decoder_blocks = nn.ModuleList()
        
        # 从瓶颈开始，逐层上采样
        for i in range(len(decoder_channels) - 1):
            self.decoder_blocks.append(
                DecoderBlock(decoder_channels[i], decoder_channels[i + 1])
            )
        
        # 最终上采样（恢复到原始分辨率）
        self.final_up = nn.ConvTranspose2d(
            decoder_channels[-1], decoder_channels[-1],
            kernel_size=2, stride=2
        )
        
        # 最终1x1卷积，输出分割类别
        self.final_conv = nn.Conv2d(
            decoder_channels[-1], num_classes,
            kernel_size=1
        )
    
    def forward(
        self, 
        bottleneck_features: torch.Tensor,
        skip_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            bottleneck_features: 瓶颈特征 [B, C_bottle, H/16, W/16]
            skip_features: 跳跃连接特征列表（从浅到深排序）
            
        Returns:
            分割logits [B, num_classes, H, W]
        """
        x = bottleneck_features
        
        # 反转跳跃连接特征顺序（从深到浅）
        skip_features_reversed = skip_features[::-1]
        
        # 通过解码器块
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x, skip_features_reversed[i])
        
        # 最终上采样
        x = self.final_up(x)
        
        # 最终卷积输出
        x = self.final_conv(x)
        
        return x


class ClassificationHead(nn.Module):
    """
    分类头网络。
    
    使用全局平均池化和全连接层，将编码器瓶颈特征转换为分类logits。
    
    架构示意：
    Bottleneck -> GlobalAvgPool -> FC1 -> ReLU -> Dropout -> FC2 -> Logits
    
    Attributes:
        global_pool: 全局平均池化
        fc1: 第一个全连接层
        fc2: 输出全连接层
        dropout: Dropout层
    """
    
    def __init__(
        self, 
        in_features: int = 512,
        num_classes: int = 43,
        dropout_rate: float = 0.3
    ):
        """
        初始化分类头。
        
        Args:
            in_features: 输入特征维度（瓶颈通道数）
            num_classes: 分类类别数
            dropout_rate: Dropout比率
        """
        super().__init__()
        
        # 全局平均池化：将空间维度压缩为1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout用于防止过拟合
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, bottleneck_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            bottleneck_features: 瓶颈特征 [B, C, H, W]
            
        Returns:
            分类logits [B, num_classes]
        """
        # 全局平均池化
        x = self.global_pool(bottleneck_features)  # [B, C, 1, 1]
        
        # 展平
        x = x.view(x.size(0), -1)  # [B, C]
        
        # 第一个全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 输出层
        x = self.fc2(x)
        
        return x


class MultiTaskUNet(nn.Module):
    """
    多任务UNet模型。
    
    该模型采用共享编码器架构，同时支持：
    1. 车道线检测（语义分割任务）
    2. 交通标志识别（图像分类任务）
    
    架构概览：
                        ┌──────────────────┐
                        │   Input Image    │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Shared Encoder  │ ← 共享特征提取
                        └────────┬─────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                                     │
     ┌────────▼─────────┐               ┌──────────▼──────────┐
     │ Segmentation     │               │ Classification      │
     │ Decoder          │               │ Head                │
     └────────┬─────────┘               └──────────┬──────────┘
              │                                     │
     ┌────────▼─────────┐               ┌──────────▼──────────┐
     │ Lane Logits      │               │ Sign Logits         │
     │ [B, C, H, W]     │               │ [B, num_classes]    │
     └──────────────────┘               └─────────────────────┘
    
    Attributes:
        encoder: 共享编码器
        seg_decoder: 分割解码器
        cls_head: 分类头
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: List[int] = [64, 128, 256, 512],
        num_segmentation_classes: int = 2,
        num_sign_classes: int = 43,
        dropout_rate: float = 0.3
    ):
        """
        初始化多任务UNet模型。
        
        Args:
            in_channels: 输入图像通道数（RGB为3）
            encoder_channels: 编码器各层通道数
            num_segmentation_classes: 分割任务类别数（包含背景）
            num_sign_classes: 交通标志分类类别数
            dropout_rate: 分类头Dropout比率
        """
        super().__init__()
        
        # 保存配置参数
        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.num_segmentation_classes = num_segmentation_classes
        self.num_sign_classes = num_sign_classes
        
        # 初始化共享编码器
        self.encoder = Encoder(
            in_channels=in_channels,
            encoder_channels=encoder_channels
        )
        
        # 初始化分割解码器
        self.seg_decoder = SegmentationDecoder(
            encoder_channels=encoder_channels,
            num_classes=num_segmentation_classes
        )
        
        # 初始化分类头
        self.cls_head = ClassificationHead(
            in_features=encoder_channels[-1],  # 瓶颈通道数
            num_classes=num_sign_classes,
            dropout_rate=dropout_rate
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        初始化模型权重。
        
        使用Kaiming初始化卷积层权重，常数初始化BatchNorm层。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # Kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播。
        
        同时执行车道线分割和交通标志分类两个任务。
        
        Args:
            x: 输入图像张量 [B, 3, H, W]
            
        Returns:
            包含两个任务输出的字典：
            - 'lane_logits': 车道线分割logits [B, num_seg_classes, H, W]
            - 'sign_logits': 交通标志分类logits [B, num_sign_classes]
        """
        # 共享编码器提取特征
        bottleneck_features, skip_features = self.encoder(x)
        
        # 分割解码器生成车道线分割结果
        lane_logits = self.seg_decoder(bottleneck_features, skip_features)
        
        # 分类头生成交通标志分类结果
        sign_logits = self.cls_head(bottleneck_features)
        
        return {
            'lane_logits': lane_logits,
            'sign_logits': sign_logits
        }
    
    def get_lane_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅获取车道线分割预测。
        
        用于推理时只需要分割结果的场景。
        
        Args:
            x: 输入图像张量 [B, 3, H, W]
            
        Returns:
            车道线分割预测（经过argmax） [B, H, W]
        """
        outputs = self.forward(x)
        lane_logits = outputs['lane_logits']
        
        # 应用softmax并取argmax获得预测类别
        lane_pred = torch.argmax(lane_logits, dim=1)
        
        return lane_pred
    
    def get_sign_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅获取交通标志分类预测。
        
        用于推理时只需要分类结果的场景。
        
        Args:
            x: 输入图像张量 [B, 3, H, W]
            
        Returns:
            交通标志分类预测（类别索引） [B]
        """
        outputs = self.forward(x)
        sign_logits = outputs['sign_logits']
        
        # 取argmax获得预测类别
        sign_pred = torch.argmax(sign_logits, dim=1)
        
        return sign_pred


def create_model(config: Dict) -> MultiTaskUNet:
    """
    根据配置创建多任务UNet模型。
    
    从配置字典中读取模型参数并创建模型实例。
    
    Args:
        config: 配置字典，包含以下键：
            - model.encoder_channels: 编码器通道数列表
            - model.num_segmentation_classes: 分割类别数
            - model.num_sign_classes: 分类类别数
            - model.dropout_rate: Dropout比率
            
    Returns:
        初始化好的MultiTaskUNet模型
    """
    model_config = config.get('model', {})
    
    model = MultiTaskUNet(
        in_channels=3,
        encoder_channels=model_config.get('encoder_channels', [64, 128, 256, 512]),
        num_segmentation_classes=model_config.get('num_segmentation_classes', 2),
        num_sign_classes=model_config.get('num_sign_classes', 43),
        dropout_rate=model_config.get('dropout_rate', 0.3)
    )
    
    return model
