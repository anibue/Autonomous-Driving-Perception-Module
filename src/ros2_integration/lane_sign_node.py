"""
ROS2 node for lane detection and traffic sign recognition.

ROS2感知节点,实时执行车道线检测和交通标志识别。
"""

import os
import argparse
from typing import Tuple

import numpy as np
import cv2
from cv_bridge import CvBridge
import onnxruntime as ort

# ROS2导入
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
except ImportError:
    print("警告: ROS2 Python包未安装。请在ROS2环境中运行此节点。")
    raise

from src.training.utils import load_config


class LaneSignPerceptionNode(Node):
    """
    车道线检测和交通标志识别感知节点。
    
    订阅相机图像话题,使用ONNX模型进行推理,
    发布车道线分割掩码和交通标志类别。
    
    Attributes:
        onnx_session: ONNX Runtime推理会话
        bridge: ROS-OpenCV桥接器
        config: 配置字典
        image_sub: 图像订阅者
        lane_mask_pub: 车道线掩码发布者
        sign_label_pub: 交通标志标签发布者
    """
    
    def __init__(
        self,
        onnx_model_path: str,
        config_path: str
    ):
        """
        初始化感知节点。
        
        Args:
            onnx_model_path: ONNX模型文件路径
            config_path: 配置文件路径
        """
        super().__init__('lane_sign_perception_node')
        
        # 加载配置
        self.config = load_config(config_path)
        ros2_config = self.config.get('ros2', {})
        training_config = self.config.get('training', {})
        
        # 图像尺寸
        self.img_height = training_config.get('img_height', 256)
        self.img_width = training_config.get('img_width', 512)
        
        # 归一化参数
        aug_config = self.config.get('augmentation', {}).get('val', {})
        norm_config = aug_config.get('normalize', {})
        self.mean = np.array(norm_config.get('mean', [0.485, 0.456, 0.406])).reshape(1, 1, 3)
        self.std = np.array(norm_config.get('std', [0.229, 0.224, 0.225])).reshape(1, 1, 3)
        
        # 创建ONNX Runtime会话
        self.get_logger().info(f'加载ONNX模型: {onnx_model_path}')
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 选择执行提供者
        providers = ['CPUExecutionProvider']
        if ros2_config.get('use_gpu', False):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.onnx_session = ort.InferenceSession(
            onnx_model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        self.get_logger().info(f'使用执行提供者: {self.onnx_session.get_providers()}')
        
        # 获取模型输入输出名称
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_names = [output.name for output in self.onnx_session.get_outputs()]
        
        # CvBridge用于ROS图像消息转换
        self.bridge = CvBridge()
        
        # 配置QoS
        qos_depth = ros2_config.get('qos_depth', 10)
        qos_reliability = ros2_config.get('qos_reliability', 'reliable')
        
        qos_profile = QoSProfile(
            depth=qos_depth,
            reliability=ReliabilityPolicy.RELIABLE if qos_reliability == 'reliable' 
                       else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # 订阅输入图像话题
        input_topic = ros2_config.get('input_image_topic', '/camera/image_raw')
        self.image_sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            qos_profile
        )
        
        self.get_logger().info(f'订阅图像话题: {input_topic}')
        
        # 发布车道线掩码
        lane_mask_topic = ros2_config.get('lane_mask_topic', '/perception/lane_mask')
        self.lane_mask_pub = self.create_publisher(
            Image,
            lane_mask_topic,
            qos_profile
        )
        
        self.get_logger().info(f'发布车道线掩码到: {lane_mask_topic}')
        
        # 发布交通标志标签
        sign_label_topic = ros2_config.get('sign_label_topic', '/perception/sign_label')
        self.sign_label_pub = self.create_publisher(
            String,
            sign_label_topic,
            qos_profile
        )
        
        self.get_logger().info(f'发布交通标志标签到: {sign_label_topic}')
        
        # 统计信息
        self.frame_count = 0
        
        self.get_logger().info('感知节点初始化完成')
    
    def preprocess_image(self, cv_image: np.ndarray) -> np.ndarray:
        """
        预处理图像用于模型推理。
        
        Args:
            cv_image: OpenCV图像 (BGR格式)
            
        Returns:
            预处理后的图像数组 [1, 3, H, W]
        """
        # BGR转RGB
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        image = cv2.resize(image, (self.img_width, self.img_height))
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # 转换为CHW格式并添加批次维度
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def postprocess_segmentation(
        self,
        lane_logits: np.ndarray,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        后处理分割输出。
        
        Args:
            lane_logits: 分割logits [1, C, H, W]
            original_size: 原始图像尺寸 (height, width)
            
        Returns:
            分割掩码 [H, W]
        """
        # 获取预测类别
        pred_mask = np.argmax(lane_logits, axis=1)[0]  # [H, W]
        
        # 调整到原始尺寸
        pred_mask = cv2.resize(
            pred_mask.astype(np.uint8),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        return pred_mask
    
    def postprocess_classification(
        self,
        sign_logits: np.ndarray
    ) -> Tuple[int, float]:
        """
        后处理分类输出。
        
        Args:
            sign_logits: 分类logits [1, num_classes]
            
        Returns:
            Tuple[预测类别, 置信度]
        """
        # 应用softmax
        exp_logits = np.exp(sign_logits - np.max(sign_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # 获取top-1预测
        pred_class = np.argmax(probs)
        confidence = probs[0, pred_class]
        
        return int(pred_class), float(confidence)
    
    def image_callback(self, msg: Image):
        """
        图像消息回调函数。
        
        接收图像消息,执行推理,发布结果。
        
        Args:
            msg: ROS图像消息
        """
        try:
            # 将ROS图像转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            original_size = (cv_image.shape[0], cv_image.shape[1])
            
            # 预处理
            input_data = self.preprocess_image(cv_image)
            
            # 推理
            outputs = self.onnx_session.run(
                self.output_names,
                {self.input_name: input_data}
            )
            
            lane_logits = outputs[0]  # [1, C, H, W]
            sign_logits = outputs[1]  # [1, num_classes]
            
            # 后处理车道线分割
            lane_mask = self.postprocess_segmentation(lane_logits, original_size)
            
            # 发布车道线掩码
            mask_msg = self.bridge.cv2_to_imgmsg(
                (lane_mask * 255).astype(np.uint8),
                encoding='mono8'
            )
            mask_msg.header = msg.header
            self.lane_mask_pub.publish(mask_msg)
            
            # 后处理交通标志分类
            sign_class, confidence = self.postprocess_classification(sign_logits)
            
            # 发布交通标志标签
            sign_msg = String()
            sign_msg.data = f"class={sign_class},confidence={confidence:.4f}"
            self.sign_label_pub.publish(sign_msg)
            
            # 更新统计
            self.frame_count += 1
            
            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f'已处理 {self.frame_count} 帧 | '
                    f'标志类别: {sign_class} (置信度: {confidence:.2f})'
                )
        
        except Exception as e:
            self.get_logger().error(f'处理图像时出错: {str(e)}')


def main(args=None):
    """主函数。"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ROS2感知节点')
    parser.add_argument(
        '--onnx-model',
        type=str,
        default='outputs/multitask_model.onnx',
        help='ONNX模型路径'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_example.yaml',
        help='配置文件路径'
    )
    
    parsed_args, unknown = parser.parse_known_args()
    
    # 初始化ROS2
    rclpy.init(args=args)
    
    try:
        # 创建节点
        node = LaneSignPerceptionNode(
            onnx_model_path=parsed_args.onnx_model,
            config_path=parsed_args.config
        )
        
        # 运行节点
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass
    finally:
        # 清理
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
