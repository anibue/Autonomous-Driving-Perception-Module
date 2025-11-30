"""
ROS2 launch file for lane and sign perception node.

ROS2启动文件,用于启动车道线和交通标志感知节点。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    生成启动描述。
    
    配置启动参数和节点。
    
    Returns:
        LaunchDescription: ROS2启动描述
    """
    # 声明启动参数
    # ONNX模型路径参数
    onnx_model_arg = DeclareLaunchArgument(
        'onnx_model_path',
        default_value='outputs/multitask_model.onnx',
        description='ONNX模型文件路径'
    )
    
    # 配置文件路径参数
    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value='configs/config_example.yaml',
        description='配置文件路径'
    )
    
    # 输入图像话题参数
    input_image_topic_arg = DeclareLaunchArgument(
        'input_image_topic',
        default_value='/camera/image_raw',
        description='输入相机图像话题名称'
    )
    
    # 车道线掩码输出话题参数
    lane_mask_topic_arg = DeclareLaunchArgument(
        'lane_mask_topic',
        default_value='/perception/lane_mask',
        description='车道线分割掩码输出话题名称'
    )
    
    # 交通标志标签输出话题参数
    sign_label_topic_arg = DeclareLaunchArgument(
        'sign_label_topic',
        default_value='/perception/sign_label',
        description='交通标志标签输出话题名称'
    )
    
    # 创建感知节点
    perception_node = Node(
        package='src.ros2_integration',
        executable='lane_sign_node',
        name='lane_sign_perception_node',
        output='screen',
        parameters=[
            {
                'onnx_model_path': LaunchConfiguration('onnx_model_path'),
                'config_path': LaunchConfiguration('config_path')
            }
        ],
        remappings=[
            ('/camera/image_raw', LaunchConfiguration('input_image_topic')),
            ('/perception/lane_mask', LaunchConfiguration('lane_mask_topic')),
            ('/perception/sign_label', LaunchConfiguration('sign_label_topic'))
        ]
    )
    
    # 构建启动描述
    return LaunchDescription([
        onnx_model_arg,
        config_path_arg,
        input_image_topic_arg,
        lane_mask_topic_arg,
        sign_label_topic_arg,
        perception_node
    ])
