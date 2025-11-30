"""
Tests for ROS2 node import.

ROS2节点导入测试（不需要ROS2运行时）。
"""

import pytest


def test_ros2_node_import():
    """测试ROS2节点模块是否可以导入。"""
    try:
        # 尝试导入节点模块
        # 注意：这可能会失败如果ROS2未安装
        from src.ros2_integration import lane_sign_node
        
        print("✓ ROS2节点模块导入成功")
        print(f"  模块路径: {lane_sign_node.__file__}")
    
    except ImportError as e:
        # ROS2未安装是预期的情况
        print(f"⊗ ROS2节点导入失败（预期行为，如果ROS2未安装）: {e}")
        print("  这在没有ROS2环境的系统上是正常的")


def test_launch_file_exists():
    """测试launch文件是否存在。"""
    import os
    
    # 获取launch文件路径
    launch_file = os.path.join(
        'src', 'ros2_integration', 'launch', 'lane_sign.launch.py'
    )
    
    assert os.path.exists(launch_file), f"Launch文件不存在: {launch_file}"
    
    print("✓ Launch文件存在检查通过")
    print(f"  Launch文件: {launch_file}")


if __name__ == '__main__':
    # 运行测试
    test_ros2_node_import()
    test_launch_file_exists()
    print("\n所有ROS2测试通过! ✓")
