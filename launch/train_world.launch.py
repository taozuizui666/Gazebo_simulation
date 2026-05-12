from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch_ros.actions import Node
import os

def generate_launch_description():
    # 获取包的路径
    pkg_share = '/home/zui/ros2_ws/src/my_gazebo_launch'
    # 使用包含 Building Editor 环境的世界文件
    world_path = os.path.join(pkg_share, 'sdf/train_world3.sdf')
    urdf_path = os.path.join(pkg_share, 'urdf/simple_car1.urdf')
    
    # 设置 GAZEBO_MODEL_PATH 环境变量，确保能找到模型
    set_gazebo_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=pkg_share + '/models:' + pkg_share + '/sdf'
    )
    
    # 启动 Gazebo 并加载世界文件
    gazebo = ExecuteProcess(
        cmd=[
            'gazebo',
            '--verbose',
            '-s', 'libgazebo_ros_factory.so',
            world_path
        ],
        output='screen'
    )

    # 生成小车到世界中
    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'my_car',
            '-file', urdf_path,
            # 修改下面的数字来调整小车初始位置
            '-x', '-0.931026',   # X 轴位置
            '-y', '-1.305602',   # Y 轴位置
            '-z', '0.1',   # Z 轴高度（略高于地面，避免陷入）
            '-R', '0.0',   # Roll 旋转
            '-P', '0.0',   # Pitch 旋转
            '-Y', '0.0',   # Yaw 朝向
        ],
        output='screen'
    )

    # 发布机器人描述
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': open(urdf_path).read()}],
        output='screen'
    )

    return LaunchDescription([
        set_gazebo_model_path,
        robot_state_publisher,
        gazebo,
        TimerAction(period=5.0, actions=[spawn])  # 增加延迟到 5 秒，确保 ROS2 网络完全初始化
    ])
