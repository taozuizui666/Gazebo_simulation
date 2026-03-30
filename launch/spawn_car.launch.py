from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

def generate_launch_description():

    gazebo = ExecuteProcess(
        cmd=[
            'gazebo',
            '--verbose',
            # '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so'
        ],
        output='screen'
    )

    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', '/home/zui/ros2_ws/src/my_gazebo_launch/urdf/simple_car1.urdf',
            '-entity', 'base_link',
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        TimerAction(period=3.0, actions=[spawn])
    ])
