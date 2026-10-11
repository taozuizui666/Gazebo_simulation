# 仿真环境安装与运行

以下命令在 `Gazebo_simulation` 目录执行。返回 [项目首页](../README.md)。

## 启动

本工程使用 Ubuntu 22.04、ROS 2 Humble 和 Gazebo Classic 11。
Python 训练依赖不包含 ROS 2 或 Gazebo。补齐仿真和构建工具：

```bash
sudo apt install gazebo ros-humble-gazebo-ros-pkgs python3-colcon-common-extensions python3-rosdep
sudo rosdep init  # 只需首次初始化，已存在配置时不用重复执行
rosdep update --rosdistro humble
```

从本目录加载环境、检查依赖并构建：

```bash
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.bash
rosdep install --from-paths . --ignore-src --rosdistro humble -y
colcon build --symlink-install --packages-select my_gazebo_launch
source install/setup.bash
ros2 launch my_gazebo_launch train_world.launch.py
```

这台机器的 `~/.bashrc` 已加载 ROS 2 和 Gazebo，打开新终端时会自动生效；
工程的 `install/setup.bash` 仍需按上面加载。
启动脚本使用 Humble 官方 Gazebo 入口和 YAML 参数文件，并在退出时清理参数文件。
世界和地面、光源模型均可从本地加载。

只运行仿真后台、关闭图形界面：

```bash
ros2 launch my_gazebo_launch train_world.launch.py gui:=false
```

车辆默认停车。另一已 source 的终端持续发标签：

```bash
ros2 topic pub --rate 10 /steering_label std_msgs/msg/Int32 '{data: 7}'
```

发 0 或 14 检查左右转；停止发布后约 500 ms 停车。显式停车和查看输入：

```bash
ros2 topic pub --once /steering_label std_msgs/msg/Int32 '{data: 200}'
ros2 topic echo /wheel_targets_mps
ros2 topic echo /lidar_distances
```

统一使用 `train_world.launch.py`。可选择世界和初始位置：

```bash
ros2 launch my_gazebo_launch train_world.launch.py world:=assets/worlds/train_world2.sdf x:=0 y:=0 yaw:=0
```

默认出生点在迷宫内，持续转弯可能碰墙。单独验证轮速可在迷宫外的空旷位置启动：

```bash
ros2 launch my_gazebo_launch train_world.launch.py gui:=false x:=5 y:=0 yaw:=3.141592653589793
```

若将配置放在实车 PID 工程，可直接读该文件：

```bash
ros2 launch my_gazebo_launch train_world.launch.py config_header:=/absolute/path/vehicle_config.h
```

构建会安装世界及模型资源。节点使用仿真时钟，重置后清除旧驾驶命令。

## 本机运行验证（2026-10-09）

已通过 `rosdep check`、`colcon build --symlink-install` 和 `python3 -m pip check`。
Gazebo 11.10.2 图形界面能够运行，世界、车辆、差速插件及雷达插件均成功加载。
在上面的空旷位置，`/odom` 实测平均速度为：

| 标签 | 前进 m/s | 转向 rad/s |
| --- | --- | --- |
| 7 | 0.10000 | 0.00000 |
| 0 | 0.10075 | 1.08479 |
| 14 | 0.09919 | -1.08335 |

原始扫描 1450 点、仿真时间频率约 5.495 Hz；转换后 200 个毫米整数，
未采集扇区全零，CSV 为 201 列。超时、非法标签和仿真时间重置均会停车。
测试期间使用独立 ROS domain，测试结束后已关闭测试进程。

当时图形界面的 OpenGL 渲染器为 `llvmpipe`（CPU 软件渲染），当时尚未配置 GPU 加速。
数据采集或训练只需后台时可用 `gui:=false` 减少界面开销。
