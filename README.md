# 固定速度小车仿真

ROS 2 Humble + Gazebo Classic 小车仿真，提供共享轮速目标、规范雷达观测、
两类 PPO 持续避障巡航训练，以及本地 / GitHub Pages 手机进度页。
当前车辆参数集中在 [vehicle_config.h](my_gazebo_launch/vehicle_config.h) 顶部。

## 目录导航

| 目录 | 用途 |
| --- | --- |
| [my_gazebo_launch/](my_gazebo_launch) | 仿真节点、轮速接口、雷达转换及共享车辆配置 |
| [launch/](launch) | 正式启动入口 `train_world.launch.py` |
| [assets/](assets/README.md) | 当前车辆、仿真地图和可复用模型 |
| [rl_training/](rl_training/README.md) | `lidar/`、`privileged/` 两类 RL 与共用逻辑、进度网页 |
| [rl_models/](rl_models/README.md) | 分类型保存的权重、日志、轨迹和验收结果 |
| [tools/](tools/README.md) | 墙高等资源编辑工具 |
| [docs/](docs/layout.md) | 目录说明、仿真使用、车辆和雷达协议 |
| [archive/](archive/README.md) | 旧车辆及网格、历史地图导出素材 |

`setup.py`、`setup.cfg`、`package.xml`、`resource/` 是 ROS 包配置；
`build/`、`install/`、`log/` 是构建生成目录，由 Git 忽略。
完整迁移对照见 [目录说明](docs/layout.md)。

## 构建与启动

以下命令均在 `Gazebo_simulation` 目录执行，使用已安装依赖的 Python 3：

```bash
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.bash
colcon build --symlink-install --packages-select my_gazebo_launch
source install/setup.bash
ros2 launch my_gazebo_launch train_world.launch.py
```

车辆默认停车；在另一个已加载环境的终端持续发布标签：

```bash
ros2 topic pub --rate 10 /steering_label std_msgs/msg/Int32 '{data: 7}'
```

当前标签 0～14，7 直行，0 最大左转、14 最大右转；停止发布后超时停车。
新仿真默认目标前进速度 0.2 m/s。正式地图为
[assets/worlds/train_world3.sdf](assets/worlds/train_world3.sdf)，车辆模板为
[assets/urdf/simple_car1.urdf](assets/urdf/simple_car1.urdf)。

安装依赖、切换地图、后台运行及控制命令见 [仿真使用说明](docs/simulation.md)。
轮速公式、实车 PID 接口、雷达列格式及标定范围见 [车辆与雷达说明](docs/vehicle_and_lidar.md)。

## 两类 RL

| 模型 | 代码 | 已验证结果 |
| --- | --- | --- |
| 雷达输入 PPO | [lidar/train.py](rl_training/lidar/train.py) | [257,276 步，0.1 m/s](rl_models/lidar/README.md) |
| 特权环境场 PPO | [privileged/train.py](rl_training/privileged/train.py) | [2,392,588 步，0.2 m/s](rl_models/privileged/README.md) |

观看已验证特权教师 3 分钟：

```bash
python rl_training/privileged/train.py \
  --evaluate rl_models/privileged/policies/latest_validated/model.zip \
  --gui --eval-episodes 1 --episode-steps 900
```

评估和续训自动读取模型随附车辆配置。新训练二选一：

```bash
python rl_training/lidar/train.py
python rl_training/privileged/train.py
```

两条默认训练命令使用同一组 domain/端口，应分别运行。
PPO 参数、续训、独立验收和时间预算见 [RL 使用说明](rl_training/README.md)。
现有验收限于记录中的地图、起点和车辆条件；后续教师动作与雷达配对采集尚未启动。

## 手机进度

进度服务独立于训练，无需加载 ROS。已有后台服务时直接访问网页：

- 本机：<http://127.0.0.1:8765>
- 同 Wi-Fi 手机：`http://电脑局域网IP:8765`
- 外网手机：[GitHub Pages](https://taozuizui666.github.io/dissertation/)

需要启动服务时执行：

```bash
python rl_training/monitoring/publish_progress.py --serve --publish
```

顶部选择模型类别，分别查看训练、碰撞 / 卡住时间、独立验收和模型参数。
服务管理与同步周期见 [进度页说明](rl_training/README.md#手机进度页)。
