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
| [results/](results/README.md) | 地图预览、仿真截图、训练/评估图及 PDF，按批次保存 |
| [tools/](tools/README.md) | 批量生成环形地图、俯视预览及墙高编辑工具 |
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
[assets/worlds/simple/train_world3.sdf](assets/worlds/simple/train_world3.sdf)，车辆模板为
[assets/urdf/simple_car1.urdf](assets/urdf/simple_car1.urdf)。

安装依赖、切换地图、后台运行及控制命令见 [仿真使用说明](docs/simulation.md)。
轮速公式、实车 PID 接口、雷达列格式及标定范围见 [车辆与雷达说明](docs/vehicle_and_lidar.md)。

## 多地图训练入口

```bash
cd /home/zui/dissertation/Gazebo_simulation

source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.bash
source install/setup.bash

# 检查配置，并显示将从哪个模型续训
python rl_training/privileged/train_multi_map.py --check

# 开始训练
python rl_training/privileged/train_multi_map.py
```

检查通过后再执行训练命令。并行数量、控制步长、地图列表、出生点扰动、PPO 参数和时间预算
统一在 `rl_training/privileged/train_multi_map.py` 顶部修改，详见 [多地图训练说明](rl_training/privileged/README.md#多地图联合训练)。

## 两类 RL

| 模型 | 代码 | 已验证结果 |
| --- | --- | --- |
| 雷达输入 PPO | [lidar/train.py](rl_training/lidar/train.py) | [257,276 步，0.1 m/s](rl_models/lidar/README.md) |
| 特权环境场 PPO | [privileged/train.py](rl_training/privileged/train.py) | [2,392,588 步，0.2 m/s](rl_models/privileged/README.md) |

### Gazebo 可视化观看

使用 [privileged/visualize.py](rl_training/privileged/visualize.py) 观看已验证特权模型控制小车。
脚本自动加载 ROS、Gazebo 和项目环境，按地图列表依次打开 Gazebo。
以下命令在 `Gazebo_simulation` 目录执行：

```bash
# 检查模型、地图和配置，不启动仿真
python rl_training/privileged/visualize.py --check

# 打开 Gazebo 观看
python rl_training/privileged/visualize.py
```

与 `train_multi_map.py` 一样，配置集中在脚本顶部：`MODEL` 指定模型，`WORLDS` 指定地图列表，
`EPISODES_PER_MAP` 和 `MAX_STEPS_PER_EPISODE` 控制观看回合数与最大步数，
`POSITION_JITTER_M` 和 `YAW_JITTER_DEG` 控制出生点扰动。
默认使用 `latest_validated/model.zip`，在 `train_world3.sdf` 观看一回合、最多约 3 分钟仿真时间；
失败会提前结束。出生点和工作边界读取地图同名 JSON。
每张地图的评估报告与轨迹保存到 `OUTPUT_DIR` 下独立的 `evaluation_*` 目录。

### 训练与评估

评估和续训自动读取模型随附车辆配置。新训练二选一：

```bash
python rl_training/lidar/train.py
python rl_training/privileged/train.py
```

两条默认训练命令使用同一组 domain/端口，应分别运行。
快速生成多张地图并选择其中一张训练：

```bash
python tools/generate_maps.py --count 6
python rl_training/privileged/train.py --world assets/worlds/simple/rounded_rectangle_001.sdf
```

`--world` 省略时使用配置区顶部的默认地图；出生点和工作边界随地图读取同名 JSON。
生成参数位于脚本顶部，命令行可覆盖，详见 [地图生成工具](tools/README.md)。
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
