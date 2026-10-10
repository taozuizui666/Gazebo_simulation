# 固定速度小车仿真（develop_1008）

这一版实现可修改标签、仿真/实车共用轮速目标、训练格式的雷达观测。
电脑使用 ROS 2 + Gazebo Classic，实车后续使用 Teensy 编码器轮速 PID。
不依赖树莓派、UDP 或 PWM 节点。现已加入 PPO 持续避障巡航训练，使用独立的
训练/评估仿真，并提供本地与 GitHub Pages 手机进度页。自动采集 RL 数据留待后续。

## 修改参数

优先修改 [my_gazebo_launch/vehicle_config.h](my_gazebo_launch/vehicle_config.h)
顶部的宏。Python 直接读取这份 C++ 头文件，启动时配置车辆和传感器。

| 宏 | 默认值 | 含义 |
| --- | --- | --- |
| `AICAR_LABEL_COUNT` | 15 | 标签 0～14；直行自动为数量 / 2，即 7 |
| `AICAR_FORWARD_SPEED_MPS` | 0.1 | 固定的左右轮平均速度，单位 m/s |
| `AICAR_MAX_WHEEL_SPEED_RATIO` | 0.6 | 最大转向时左右轮相对基础速度的增减比例 |
| `AICAR_WHEEL_RADIUS_M` | 0.0325 | 轮半径，沿用原模型，需与实车一致 |
| `AICAR_WHEEL_SEPARATION_M` | 0.11 | 左右轮中心间距，沿用原模型，需与实车一致 |
| `AICAR_LIDAR_POINT_COUNT` | 200 | 模型输入距离列数 |
| `AICAR_LIDAR_MAX_DISTANCE_MM` | 10000 | 有效距离上界，不包含该值 |
| `AICAR_LIDAR_FRONT_SECTOR_ONLY` | 1 | 仅保留现有实车采集扇区 |
| `AICAR_LIDAR_MOUNT_YAW_DEG` | 0 | 雷达零度方向相对车头的逆时针夹角 |
| `AICAR_LIDAR_SCAN_HZ` | 5.5 | A1M8 手册典型频率，实际应测量后调整 |
| `AICAR_LIDAR_RAW_SAMPLES` | 1450 | 原始射线数，转换后仍是 200 桶 |
| `AICAR_LIDAR_NOISE_STDDEV_M` | 0.01 | 沿用旧模型的噪声标准差，尚未实测标定 |
| `AICAR_COMMAND_TIMEOUT_MS` | 500 | 未持续收到指令时停车，单位为仿真时间 ms |
| `AICAR_SIM_TIME_STEP_SEC` | 0.001 | 仿真物理步长，RL 按整数步推进 |

标签数量使用奇数（3～199），中间自动直行：0 最大左转，最后一个最大右转。
停车 200 是独立控制指令，不是训练标签。标签数量增加只使档位更密，
最大转向幅度由 `MAX_WHEEL_SPEED_RATIO` 调整。
整数角度分桶要求距离桶数为 4 的倍数且不超过 360。

文件、话题和初始位置在 `my_gazebo_launch/config.py` 顶部。
`urdf/simple_car1.urdf` 是几何模板，启动时由共享宏生成实际加载的 URDF。
不再使用旧 `/home/zui/ros2_ws` 中的配置。

## 实车 PID 与仿真如何关联

设基础速度为 v，归一化转向为 s，轮速增减比例为 k：

```text
s = (直行标签 - 当前标签) / 直行标签
左轮目标速度 = v - v*k*s
右轮目标速度 = v + v*k*s
前进速度 = (左轮 + 右轮) / 2 = v
转向角速度 = (右轮 - 左轮) / 轮距
```

| 标签 | 左轮 m/s | 右轮 m/s | 含义 |
| --- | --- | --- | --- |
| 0 | 0.04 | 0.16 | 最大左转 |
| 7 | 0.10 | 0.10 | 直行 |
| 14 | 0.16 | 0.04 | 最大右转 |

后续 PID 直接包含同一份头文件：

```cpp
#include "vehicle_config.h"  // 指向共享源文件，不手工维护另一套数值
const AiCar::WheelTargets target = AiCar::wheelTargets(command);
// PID 用 m/s：target.left_mps / target.right_mps
// PID 用 rad/s：target.left_rad_s / target.right_rad_s
// PID 用 RPM：target.left_rpm / target.right_rpm
```

当前目录结构下，Teensy 的 selfdrive/main 或 collect_data/main 可用：

```cpp
#include "../../../../Gazebo_simulation/my_gazebo_launch/vehicle_config.h"
```

Gazebo 只用轮速接口，不模拟电流、PWM、编码器脉冲或 PID 参数。
`0.1 m/s` 是目标轮速对应的前进速度，实际运动仍可能受打滑等影响，
可看 `/odom`。实车旧 `CAR_SPEED=100` 是 PWM，不能换算成 `0.1 m/s`。
本次未替换实车旧 PWM 程序，PID 尚待后续实现。

## 雷达输入协议

依据 `hardware/RP_Lidar_files/lidar protocol.pdf` 第 15～16 页和 Arduino
`RPLidarDriver`：A1 原始角度顺时针增加，单位为度；距离为毫米，
协议精度为 1/64 度、1/4 mm。`lidar_file.pdf` 第 9 页给出 A1M8 的
0.15～12 m 典型量程和 5.5 Hz 典型扫描频率。

用户已确认型号为 A1M8。手册第 9 页的量程下限栏是 TBD，0.15～12 m 位于
典型值栏，不能解读为驱动必须删除所有 15 cm 内的有效点。默认仿真下界仍按
该典型值设置为 0.15 m，可通过 `AICAR_LIDAR_MIN_RANGE_M` 修改；转换器遵循
扫描自身的有效量程，没有再用配置下界额外过滤有效近点。

Gazebo `/scan_raw` 保持 ROS 标准的逆时针弧度、米，仅用于调试。
RL/机器学习模型使用 `/lidar_distances`：

```text
std_msgs/UInt16MultiArray
data = [d0, d1, ..., d199]  # 200 个毫米整数，列顺序与训练一致
```

转换为：ROS 角度转顺时针 → A1 q6/q2 量化 → 角度丢弃小数 →
`index = int((整数角度 / 360.0f) * N)` → 毫米整数截断。
整圈按原始顺时针角度处理，桶内保留最后一个有效点。

默认保留 **0～49、151～199**，**50～150** 全为零。
零表示无效/未采集；inf、NaN、盲区和大于等于 10000 mm 的点不作为有效点。
不会把未命中障碍物伪装成 9999 mm。每次扫描重建数组，不补入上一帧。

后续保存训练行可调用纯函数：

```python
from my_gazebo_launch.config import load_config
from my_gazebo_launch.lidar_processing import training_row
cfg = load_config()
row = training_row(distances, label, cfg)
# d0,d1,...,d199,label\n：201 列，无表头、无末尾逗号
```

当前 `software/MLcode/train_model.py` 可接受本版 0～14 的行，距离配置一致。
若修改 N、距离上界、扇区，需同步训练脚本顶部配置。
当前训练器最大标签为 15，将来增加到 17 个及以上需扩展其标签检查。
旧数据的 10 直行与本版 7 直行含义不同，不能直接混用；
旧 PWM 固件不能直接使用本版模型。本阶段未修改训练器或生成模型文件。

驱动只能确定雷达自身零度，不能推断它装在车上时是否朝车头。
默认零度朝车头，实物安装偏转时修改 `LIDAR_MOUNT_YAW_DEG`。
本版使用整圈观测，原实车采集是 120 ms 滚动旧值缓存，未来应同步为单圈有效点处理。
格式一致不代表实测误差、安装朝向和扫描时序已经全部标定。

## 无 ROS 预览

在 `Gazebo_simulation` 内：

```bash
python -m my_gazebo_launch.preview --label 7
python -m my_gazebo_launch.preview --label 0
python -m my_gazebo_launch.preview --urdf-output /tmp/aicar.urdf
python -m my_gazebo_launch.preview --label 7 --csv-row
```

最后一条输出 1 m 人工示例行，只检查格式，不写入 dataset 或覆盖模型。

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

三个启动入口共用配置。可选择世界和初始位置：

```bash
ros2 launch my_gazebo_launch train_world.launch.py world:=sdf/train_world2.sdf x:=0 y:=0 yaw:=0
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

## 持续避障巡航 RL

训练入口是 [rl_training/train_rl.py](rl_training/train_rl.py)，详细配置和命令见
[rl_training/README.md](rl_training/README.md)。加载上述环境后可直接执行：

```bash
python3 -m pip install --user -r rl_training/requirements.txt
python3 rl_training/train_rl.py
```

脚本默认启动 4 个训练 Gazebo（domain 81/83/85/87，端口 11451/11453/11455/11457）
和 1 个独立评估 Gazebo（domain 82，端口 11452）。无需先启动手动仿真。
每个动作默认推进 0.2 秒仿真时间，网络更新时物理世界暂停。
策略只看与实车训练列一致的雷达输入；车身/轮子的接触及里程计只用于训练奖励。
撞墙或卡住后停车，下一回合明确恢复出生位置、朝向和零速度，不依赖雷达距离
阈值判定碰撞。加入转向变化、大转向和前方障碍风险惩罚，并检测持续低速、
小范围反复绕圈和越界。正常 750 步（150 秒仿真时间）作为采样截断。
训练加速时使用原生 Gazebo 插件核对每个物理步的接触，策略仍只接收雷达输入。
独立评估会保存每条轨迹、实际速度和失败原因，不只看奖励是否升高。

车辆参数仍直接读取共享头文件；训练地址、超参数、奖励和卡住阈值集中在
[rl_training/config.py](rl_training/config.py) 开头。每次训练建立新模型目录并记录
实际配置，保留旧模型。手机页面的启动和同步方式见下文。

本次在真实 Gazebo 后台验证了接触碰撞与复位、正常时限、独立评估隔离、卡住
检测、精确仿真步进，以及缺失新雷达时停止报错。已实际训练 PPO 并进行独立
长时巡航验证，模型、轨迹和实测结果见 [rl_models/README.md](rl_models/README.md)。
通过长时验证的权重保存在 `rl_models/policies/latest_validated/model.zip`，
普通训练检查点不会自动替换它。加载环境后观看：

```bash
python3 rl_training/train_rl.py --evaluate rl_models/policies/latest_validated/model.zip \
  --gui --eval-episodes 1 --episode-steps 1500
```

该命令以正常仿真时间运行一段 300 秒巡航，结束后保存报告并关闭本次窗口。
验证结论针对记录中的地图、车辆配置、噪声和起点，实车部署仍需按实际硬件标定。

## 手机查看训练进度

在本目录另开一个终端运行（进度服务无需加载 ROS 环境）：

```bash
python3 rl_training/publish_progress.py --serve --publish
```

电脑打开 <http://127.0.0.1:8765>；同一 Wi-Fi 的手机使用
`http://电脑局域网IP:8765`（用 `hostname -I` 查看 IP）。外网手机打开
[GitHub Pages 进度页](https://taozuizui666.github.io/dissertation/)。
只需局域网查看时省略 `--publish`。保持服务和电脑运行，后续训练会自动接入；
服务本身不启动训练。使用 Ctrl+C 结束前台服务。

页面适配手机，包含当前回合持续时间、实测速度和距离、碰撞/卡住发生时间、
完成率、训练曲线、回合筛选、独立验收以及模型参数。训练探索与固定模型验收
分别展示；“达到时限”表示至少行驶到该时间，并不表示到时发生碰撞。

本地每 5 秒更新；GitHub 进度数据约每 60 秒检查上传，手机每 15 秒检查。
网络和 GitHub 缓存可能增加延迟。页面标明快照时间和数据中断，停止训练时显示
历史结果。参数统一放在 [progress_config.py](rl_training/progress_config.py) 顶部。
发布方式和后台服务命令见 [RL 说明](rl_training/README.md#手机进度页)。

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

当前图形界面的 OpenGL 渲染器为 `llvmpipe`（CPU 软件渲染），GPU 加速尚未配置。
数据采集或训练只需后台时可用 `gui:=false` 减少界面开销。
