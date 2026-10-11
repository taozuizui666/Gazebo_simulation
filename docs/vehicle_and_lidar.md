# 车辆参数、轮速接口与雷达协议

以下命令在 `Gazebo_simulation` 目录执行。返回 [项目首页](../README.md)。

## 修改参数

优先修改 [my_gazebo_launch/vehicle_config.h](../my_gazebo_launch/vehicle_config.h)
顶部的宏。Python 直接读取这份 C++ 头文件，启动时配置车辆和传感器。

| 宏 | 默认值 | 含义 |
| --- | --- | --- |
| `AICAR_LABEL_COUNT` | 15 | 标签 0～14；直行自动为数量 / 2，即 7 |
| `AICAR_FORWARD_SPEED_MPS` | 0.2 | 固定的左右轮平均速度，单位 m/s |
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
`assets/urdf/simple_car1.urdf` 是几何模板，启动时由共享宏生成实际加载的 URDF。
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
| 0 | 0.08 | 0.32 | 最大左转 |
| 7 | 0.20 | 0.20 | 直行 |
| 14 | 0.32 | 0.08 | 最大右转 |

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
`0.2 m/s` 是当前目标轮速对应的前进速度，实际运动仍可能受打滑等影响，
可看 `/odom`。实车旧 `CAR_SPEED=100` 是 PWM，不能直接换算成 m/s。
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
雷达基线策略和后续随机森林采集使用 `/lidar_distances`；特权教师不使用该话题：

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

