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

## 随机森林驾驶

使用 [run_random_forest.py](run_random_forest.py) 观看训练好的随机森林控制小车。
默认加载 `../software/ML_model/simulation/randomForest.joblib`，在保留验证地图
`ellipse_002` 的第一个评估出生点启动。模型同目录需保留配套 `training_report.json`。
脚本自动加载 ROS/Gazebo/项目环境；首次使用先完成上面的构建。

```bash
cd /home/zui/dissertation/Gazebo_simulation

# 检查模型、雷达格式、标签和地图，并试推理；不启动 Gazebo
python run_random_forest.py --check

# 打开 Gazebo，Ctrl+C 停车并关闭本次仿真
python run_random_forest.py

# 指定模型和地图
python run_random_forest.py --model ../software/ML_model/simulation/randomForest.joblib --world assets/worlds/simple/train_world3.sdf

# 无窗口运行：收到第一帧有效雷达后，再运行 10 仿真秒
python run_random_forest.py --headless --seconds 10
```

模型路径、默认地图、车辆配置、ROS domain 和 Gazebo 端口集中在脚本顶部。
`--model` 相对路径以当前终端目录为基准，`--world` 相对路径以 `Gazebo_simulation` 为基准。
雷达 `/lidar_distances` 的毫米整数按保存的特征索引直接输入模型，零值保持不变；
预测的原始转向标签发送到 `/steering_label`，由现有控制器转换为左右轮速度。
无效或全零帧停车；缺帧后由现有控制器按 `COMMAND_TIMEOUT_MS` 超时停车。
仿真日志写入 `results/random_forest/gazebo.log`，每次运行覆盖。
单次驾驶用于 GUI 观察；添加 `--evaluate` 进入下面的批量评分模式。

### 随机森林闭环评分

```bash
python run_random_forest.py --evaluate --check
# 默认四张地图，每张 5 个起点，每回合最多 180 仿真秒
python run_random_forest.py --evaluate
# 短回合仅检查流程，不作为最终验收
python run_random_forest.py --evaluate --world assets/worlds/simple/ellipse_002.sdf --episodes 1 --eval-seconds 10
# 同一组出生点与种子下比较正常扫描和缺点扫描
python run_random_forest.py --evaluate --scenario normal --scenario dropout05 --scenario dropout10 --scenario sector
```

配置集中在 `run_random_forest.py` 的 `EVAL_*` 常量；评估固定使用无 GUI、暂停步进环境，
每步 0.2 仿真秒（不低于一个扫描周期），直接使用扫描处理后的 200 个毫米整数预测。
模型不接收地图或真实位置；真实位置仅用于评分。因此评估验证的是固定控制周期下的策略，
单次 GUI 驾驶中的异步调度延迟仍需单独测试。

每回合使用地图 JSON 的 `eval_spawn_poses`，超过其数量会循环使用；种子为 `SEED + 回合索引`。
默认 `normal` 不额外置零；`dropout05/10` 每帧独立置零有效扇区内 5%/10% 的点。
`sector` 每帧随机屏蔽前方连续 20 个角度桶（36°），是瞬时扇区缺失，不是整轮固定遮挡。
更换默认地图列表可增加最终测试地图，但不要将已经用于调参的地图称为未知测试地图。

碰撞沿用环境接触检测，排除地面及自身接触；卡住、低速、小范围绕圈、越界沿用 RL 环境阈值。
任何失败立即结束该回合。系统异常单独记录并中止批次，已有轨迹保留，未完成批次不能通过。

| 分项 | 满分 | 定义 |
| --- | ---: | --- |
| 无碰撞 | 40 | 无碰撞回合比例 × 40 |
| 任务成功 | 30 | 完成时长且无碰撞、卡住、低速、绕圈、越界的比例 × 30 |
| 行驶进度 | 20 | 有效检查点进度 / (目标速度 × 规定时长)，单回合封顶 1 |
| 安全净空 | 10 | 净空至少 0.05 m 的时间比例 × 10 |

各地图×扰动条件先独立评分，再等权平均。通过要求：整批完成、总分 ≥90、零碰撞、
每个地图×条件的成功率 ≥95%。这是项目实验标准，不是安全认证。

**进度适用范围**：当前四张简单凸环路。取 `train_spawn_poses` 的位置，按环路角度排序为检查点，
根据初始朝向确定方向；必须按顺序经过检查点才累加相邻检查点的直线距离。
它保守低估弯道弧长，未完成的末段不计分；短回合可能没有通过检查点，因此不能用短测分数验收。
不适用于死胡同、八字或分叉地图；这些地图需要单独定义路线和任务成功条件。
净空使用现有 GroundTruthMap 的保守车身圆盘到静态障碍距离，并按控制周期采样，
不是实际多边形车身的连续最小距离，圆盘半径也写入报告。

每次独立输出到 `results/random_forest/evaluation_时间_编号/`：
- `evaluation.json`：模型/地图 SHA-256、车辆和阈值、出生点、种子、逐回合结果、分项分数与验收。
- `地图/条件/episode_编号/steps.jsonl`：仿真时间、位置、动作、净空、碰撞、有效进度。
- 同目录 `trajectory.npy`、`trajectory.png/.pdf` 和仿真日志。

还记录首次碰撞时间、最小/第 5 百分位净空、动作切换次数、推理耗时第 95 百分位和速度达标率。
每 100 米碰撞数统计首碰即终止的碰撞回合，不将持续接触的多帧重复计数。


## 多地图训练入口

本轮 8 字形与直角矩形任务先阅读 [仿真与 RL 训练待办](../article/planning/experiment_data_plan.md)。
当前脚本仍是旧五图配置；先确认并实施配置修改，再按待办文档检查和启动。

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
现有验收限于记录中的地图、起点和车辆条件。

### 采集随机森林训练数据

```bash
# 在 Gazebo_simulation 目录运行；脚本自动加载 ROS/Gazebo 环境
python rl_training/privileged/collect_data.py --check
python rl_training/privileged/collect_data.py
# 临时关闭 GUI，优先采集速度
python rl_training/privileged/collect_data.py --headless
```

在脚本顶部设置 `GUI`、`GUI_REAL_TIME_FACTOR`、`WORLDS`、
`COLLECTION_SECONDS` 和 `COLLECTION_ROUNDS`。GUI 模式限制仿真速度、关闭阴影，
并根据地图边界设置默认俯视视角；
无 GUI 模式不限速，同时保留雷达与动作同步检查。
`SPAWN_JITTER_ENABLED` 开启出生位置/朝向扰动；`RECOVERY_ENABLED` 开启近墙纠正场景，
`RECOVERY_FRACTION=0.25` 表示约四分之一轮次从偏向墙壁的姿态开始。
`--check` 会检查全部出生计划；几何检查通过后仍需短程验证教师的纠偏表现。
采集雷达通过顶部 `LIDAR_CONFIG` 读取共享配置的扫描频率和原始点数，默认
5.5 Hz、364 点/圈，按实车标准 SCAN 模式建模；输出仍为 200 个雷达值加动作标签。
教师模型不修改；旧 1450 点/圈批次需保留原配置，新参数另开批次采集。
每张地图输出一份 raw 和一份 dropout CSV，分别保存到两个文件夹，并复制至
`../dataset/simulation/<批次>/`。元信息、时间戳和日志只留在 `results/datasets/`。
详见 [采集说明](rl_training/privileged/README.md#采集随机森林训练数据)。

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
