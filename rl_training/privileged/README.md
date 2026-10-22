# 特权环境场 RL

这一类是使用仿真真值的 PPO 教师。输入为车身坐标系下的 17×17 地图有符号距离场，
加真实前进速度、角速度、上一动作，共 292 维。位姿用于裁剪局部地图，
不会把绝对坐标直接送入网络。训练与评估均关闭雷达，奖励也不读取雷达数据。

- 入口：[train.py](train.py)。
- 模型路径、网格、范围和几何安全间距：[config.py](config.py) 顶部。
- 地图解析与观测：[observation.py](observation.py)。
- 历史模型与验收结果：[rl_models/privileged](../../rl_models/privileged/README.md)。

在 `Gazebo_simulation` 目录加载 ROS 与项目环境后：

```bash
python rl_training/privileged/train.py
python rl_training/privileged/train.py \
  --evaluate rl_models/privileged/policies/latest_validated/model.zip \
  --gui --eval-episodes 1 --episode-steps 900
```

当前已验证教师为 2,392,588 步、0.2 m/s，来自六小时续训。

可视化入口 [visualize.py](visualize.py) 与 `train_multi_map.py` 一样，配置集中在文件顶部。
修改 `MODEL`、`WORLDS`、`EPISODES_PER_MAP`、`MAX_STEPS_PER_EPISODE` 和出生点扰动，
即可依次观看指定地图。默认打开 `train_world3.sdf`，观看一回合、最多 180 仿真秒。
脚本自动加载 ROS、Gazebo 和项目环境；每张地图的报告和轨迹保存到 `OUTPUT_DIR` 下独立的 `evaluation_*` 目录。

```bash
python rl_training/privileged/visualize.py --check  # 只检查，不启动仿真
python rl_training/privileged/visualize.py          # 打开 Gazebo 观看
```

地图解析目前支持静态、水平放置的 box 碰撞体，不支持的几何会报错。
更换地图使用同名 JSON 的出生点和工作边界，并重新验证泛化表现。

## 采集随机森林训练数据

入口 [collect_data.py](collect_data.py) 自动加载 ROS/Gazebo 环境。配置集中在文件顶部：
模型、地图列表、每轮秒数、每图轮数、出生扰动、输出路径、GUI、置零比例和随机种子。
当前使用已验证教师，按列表依次采集四张地图，每张 20 轮，每轮最多 3600 仿真秒。
GUI 默认关闭，实际墙钟耗时还包含启动、复位和保存；首次验证建议先用下方的短程命令。

```python
WORLD_DIR = ROOT / "assets/worlds/simple"
WORLDS = [WORLD_DIR / name for name in (
    "train_world3.sdf",
    "rounded_rectangle_001.sdf",
    "ellipse_002.sdf",
    "rounded_rectangle_003.sdf",
)]
COLLECTION_SECONDS = 3600  # 每轮仿真秒数上限
COLLECTION_ROUNDS = 20     # 每张地图轮数
SPAWN_JITTER_ENABLED = True
SPAWN_POSITION_JITTER_M = 0.03  # 普通轮次 x/y 各 ±3 cm
SPAWN_YAW_JITTER_DEG = 10.0     # 普通轮次朝向 ±10 度
RECOVERY_ENABLED = True
RECOVERY_FRACTION = 0.25        # 每图 20 轮中 5 轮近墙纠正，轮次顺序由种子决定
RECOVERY_MAX_OFFSET_M = 0.12
RECOVERY_CLEARANCE_M = (0.04, 0.08)  # 保守车体净空，不是雷达量程
RECOVERY_YAW_DEG = (10.0, 20.0)      # 朝横移的一侧偏航
RECOVERY_FRONT_M = (0.15, 0.40)      # 前进方向车体首次接触障碍前的距离
SPAWN_MARGIN_M = 0.01
SPAWN_ATTEMPTS = 200
LIDAR_WAIT_SEC = 0.2      # 等待扫描的现实秒数，偶发超时继续驾驶
MAX_CONSECUTIVE_LIDAR_MISSES = 5  # 连续缺帧决策点达到此数时停止
GUI = False              # True 显示优先；False 无窗口、不限速采集
GUI_REAL_TIME_FACTOR = 1.0  # 仅 GUI 生效；1 为实时，0.5 为半速，必须 > 0
```

`GUI=True` 限制物理推进速度，并关闭场景阴影以减轻软件渲染负担。
地图 JSON 有工作边界且 SDF 未指定相机位置时，默认从上方观看整个赛道，避免墙体遮挡小车；
仍可在窗口中手动调整视角。
`GUI=False` 或 `--headless` 不启动 gzclient，实时倍率设为 0（不限速），
不等待 GUI 启动；仍保留完整的雷达更新、时间戳和动作对应检查。
两种模式使用同样的模型、控制周期和 CSV 格式。每轮结束打印实际采集耗时及条数/秒，
并在 Gazebo 的 `metadata.json` 中记录 `wall_seconds`（不含启动、复位和增强复制）。
程序使用暂停状态下的精确步进，窗口中的暂停图标、Real Time Factor 或 Real Time
可能保持不变；应观察 **Sim Time 和小车位置**。采集中不要手动点击播放或重置。

每张地图的多轮记录合并为一份原始 CSV，再生成一份同样行数的增强 CSV。
地图文件名必须互不相同；以同名 JSON 的评估出生点依次循环作为采样基准。
普通轮次在基准附近随机改变位置和朝向；近墙轮次随机向左/右横移，并朝同侧偏转，
让教师从已有偏差的状态开始纠正。所有姿态都检查车体净空和工作边界；
近墙候选还检查一个完整控制周期内最大纠正转向的保守圆扫掠路径。
这项几何检查只排除明显不可行的候选，不保证 PPO 对任意地图都能恢复；新增地图需短程验证。
行驶中的所有动作仍来自原教师，不施加随机错误动作，也不修改奖励或模型。

每图近墙轮数为 `floor(COLLECTION_ROUNDS * RECOVERY_FRACTION + 0.5)`，其余轮次为普通扰动。
这是轮数比例，不是 CSV 行数比例；例如 1 轮、25% 时近墙轮数为 0。
关闭 `RECOVERY_ENABLED` 可只保留普通扰动；两个开关都关闭即恢复原固定出生姿态。
`--check` 会预先生成并检查全部出生场景；同地图、参数和种子得到相同计划。
候选尝试耗尽会明确报错，不会静默退回固定出生点。
切换地图前会等待当前 Gazebo 进程组退出并确认端口释放；即使 `ros2 launch` 已先退出，
也会清理本实例残留的子进程。若清理失败或端口仍被其他实例占用，会明确报错停止切图，
不会终止其他仿真。
时长自动按模型的实际控制周期换算为完整步数，向下取整，至少一个控制步；
例如当前周期为 0.2 秒，配置 180 秒得到 900 步，配置 1.1 秒实际最多采集 1.0 秒。

```bash
# 在 dissertation 根目录运行
python Gazebo_simulation/rl_training/privileged/collect_data.py --check
python Gazebo_simulation/rl_training/privileged/collect_data.py
# 可选：关闭窗口，覆盖顶部的每轮秒数和每图轮数
python Gazebo_simulation/rl_training/privileged/collect_data.py --headless --seconds 20 --rounds 4
```

采集时额外开启雷达；教师仍用原有 292 维地图/物理状态决策，奖励也仍用地图真值。
`LIDAR_CONFIG` 默认指向共享 `my_gazebo_launch/vehicle_config.h`，仅覆盖采集雷达的
扫描频率和原始点数：目前为 5.5 Hz、364 点/圈，对齐实车标准 SCAN 模式约 2000 点/秒。
车辆动力学、动作标签、200 桶输出和控制周期仍使用教师原配置；模型文件及随附配置不变。
每批在 Gazebo 结果目录保存生效的 `vehicle_config.h`，元信息中的 `vehicle` 为采集配置，
`teacher_vehicle` 为教师原配置。修改这两个雷达参数后应新建批次；
旧 1450 点/圈的批次不会被覆盖，也不能直接以 364 点/圈续采，避免同批混入不同点密度。
每个控制边界取最近的新扫描，配对接下来实际执行的教师动作。检查雷达时间戳递增、
不晚于决策时刻、帧龄不超过一圈加一个物理步长。
新扫描最多等待 `LIDAR_WAIT_SEC` 现实秒；超时后教师继续按原控制周期驾驶，
缺帧的决策点不生成 CSV 行，下一次正常物理步进可唤醒暂停边界处等待更新的雷达。
本次动作前有有效扫描时，即使动作结束后缺帧，仍保留本次已经执行的动作对应行。
连续 `MAX_CONSECUTIVE_LIDAR_MISSES` 个决策点缺帧会明确报错停止，保留已有文件。
此处理只对特权教师采集启用；传感器格式错误、仿真退出及其他数据错误仍会报错。
缺帧不补零、不沿用旧扫描；一帧真实的全零扫描仍是有效输入。
扫描与教师物理状态使用各自的时间戳，并非假设两者严格同时。
缺帧处理参数、出生场景参数及完整计划也用于续采校验；旧批次缺少这些字段时需新建批次。

每次生成 `results/datasets/collection_<时间>_<批次>/`：

- `raw/<地图名>_raw.csv`：该地图各轮按采集顺序追加的原始记录，每行 200 个整数毫米距离
  及一个动作，共 201 列，无表头。
- `dropout/<地图名>_dropout.csv`：原始数据采集完成后生成的增强副本，行数、顺序和标签不变。
  每行从非零距离中无放回随机抽取 `floor(非零点数 × 0.05 + 0.5)` 个位置设为零；
  未选中的距离及原有零值不变。每行独立抽取，固定随机种子可复现。
- `<地图名>_timing.jsonl`：逐行保存地图 CSV 行号、轮次、轮内步数，以及雷达、决策和动作结束
  时间戳，单位纳秒。缺帧后 CSV 行号连续，控制步号可能跳跃。
- `<地图名>_lidar_events.jsonl`：缺帧与恢复事件、决策时间戳和连续缺帧次数。
- `metadata.json`：模型/地图指纹、车辆参数、行数、标签分布、结束原因、增强参数、
  文件指纹和原始/增强文件共同的 `group_id`。`maps` 汇总每张地图，
  `episodes` 记录每轮及其 `row_start`、`row_end`（从 1 开始，包含两端）。
  `rows` 为实际保存行数，`control_steps` 为已执行控制步数，`skipped_rows` 为缺帧而未记录的步数；
  一轮没有行时行号范围为 `null`。整张地图没有行则报错，不发布空训练集。
  `spawn_randomization` 保存出生参数，`spawn_plans` 保存各地图指纹、种子和逐轮计划；
  每轮 `spawn_scenario` 包含场景类型（`fixed` / `jitter` / `near_wall`）、基准出生点编号、
  采样姿态及初始几何净空，可结合回合终止原因检查教师表现。该信息不增加 CSV 列。
- `runtime_*/`：本次仿真的日志与运行状态。

完整采集和增强结束后，仅原始与增强 CSV 原样复制至
`dissertation/dataset/simulation/<同名批次>/`，保持 `raw/`、`dropout/` 两个子目录；
复制校验成功后才发布该目录。四张地图得到 4 份原始 CSV 和 4 份增强 CSV。
`metadata.json`、逐行时间戳和运行日志只保留在 Gazebo 的原始采集目录，
通过同名批次目录查找。
原始和增强记录数量为 1:1，例如 900 条原始记录对应 900 条增强记录。
失败或中断的采集留在结果目录，元信息标记原因，不发布到训练目录。
碰撞等正常回合终止会保留终止动作对应的样本，并记录 `success` 和 `termination_reason`；
采集阶段不清理、筛选样本，也不执行 K 特征选择或随机森林训练。

后续训练需要处理以下约定：

- 同一地图的原始与增强文件按 Gazebo 原始目录元信息中的 `group_id`
  一起分到训练集或验证集；若按回合划分，则按 `episodes` 行号范围保持原始/增强样本配对，
  避免数据泄漏。
- 当前训练器会删除选中特征含零的行，训练增强数据前需调整该策略。
- 仿真标签为 0～14、7 直行；旧实车是 10 直行。`dataset/physical` 与
  `dataset/simulation` 可由训练器统一递归读取，但混训前必须统一标签含义与雷达定义。

PPO、奖励与仿真配置共用 [common/config.py](../common/config.py)。
完整训练、续训、验收和网页命令见 [总说明](../README.md)。

## 多地图联合训练

编辑 [train_multi_map.py](train_multi_map.py) 顶部的配置，然后运行：

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

检查通过后再执行训练命令。脚本管理自己的仿真，无需手动开 Gazebo。
`--check` 校验配置、地图和出生点净空，并打印自动选择的续训文件，不启动训练或发布器。
车辆参数仍由 `CONFIG_HEADER` 指定。

### 自动接着上次训练

```python
AUTO_RESUME = True
RESUME_MODEL = ROOT / "rl_models/privileged/policies/latest_validated/model.zip"
```

开启后扫描 `OUTPUT_DIR` 中的特权多地图批次，按批次由新到旧查找：优先使用该批次的
`rl_policy_final.zip` 或 `rl_policy_interrupted.zip`，否则按步数由高到低尝试定期检查点。
只选择模型和对应归一化文件均能读取、输入/动作空间匹配的组合；缺失或损坏时打印原因并回退。
没有可用多地图历史时才使用 `RESUME_MODEL`；它为 `None` 则从零训练。
不会自动选择较早的 `best_model.zip`，也不会修改 `latest_validated`。

每次启动都会新建结果目录，并重新获得本次设置的时间/步数预算。
Ctrl+C 保存后可直接再次运行；断电或强制终止只能恢复到最后一个完整检查点，
未保存的采样和当前车辆位置不会恢复。模型配置不兼容时明确报错，不自动改车辆或网络。
请不要同时启动两个训练任务共享同一套输出目录及仿真端口。

要固定从某个模型开始，设置 `AUTO_RESUME=False` 并填写 `RESUME_MODEL`；
要完全重新训练，设置 `AUTO_RESUME=False`、`RESUME_MODEL=None`。
扫描范围是 `OUTPUT_DIR`，需要独立实验时可更换此目录，或关闭自动续训。

| 配置 | 默认行为 |
| --- | --- |
| `TRAIN_WORLDS` / `EVAL_WORLDS` | 5 张训练地图、2 张独立选模地图；修改显式列表即可换地图 |
| `NUM_TRAIN_ENVS` | 4 个独立训练实例，共同更新一个 PPO 模型 |
| `EPISODES_PER_MAP` | 每实例完成 10 回合后切图；初始按 worker 编号错开，地图多于实例也会轮到 |
| `STEP_SIM_TIME_SEC` | 每个动作持续 0.2 仿真秒；物理积分步长仍在车辆配置中 |
| `MAX_STEPS_PER_EPISODE` | 750 步，最多 150 仿真秒；撞墙/卡住等会提前结束并重新抽取出生点 |
| `TRAIN_POSITION_JITTER_M` / `TRAIN_YAW_JITTER_DEG` | 每次从 JSON 抽取训练起点，x/y ±3 cm、朝向 ±10° |
| `EVAL_*JITTER*` / `EVAL_EPISODES_PER_MAP` | 每图 8 回合；固定种子、固定起点顺序与扰动，便于比较训练进展 |
| `TOTAL_TIMESTEPS` / `MAX_TRAINING_HOURS` | 合计 1000 万步或 6 小时预算；预算在采样回调检查，评估/保存可能使结束稍延后 |
| `N_STEPS` / `BATCH_SIZE` | 每实例采样 512 步后更新；批量 256，须整除总 rollout 大小 |
| `EVAL_FREQ` / `SAVE_FREQ` | 每 20000 合计步数评估/保存（按并行实例数取整） |

奖励、卡住/绕圈阈值、学习率、网络结构、随机种子、端口、输出目录也在脚本顶部。
运动速度和轮距继续使用共享 `vehicle_config.h`。仿真按算力尽快步进，
`STEP_SIM_TIME_SEC` 不是仿真加速倍数，也不是车速；改大它会降低控制频率。
目前扰动是初始位置和朝向，不模拟 PWM、电流或电机内部 PID。

出生点使用车体保守半径加额外净空筛选，越界/贴墙候选会重采样，超过配置次数则报错。
切图需要重启该 worker 的 Gazebo，因此不建议每回合都换图。默认未加入死胡同、
直角矩形和 8 字交叉图；先确认道路允许当前固定前进速度和转弯半径再添加。
不要在其他仿真占用的 ROS domain/端口上启动训练；训练实例使用基础值加 `2*i`，
评估单独占用一组，默认共最多 5 个 Gazebo 进程。

每次生成新的 `PPO_privileged_*` 批次，保存实际参数、地图及 JSON 快照、源码、
Monitor 出生点记录、模型和归一化状态。`evaluation/multi_map.json` 是最近完整一轮
逐地图评估，带训练步数；同目录按步数保留历史报告。训练和评估均不读取雷达。
单地图 `train.py --evaluate` 会读取模型记录中的控制步长和失败/奖励配置，
回合步数及地图边界仍由本次评估参数决定。续训要求控制步长和网络结构与原模型一致。

手机页显示各 worker 当前地图/出生点，以及每张评估地图的完成率、均速、失败次数、
最早/平均撞墙和卡住时间。无失败只表示本次测试时间内没有观察到；不能解释为永不撞墙。
`best_model.zip` 仍按等回合数的跨地图平均奖励选择。只有下方的独立自动验收全部通过，
才会更新 `latest_validated`；定期选模本身不会更新这个链接。

进度发布器由 `train_multi_map.py` 顶部的开关控制，默认随训练自动启动：

```python
ENABLE_PROGRESS = True       # 自动开启本地手机网页
PUBLISH_TO_GITHUB = True     # 同时同步 GitHub；False 仅本地
```

`ENABLE_PROGRESS=False` 时不启动或停止任何发布器；`--check` 也不会启动发布器。
如果已有发布器运行则复用，保留它原有的上传设置，不重复启动或关闭它。
本脚本启动的发布器会在训练正常结束、失败或 Ctrl+C 后保存最终快照，
开启 GitHub 时尝试最后一次上传，然后退出；已有发布器会收到刷新请求并继续运行。
网络故障不会阻止模型保存，最终同步失败会打印提示，退出最多等待 180 秒。
自动发布器退出后本地网页服务也会停止，GitHub 页面保留最后一次成功同步的数据。

需要独立保留网页服务或稍后重新同步时，在另一终端执行：

```bash
python rl_training/monitoring/publish_progress.py --serve --publish
```

本地手机同 Wi-Fi 访问 `http://电脑IP:8765`，远端访问既有 GitHub Pages 页面。
中止训练可按 Ctrl+C，或在本次模型目录创建 `STOP` 文件；程序会保存当前权重。

## 训练结束后自动验收

`train_multi_map.py` 顶部增加了独立验收配置：

```python
AUTO_VALIDATE = True
VALIDATION_WORLDS = [...]             # 与训练/定期评估分开的地图
VALIDATION_MODEL_NAME = "best_model.zip"
VALIDATION_EPISODES_PER_MAP = 8
VALIDATION_MIN_SCENARIOS = 8
VALIDATION_EPISODE_SECONDS = 300.0
VALIDATION_MIN_SPEED_COMPLIANCE = 0.95
VALIDATION_POSITION_JITTER_M = 0.03
VALIDATION_YAW_JITTER_DEG = 10.0
VALIDATION_SEED = 20261012
```

默认地图在 `assets/worlds/validation/`，用种子 20261012 生成圆角矩形和椭圆各一张，
文件名带 `validation_` 前缀，未放入训练或选模列表。每图至少 8 个不同位置/朝向，
每回合巡航 300 仿真秒；所有回合必须正常达到时限，且零碰撞、卡住、小范围绕圈、
持续低速及越界，每回合速度合规率至少 95%。速度容差沿用模型训练配置。
两图完整跑满共计 4800 仿真秒（80 分钟），实际墙钟耗时取决于仿真速度。
这是额外的验收时间，不包含在 `MAX_TRAINING_HOURS` 的训练预算内。

训练正常完成或达到时间预算后，仿真训练实例先关闭，再串行验收本批次的最佳模型；
若训练太短尚未产生 `best_model.zip`，使用本次 `rl_policy_final.zip`。
验收只执行固定策略，不进行 PPO 更新。按 Ctrl+C 或创建 STOP 主动停止训练时不自动验收。
验收中再按 Ctrl+C 会停止验收，已保存的训练权重仍保留。

全部通过才冻结到 `validated_privileged_步数_时间/` 并更新 `latest_validated`。
未通过或验收未完成均不会晋升模型。每批次的 `auto_validation.json` 记录验收状态、
参数和报告路径；各地图报告仍保存为 `policies/evaluation_时间_编号/`，网页现有的
独立评估区域可以查看实时进展，发布器会等验收结束后再同步最终状态并退出。
`--check` 会提前检查验收地图是否与训练/选模重复、起点数、扰动后净空及参数有效性。
验收地图与 JSON 也会随训练批次保存快照。

关闭自动验收只需设置 `AUTO_VALIDATE=False`，不影响训练保存或 AUTO_RESUME。
自动续训仍选择最近训练状态；`latest_validated` 专门保留已通过所配置验收的模型。
通过这些测试仅说明在记录的地图、起点、扰动和时长下达标，不代表任意地图都不会撞墙。
