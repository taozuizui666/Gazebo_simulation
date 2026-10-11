# 两类 RL：雷达策略与特权教师

代码按输入类型分开，训练结果也分别保存。两类策略都使用 PPO，任务是固定目标前进速度下的持续避障巡航。

| 类型 | 网络输入 | 代码入口 | 结果目录 |
| --- | --- | --- | --- |
| [雷达策略](lidar/README.md) | 200 列规范雷达距离 | `lidar/train.py` | `rl_models/lidar/` |
| [特权教师](privileged/README.md) | 17×17 局部地图距离场 + 真实前进速度、角速度、上一动作，共 292 维 | `privileged/train.py` | `rl_models/privileged/` |

`common/` 保存两类共用的 PPO 训练、Gazebo 步进、奖励、失败检测与评估代码；
`monitoring/` 保存本地和 GitHub Pages 进度显示。相同功能只维护一份。

## 运行环境

以下命令均在 `Gazebo_simulation` 目录执行：

```bash
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.bash
colcon build --symlink-install --packages-select my_gazebo_launch
source install/setup.bash
```

使用已安装依赖的 Python 3，无需 venv。需要补装 Python 依赖时：
`python -m pip install --user -r rl_training/requirements.txt`。
训练自动创建独立的 Gazebo，无需提前启动手动仿真。

## 训练、续训与评估

新训练二选一：

```bash
python rl_training/lidar/train.py
python rl_training/privileged/train.py
```

入口固定输入类型，已移除旧的 `--observation-mode` 参数。两个入口使用相同的
`--help` 选项，但分别写入各自的 `policies/` 和 `logs/`。

例如续训特权教师，给这一轮最多 6 小时：

```bash
python rl_training/privileged/train.py \
  --resume rl_models/privileged/policies/latest_validated/model.zip \
  --timesteps 10000000 --hours 6
```

`--timesteps` 是本轮新增采样目标；步数或时间预算先到即结束。PPO 以 rollout 为单位
更新，最终步数可能略超过目标。计时包括采样和定期评估，启动及最终保存也会占用实际时间。
续训总是创建新批次，不覆盖原模型。

评估当前特权教师并打开 Gazebo，观看 3 分钟仿真巡航：

```bash
python rl_training/privileged/train.py \
  --evaluate rl_models/privileged/policies/latest_validated/model.zip \
  --gui --eval-episodes 1 --episode-steps 900
```

评估雷达模型：

```bash
python rl_training/lidar/train.py \
  --evaluate rl_models/lidar/policies/latest_validated/model.zip \
  --eval-episodes 8 --episode-steps 3000
```

评估、续训默认使用模型同目录的 `vehicle_config.h`。这很关键：目前已验证雷达模型
是 **0.1 m/s**，特权教师是 **0.2 m/s**。新训练使用共享车辆头文件的当前设置。
入口与模型类型不符时立即报错。模型输入规格改变后也不能直接续训。

如需有意改变车辆参数来微调，使用 `--config-header 新头文件路径` 和
`--resume-allow-vehicle-change`，随后重新独立验收。现有结果不能证明新地图上的表现。

## 选择地图与出生点

`--world` 指定地图；省略时使用 [common/config.py](common/config.py) 顶部的 `WORLD`，
默认值来自 `my_gazebo_launch/config.py` 的 `DEFAULT_WORLD`（当前为 `train_world3.sdf`）。
出生点和工作边界随地图从同名 JSON 读取，不在训练代码里固定。
原地图对应 `assets/worlds/train_world3.json`；生成地图自动带有此文件。

```bash
python tools/generate_maps.py --count 6
python rl_training/privileged/train.py --world assets/worlds/generated/rounded_rectangle_001.sdf

# 用新地图评估已保存模型，也使用新地图的出生点和边界。
python rl_training/privileged/train.py \
  --evaluate rl_models/privileged/policies/latest_validated/model.zip \
  --world assets/worlds/generated/ellipse_002.sdf --gui --eval-episodes 8
```

同名 JSON 使用 `train_spawn_poses`、`eval_spawn_poses`、`workspace_bounds` 三个字段，
每个出生点为 `[x, y, z, yaw]`（米、弧度）；边界为 `[xmin, xmax, ymin, ymax]`，
设为 `null` 时关闭越界判定。命令行可覆盖：

```bash
python rl_training/privileged/train.py --world assets/worlds/train_world3.sdf \
  --train-spawn -0.931026 -1.305602 0 0 \
  --train-spawn 0.05 -3.0 0 1.5707963267948966 \
  --eval-spawn -1.2 -4.94 0 3.141592653589793 \
  --workspace-bounds -2.9 0.5 -5.45 -0.8
```

`--train-spawn` / `--eval-spawn` 可重复，分别替换整组起点；并行车按起点列表循环分配。
`--no-workspace-bounds` 显式关闭越界判定，不能和 `--workspace-bounds` 同用。
没有同名 JSON 的自定义地图需要显式传入起点；只提供训练或评估起点时，两者共用该列表，
未指定边界时不启用越界判定。所有起点必须在已启用的边界内。
训练和评估结果记录实际起点、边界与地图路径，训练快照同时保存原地图 JSON。

地图路径可为绝对路径，或相对于 `Gazebo_simulation` 的资源路径。
生成的地图可直接读取，无需为每次生成重新构建资源。
一次训练使用一张指定地图；`--num-envs` 控制在该图上的并行车辆数量。
地图批量生成说明见 [工具说明](../tools/README.md)。
目录与旧路径对照见 [工程目录说明](../docs/layout.md)。

## 配置在哪里

重要配置继续以文件顶部的大写常量集中提供：

| 文件 | 修改内容 |
| --- | --- |
| [my_gazebo_launch/vehicle_config.h](../my_gazebo_launch/vehicle_config.h) | 前进速度、标签数、轮距、轮径、雷达点数与协议 |
| [lidar/config.py](lidar/config.py) | 雷达模型、日志、运行文件的默认路径 |
| [privileged/config.py](privileged/config.py) | 特权模型路径、距离场网格、覆盖范围、归一化和几何安全间距 |
| [common/config.py](common/config.py) | 两类共用的默认地图、domain/端口、回合时限、奖励、卡住检测及 PPO 参数 |
| `assets/worlds/地图名.json` | 该地图的训练/评估出生点、朝向和工作边界；支持命令行覆盖 |
| [monitoring/progress_config.py](monitoring/progress_config.py) | 网页地址、端口、目录、刷新周期和统计窗口 |

默认 4 个训练实例：domain 81/83/85/87、端口 11451/11453/11455/11457；
独立评估使用 domain 82、端口 11452。两种模型默认共用这些资源，因此默认命令应
分别运行。需要同时训练时，为其中一组指定不重叠的 `--train-domain`、
`--eval-domain`、`--train-port`、`--eval-port`；训练实例每增加一辆，domain 和端口均加 2。

## 驾驶任务与保存内容

每个动作默认推进 0.2 秒仿真时间，网络更新时物理世界暂停。雷达模式的步长还需覆盖
至少一圈扫描。动作只决定转向，目标前进速度固定，左右轮均保持前进；实际速度由
物理仿真测量。接触障碍、卡住、持续低速、局部绕圈、越界都会终止回合。
下一回合恢复该实例的出生位置、朝向和零速度。正常到达回合时限属于采样截断，不是碰撞。

奖励包含行驶距离、速度偏差、障碍风险、转向幅度及转向变化。雷达策略读取规范雷达
距离；特权教师使用地图几何及物理真值，关闭雷达传感器和转换节点。

每个 `policies/PPO_*/` 批次保存：

- 模型权重及配套 `vecnormalize.pkl`，用于正确续训。
- `run_config.json`、车辆头文件、共享训练配置和本类 `input_config.py`。
- `source_snapshot/`：当时源码、世界、车辆资源及散列清单。
- 各训练车的回合、实时状态和 Gazebo 日志，以及定期评估结果。

独立评估将模型散列、原始轨迹、速度和失败原因写入本类 `policies/evaluation_*/`。
轨迹 PNG/PDF 统一写入 `results/training/<类型>/policies/evaluation_*/`；
已验证模型的图片也保存在对应结果批次中。结果根目录由 `common/paths.py` 的 `RESULTS_DIR` 配置。
自定义 `--output-dir` 位于 `rl_models/` 外时，图片归入 `results/training/external/<绝对路径>/`。
TensorBoard 分别查看 `rl_models/lidar/logs` 或 `rl_models/privileged/logs`。
正常结束或 Ctrl+C 中断会尝试保存当前模型；也可在本次批次目录创建 `STOP` 文件，
让训练在下一次采样检查时停止。

## 冻结通过验收的模型

普通训练检查点不会自动替换 `latest_validated`。从本类目录选定同一份权重和报告后：

```bash
python rl_training/common/qualify_model.py \
  --model rl_models/privileged/policies/具体训练批次/best_model.zip \
  --report rl_models/privileged/policies/具体评估批次/evaluation.json
```

上面的“具体批次”须替换为实际目录名。至少要求 8 个不同起点、每段至少 300 秒、
零失败、速度合规率至少 95%，且报告的权重散列、车辆及输入规格匹配。
通过后按模型类型保存到对应 `policies/` 并更新该类的 `latest_validated`。
这表示在所记录条件下通过测试，不保证其他地图上不会碰撞。

## 手机进度页

本地和 GitHub Pages 共用 [monitoring/dashboard/](monitoring/dashboard) 的页面。
进度服务无需加载 ROS，不会启动训练：

```bash
python rl_training/monitoring/publish_progress.py --serve --publish
```

本机：<http://127.0.0.1:8765>；同 Wi-Fi 手机：`http://电脑局域网IP:8765`；
外网手机：[GitHub Pages](https://taozuizui666.github.io/dissertation/)。
只看本地时省略 `--publish`。新训练及评估自动接入，电脑和进度服务需保持运行。

页面顶部先选“模型类别”，再选该类训练批次。训练曲线、失败时间、独立验收、
模型参数及图片均按类别切换。撞墙和卡住时间仅统计实际发生该失败的回合；
正常达到时限显示“至少运行到此时”，不会推算之后多久失败，也不会把多段累计时间
当成一段连续巡航。

本地约 5 秒更新；GitHub 数据约 60 秒检查同步，手机约 15 秒检查。网络和缓存可能
增加延迟。页面显示快照时间，区分训练、评估、历史结果及数据中断。

后台运行（与前台命令二选一；本机已有服务时不要重复启动）：

```bash
systemd-run --user --unit=aicar-rl-progress --description='AIcar RL progress' \
  --property="WorkingDirectory=$PWD" --property=Restart=on-failure \
  /usr/bin/python3 -u "$PWD/rl_training/monitoring/publish_progress.py" --serve --publish
systemctl --user status aicar-rl-progress.service
journalctl --user -u aicar-rl-progress.service -n 30 --no-pager
```

停止：`systemctl --user stop aicar-rl-progress.service`。此临时服务重启电脑后需重新启动。
网页代码或配置变化时更新 `gh-pages`；统计数据单独同步到 `rl-progress-data`。
发布器只上传白名单统计、页面和图片，保留模型及原始日志在本机；
上传失败时本地继续更新，后续周期重试。`--models-dir` 指定包含两类子目录的结果根目录。

## 历史记录与故障检查

本次分类只迁移历史模型、日志和报告，不改写历史 `policies/` 中的文件内容。
旧报告及源码快照里的原始路径用于记录当时环境，不能直接作为当前命令运行；
路径对照保存在本机 `rl_models/runtime/layout_migration.json`。
当前模型与证据索引见 [rl_models/README.md](../rl_models/README.md)。

包不存在时先构建并加载环境；启动失败查看本次批次的 `worker_*/gazebo_train.log`、
`evaluation_server/gazebo_eval.log` 及 ROS 日志；端口冲突时结束对应旧进程或更换
整组 domain/端口。不要同时运行占用同一组资源的两套默认训练命令。
