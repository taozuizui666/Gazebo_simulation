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
地图解析目前支持静态、水平放置的 box 碰撞体，不支持的几何会报错。
更换地图使用同名 JSON 的出生点和工作边界，并重新验证泛化表现。

后续若用教师生成随机森林数据，仍需在同一车辆状态下配对雷达帧和教师动作；
本目录负责教师训练，当前没有启动这项数据采集。

PPO、奖励与仿真配置共用 [common/config.py](../common/config.py)。
完整训练、续训、验收和网页命令见 [总说明](../README.md)。

## 多地图联合训练

编辑 [train_multi_map.py](train_multi_map.py) 顶部的配置，然后运行：

```bash
python rl_training/privileged/train_multi_map.py --check
python rl_training/privileged/train_multi_map.py
```

先按总说明加载 ROS/Gazebo 和 `install/setup.bash`。脚本管理自己的仿真，无需手动开 Gazebo。
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
`best_model.zip` 仍按等回合数的跨地图平均奖励选择，查看逐地图结果后还需在额外未参与选模
的地图长时验收；不会自动替换 `latest_validated`。

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
