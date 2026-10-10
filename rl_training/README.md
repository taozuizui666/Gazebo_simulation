# 持续避障巡航训练

任务是在固定前进速度下，靠雷达选择转向并持续巡航。没有终点；撞到障碍物或
卡住、持续低速、小范围反复绕圈或离开道路边界都是失败。每集正常达到采样时限
后截断。每个训练实例有固定出生位置和朝向，下一集回到该位置。

默认使用 PPO：200 个雷达距离输入、15 个标签（0～14，7 直行）、0.1 m/s。
这三个数直接来自共享的 [vehicle_config.h](../my_gazebo_launch/vehicle_config.h)，
没有在 RL 代码里再维护一套数值。该模型用于仿真策略学习，后续再采集数据训练
随机森林；本次没有生成或覆盖实车的 `randomForest.h`。

本机已完成训练及 16 段长时独立验证，模型入口为
`rl_models/policies/latest_validated/model.zip`，实测结果与观看命令见
[已训练模型](../rl_models/README.md#已训练模型2026-10-10)。

## 启动训练

在 `Gazebo_simulation` 目录执行：

```bash
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.bash
python3 -m pip install --user -r rl_training/requirements.txt
colcon build --symlink-install --packages-select my_gazebo_launch
source install/setup.bash
python3 rl_training/train_rl.py
```

依赖直接安装到当前用户 Python，无需 venv。要求系统已有 ROS 2 Humble、Gazebo
Classic 11 和相应 ROS 插件，系统软件安装见 [主 README](../README.md#启动)。
依赖文件限制 NumPy < 2、setuptools < 80，避免本机 ROS OpenCV 和 colcon 的兼容问题。

加速步进使用本地 C++ Gazebo 插件，首次启动自动编译到被忽略的 `build/rl_native/`。
新机器还需安装 `sudo apt install build-essential pkg-config ros-humble-gazebo-dev`。
修改 `rl_training/native/*.cpp` 后，下次启动自动重新编译。

训练脚本默认自动启动、管理和关闭 4 个训练仿真和 1 个评估仿真：

| 用途 | ROS domain | Gazebo master 端口 |
| --- | --- | --- |
| 训练 0～3 | 81、83、85、87 | 11451、11453、11455、11457 |
| 评估 | 82 | 11452 |

不需要预先手动启动 Gazebo。各仿真的话题名称相同，但 domain 和 master 不同，
评估复位不会移动训练小车。若端口被占用，脚本报错并退出，不接管已有仿真。

先跑一次小规模流程检查：

```bash
python3 rl_training/train_rl.py --timesteps 1024 --num-envs 4 --rollout-steps 256 \
  --eval-freq 512 --eval-episodes 1 --episode-steps 50 --hours 0.1
```

正式训练可覆盖步数和评估配置：

```bash
python3 rl_training/train_rl.py --timesteps 500000 --num-envs 4 --hours 4 \
  --eval-freq 10000 --eval-episodes 4
```

`--eval-freq` 是多少个训练动作后评估一次；`--eval-episodes` 是每次评估多少集。
PPO 按完整 rollout 更新，实际步数可能补齐到 `rollout-steps × num-envs` 的整数倍。
该乘积必须是批次大小 256 的整数倍。`--hours` 按实际墙钟时间限制训练，达到预算就
保存当前模型；不保证此时策略已经达标。Ctrl+C 也会尝试保存并关闭本次的仿真。
训练覆盖道路不同位置及两个方向，不只从单一位置学习。
也可在本次运行目录创建空文件 `STOP`，程序会在一个完整动作之后保存并退出，
适合后台训练。多进程清理有超时保护，传感器错误会关闭对应仿真并保留失败状态。

## 观测、动作和物理步进

策略输入只有雷达距离。环境读取带时间戳的 `/scan_raw`，调用与
`/lidar_distances` 相同的 `scan_to_distances()`，得到同列顺序的整数毫米数据后
除以共享距离上界，作为 float32 输入。网络内部再对输入开平方，让近距离差异
更容易学习，零值保持为零；观测仍是同样的 N 列，没有额外位置、朝向或地图输入。
策略为 `[128, 128]` MLP，奖励使用 `VecNormalize` 归一化；独立评估报告原始奖励。

- 默认 200 列；只保留 0～49、151～199，50～150 为零。
- 零表示无效或未采集，不把无回波伪装成最大距离，不沿用上一帧。
- 角度方向、q6/q2 量化、整数角度分桶和有效距离规则与实车训练格式一致。
- 接触传感器和里程计只用于奖励、复位检查及诊断，不送入策略网络。

A1M8 手册给出 0.15～12 m **典型量程**；默认仿真下界为 0.15 m，可在共享
头文件中修改。转换器按扫描自身的 `range_min` 判断有效性，不再用 0.15 m
额外删除真实驱动已经确认有效的近距离点。碰撞判断使用车身和左右轮的实际
Gazebo 接触，因此不依赖雷达能否看到 15 cm 内的墙。安装角度、扫描噪声、
遮挡和车轮打滑仍需要按实车标定，输入格式一致不代表物理误差完全一致。

动作标签经同一份左右轮目标速度公式控制仿真车，停车指令独立于学习标签。
每个动作默认推进 **0.2 秒仿真时间**，物理步长由共享宏设置为 0.001 秒。
如果降低雷达频率，动作时长自动增大到能容纳一圈新扫描，并补齐到整数物理步。
其余时间物理世界暂停，包括神经网络更新和等待独立评估期间。仿真按算力加速，
并使用 Gazebo lockstep 保持雷达采样时序；0.1 m/s 始终按仿真时间解释。
暂停时仿真时钟不前进，物理步长和车速目标不随电脑算力改变。

每次步进要求本次动作之后的新扫描、步进末尾附近的里程计，以及原生插件返回的
确切结束时间和物理接触结果。原生插件检查该动作内每个物理步的整车接触并锁存
碰撞；加速训练不依赖异步 ROS bumper 消息是否刚好及时送到。
传感器缺失、仿真退出或时钟倒退会停车并报错，不给 PPO 填入旧数据或零数据。
控制指令有独立心跳，避免计算期间的等待导致车辆指令失效。

## 回合与奖励

| 条件 | 行为 |
| --- | --- |
| 车身或左右轮接触障碍物 | `terminated=True`，碰撞罚分，停车 |
| 3 秒仿真时间内实际移动不足 0.02 m | `terminated=True`，卡住罚分，停车 |
| 3 秒内平均前进速度低于目标的 75% | `terminated=True`，低速失败 |
| 至少 6 秒、最近 12 秒轨迹范围小于 0.65 m，累计转角达到 5.5 rad | `terminated=True`，小范围绕圈失败 |
| 超出配置的道路边界 | `terminated=True`，越界失败 |
| 正常达到 750 步 | `truncated=True`，记录完成一段巡航，无额外失败罚分 |
| 调用 `reset()` | 停车、暂停、复位关节、恢复出生位置/朝向并清零速度，等待新传感器数据 |

地面及车辆自身接触不计作撞墙。复位保留单调递增的仿真时钟，但清空旧传感器
缓存和回合统计。正常 750 步约为 150 秒巡航，不含复位稳定时间。
所有标签都保持两轮向前，不提供停止、倒车或原地旋转动作。小范围绕圈检测使用
一段时间内的轨迹包围框和实际累计转角；沿整个环形道路正常跑圈不属于这种行为。

奖励项在 [reward.py](reward.py) 中，系数集中放在 [config.py](config.py) 顶部：

| 奖励项 | 默认规则 |
| --- | --- |
| 巡航距离 | 本步有效前进米数 × 10；同时受实际前进速度和目标速度上限约束 |
| 速度误差 | −3 × abs(实际速度 − 目标速度) × 仿真时间 |
| 近障碍 | 最小有效雷达距离低于 250 mm 时按接近程度及仿真时间扣分 |
| 转向变化 | −0.1 × abs(本次标签 − 上次标签) / (标签数量 − 1) |
| 前方风险 | 前方 ±20° 有效距离小于 450 mm 时，按接近程度平方扣分 |
| 持续大转向 | 按归一化转向幅度平方及仿真时间轻微扣分 |
| 失败 | 碰撞、卡住、低速、绕圈、越界均为 −20，并取消本步距离奖励；失败罚分不叠加 |

保持同一转向不会产生变化惩罚；第一步没有上一动作，也不扣这项。
接近障碍不是碰撞，巡航采样时限也不是失败。

## 修改配置

| 文件 | 负责内容 |
| --- | --- |
| `my_gazebo_launch/vehicle_config.h` | 标签数量、目标速度、轮距、轮半径、雷达点数/量程/频率、物理步长 |
| `my_gazebo_launch/config.py` | 仿真资源、话题名称和默认出生位置 |
| `rl_training/config.py` | 训练输入输出地址、domain/端口、回合时限、卡住阈值、奖励系数、PPO 参数 |

这些文件开头都有集中配置区。改变标签或点数会改变网络输入输出尺寸，需要重训。
评估新格式模型时会核对模型保存的共享车辆配置；配置不一致会报错。
临时更换头文件、世界或输出目录可用命令行，不必修改源代码：

```bash
python3 rl_training/train_rl.py --config-header /absolute/path/vehicle_config.h \
  --world sdf/train_world2.sdf --output-dir /absolute/path/policies \
  --log-dir /absolute/path/logs
```

当前 `TRAIN_SPAWN_POSES`、`EVAL_SPAWN_POSES` 和 `WORKSPACE_BOUNDS` 针对
`train_world3.sdf`。更换世界时必须同步检查这些配置，避免出生点碰墙或错误越界。

## 模型和本地进度

每次训练使用新目录 `rl_models/policies/PPO_cruise_时间_随机后缀/`，保存共享配置、
运行参数、状态和已完成步数，同时输出最佳模型、最终模型、定期检查点和评估结果。
旧实验产物保留，目录结构见 [rl_models/README.md](../rl_models/README.md)。

另开已加载环境的终端查看本地 TensorBoard：

```bash
tensorboard --logdir rl_models/logs
```

打开 http://localhost:6006。除了 PPO 损失，还记录最近最多 100 个训练回合的
`cruise/collision_rate`、`stuck_rate`、`loop_rate`、`slow_rate`、`out_of_bounds_rate`、
`completion_rate`、`distance_m`、`sim_duration_sec`、`mean_speed_mps` 和
`speed_compliance`；独立评估另有 `eval/mean_reward`、`eval/success_rate`。
速度合规率表示实际前进速度落在目标 ±20% 内的动作比例。
本次运行目录的 `training_status.json` 每约 5 秒记录步数与上述指标，结束时补写
最终步数和状态。每个 `worker_*/live_state.json` 记录当前回合的仿真时间、距离、
速度、动作标签和复位状态；定期评估使用独立的 `evaluation_server/live_state.json`。
“成功率”指完成设定时长的巡航比例，不能解读为到达终点。改变巡航时限后
成功率也会变化，比较模型应保持世界、出生点、配置和评估时长一致。

单独评估已保存的模型：

```bash
python3 rl_training/train_rl.py \
  --evaluate rl_models/policies/PPO_cruise_具体运行目录/best_model.zip \
  --eval-episodes 8 --episode-steps 3000
```

上述命令从 8 个预定义起点/朝向各运行最多 600 秒仿真时间。评估只启动独立仿真，
保存逐集结束原因、距离、速度、合规率到 `evaluation_时间_后缀/evaluation.json`，
同时保存 `trajectories.npz` 和 `trajectories.png`。只有所有回合完成巡航且每集速度
合规率至少 95%，报告的 `passed` 才为真。这是指定场景的实测结果，不能代替任意
未知环境的保证。旧模型没有配置记录时会明确警告。

同一版模型继续训练：

```bash
python3 rl_training/train_rl.py --resume rl_models/policies/具体运行目录/rl_policy_final.zip \
  --timesteps 100000 --hours 1
```

最终模型续训需要同目录的 `vecnormalize.pkl`；最佳模型使用 `best_vecnormalize.pkl`，
定期检查点使用同一步数的 `rl_policy_vecnormalize_*_steps.pkl`。保留配置和状态，不能只
移动 zip 后再期望完整恢复训练。评估不归一化奖励，读取 zip 即可运行推理。

独立评估同时写入 `evaluation_status.json` 与 `live_state.json`，手机页面可以看到
已完成段数、当前段巡航时长和失败统计。评估完成或异常退出会明确更新状态。
评估时添加 `--eval-jitter-m 0.02 --eval-jitter-deg 5` 可检查起点/朝向扰动。
添加 `--gui` 则打开 Gazebo 图形界面，以正常时间观看策略驾驶；后台评估默认加速。

通过长时验证后，可冻结一份与验证报告严格对应的模型：

```bash
python3 rl_training/qualify_model.py --model rl_models/policies/具体运行目录/rl_policy_30000_steps.zip \
  --report rl_models/policies/具体评估目录/evaluation.json
```

该命令核对模型 SHA-256，要求每份报告至少 8 个场景、每集至少 300 秒、零失败、
每集速度合规率至少 95%。通过后保存为 `validated_步数_时间/model.zip`，同时保留
配套归一化状态、配置、源码快照和轨迹报告。`--report` 可以重复提供多份报告。
门槛集中在 `qualify_model.py` 开头；短程流程检查不会被当作合格驾驶模型。
同时更新本地符号链接 `rl_models/policies/latest_validated`，指向最近通过验证并冻结
的模型目录，保留之前的所有模型。训练本身不会更新这个入口。观看已验证模型：

```bash
python3 rl_training/train_rl.py --evaluate rl_models/policies/latest_validated/model.zip \
  --gui --eval-episodes 1 --episode-steps 1500
```

这会启动独立 Gazebo 窗口，最多运行 300 秒仿真时间，结束后保存评估报告并关闭。

目前尚未实现用 RL 策略自动采集 CSV 或直接导出实车 `.h`；下一阶段再接入
现有的随机森林训练流程。

## 手机进度页

本地和 GitHub Pages 使用同一套页面，前端位于 `rl_training/dashboard/`。
在 `Gazebo_simulation` 目录运行：

```bash
# 本地 + 手机局域网 + GitHub 数据同步；无需 source ROS
python3 rl_training/publish_progress.py --serve --publish
```

- 本机：<http://127.0.0.1:8765>；同一 Wi-Fi 手机：`http://电脑局域网IP:8765`。
- 外网手机：[GitHub Pages 进度页](https://taozuizui666.github.io/dissertation/)。
- 只看本地：省略 `--publish`；只同步 GitHub、不开放本地端口：`--watch --publish`。
- 只生成一次本地快照：不加参数；一次性更新远端和备用快照：`--publish --deploy`。

地址、端口、目录、刷新周期、最近回合数量、Git 分支统一在
`rl_training/progress_config.py` 顶部配置。`--models-dir`、`--logs-dir`、
`--output-dir` 可用于查看其他实验。前台服务使用 Ctrl+C 停止。
服务与训练独立，电脑保持开机且进度服务持续运行，新的训练/评估会自动出现。

手机底部有四个入口：

| 入口 | 内容 |
| --- | --- |
| 概览 | 当前回合时间、各车速度/距离、最近回合的撞墙/卡住时间、完成率、独立评估进度 |
| 曲线 | 巡航时长、奖励、完成率、碰撞率、卡住率、速度；轻点查看采样值；PPO 更新指标 |
| 回合 | 按失败原因筛选训练回合；切换已验证模型的独立验收记录 |
| 模型 | 模型 SHA-256、步数、网络、雷达/动作规格、训练设置、验收轨迹和完整训练图 |

撞墙和卡住的平均发生时间**只统计实际发生该类失败的回合**。正常达到回合时限
属于未观察到失败的截断记录，显示“≥ 时长”。因此 16 段各 1000 秒的零失败验收
意味着每段至少巡航 1000 秒，不能解释成单段连续运行 16000 秒，也不推算之后多久
会失败。训练统计默认最近 100 个已结束回合，独立验收单独统计。旧训练没有逐步
`live_state.json`，仍可显示已有回合与曲线；新训练会显示当前回合。

本地每 5 秒生成快照；GitHub 进度分支 `rl-progress-data` 约每 60 秒检查同步，
手机每 15 秒读取原始 JSON。只有数据变化时提交进度；`gh-pages` 只在网页代码或
网页配置变化、或指定 `--deploy` 时更新，避免每分钟重建 Pages。
这也避开 GitHub Pages 分支构建的[频率限制](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)。
GitHub 缓存和网络可能延长延迟，因此外网页面不承诺秒级实时。页面分别显示真实
快照时间、数据过期、备用快照和连接中断；已结束的训练显示历史结果。

发布器使用独立 detached worktree，既不切换源码分支，也不提交源码目录。
仅发布白名单统计数据、网页资源和验收图片，模型 zip、原始日志、源码快照和本机
绝对路径保留在本机。GitHub 认证沿用仓库现有 Git 配置；浏览器内不放令牌。
发布工作区如果出现人工修改或远端分歧，停止本轮同步并保留文件，不强推覆盖。
上传在独立线程运行，上传失败时本地仍继续更新，后续周期重试。

需要关闭终端后继续运行时，可使用本机的用户 systemd 服务（与前台命令二选一）：

```bash
systemd-run --user --unit=aicar-rl-progress --description='AIcar RL progress' \
  --property="WorkingDirectory=$PWD" --property=Restart=on-failure \
  /usr/bin/python3 -u "$PWD/rl_training/publish_progress.py" --serve --publish

# 查看 / 停止；重启电脑后需重新执行上面的启动命令
systemctl --user status aicar-rl-progress
journalctl --user -u aicar-rl-progress -n 30 --no-pager
systemctl --user stop aicar-rl-progress
```

同一个输出目录只允许一个发布器，重复启动会明确报错。

## 故障检查

- 首先查看本次模型目录中的 `worker_*/gazebo_train.log`、
  `evaluation_server/gazebo_eval.log` 和对应 ROS 日志。
- 提示包不存在：检查是否已构建并 `source install/setup.bash`。
- 提示端口被占用：关闭对应旧训练，或更换配置中的端口和 domain；两组必须不同。
- 提示出生点碰障碍：修改默认出生位置，确认留有整车空间，不能靠忽略接触绕过。
- 数据等待超时：检查雷达、里程计、接触插件是否加载，不能通过放宽为旧帧继续训练。
- 奖励改善不代表能实车部署；应先用固定条件的独立长时评估检查碰撞率和巡航距离，
  再核对实车标定及模型部署链路。
