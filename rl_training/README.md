# 持续避障巡航训练

任务是在固定前进速度下选择转向、持续避障巡航。没有终点；碰撞、卡住、持续低速、
小范围反复绕圈或越界都判为失败。正常达到采样时限则截断，下一集返回出生点。

本分支默认训练 **PPO 特权教师**，完全关闭雷达传感器。教师读取静态障碍地图和
Gazebo 物理真值；当前输入为 292 维，动作 0～14、7 直行，目标速度 **0.2 m/s**。
速度和动作规格直接读取 [vehicle_config.h](../my_gazebo_launch/vehicle_config.h)。
原来的雷达 PPO 可用 `--observation-mode lidar` 作为对照；两种模型分别保存、验证。

教师未来提供动作标签，随机森林未来只用雷达作为推理输入。这两个阶段必须配对同一
时刻的雷达和教师动作，并共用动作定义、速度和动力学。教师可见地图与真实状态，
其成功不能保证只有雷达的学生能完全模仿。本阶段不采集 CSV，也不生成 `randomForest.h`。

旧雷达模型入口 `rl_models/policies/latest_validated/model.zip` 保留；新的特权模型
只有通过长时验收后才发布到 `rl_models/policies/latest_validated_privileged/model.zip`。
当前已冻结 42,588 步教师：8 个扰动起点各巡航 600 秒，零失败、速度合规率均为 100%。
完整范围、报告和观看命令见 [已验证特权教师](../rl_models/README.md#已验证特权教师2026-10-11)。

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

本机 2026-10-11 的 6 小时续训由用户服务 `aicar-rl-train-6h-20261011.service`
运行，从已验证的 42,588 步特权教师继续。训练预算为北京时间约 01:25～07:25，
包括定期评估；每新增 10,000 步保存检查点，每 50,000 步评估全部 8 个起点。
结束后另行对定期评估最佳权重和最终权重做 8 起点 × 600 秒、±0.06 m / ±15°
的独立验收，按通过验收的候选中平均回报较高者更新固定模型入口；均未通过则保留原模型。
验收耗时在训练预算之外，结果会继续同步到网页。
本次运行编排与状态保存在被 Git 忽略的 `rl_models/runtime/train_privileged_6h.py`
和 `rl_models/runtime/privileged_6h_campaign.json`，模型目录为
`rl_models/policies/PPO_privileged_20261011_012519_9587cf/`。

```bash
systemctl --user status aicar-rl-train-6h-20261011.service
journalctl --user -u aicar-rl-train-6h-20261011.service -n 40 --no-pager
# 如需提前停止并保存本轮训练，创建 STOP；此时不继续自动验收：
touch rl_models/policies/PPO_privileged_20261011_012519_9587cf/STOP
```

## 观测、动作和物理步进

默认特权教师的输入为：

| 输入 | 含义 |
| --- | --- |
| 17×17 局部距离场 | 车后 0.4 m 至前方 1.4 m、左右各 0.9 m；各网格点到静态障碍的带符号距离，按 0.5 m 截断、归一化到 [-1, 1] |
| 实际前进速度 | Gazebo 真实线速度变换到车身坐标系、按目标速度归一化 |
| 实际角速度 | Gazebo 真实角速度、按最大目标转向速度归一化 |
| 上一动作 | 按标签数量归一化；复位时为中间值 |

合计 `17×17+3=292` 维，使用 `[128, 128]` MLP。距离场由 SDF 碰撞几何直接计算，
不做射线扫描，不使用雷达的范围、盲区、噪声或频率。位姿用于把地图变换到车身坐标系，
绝对位置和朝向不直接作为网络输入。奖励归一化使用 `VecNormalize`，观测不额外归一化。

[privileged.py](privileged.py) 顶部可修改网格尺寸、范围和归一化等参数。当前解析器
支持静态 box 障碍以及 model/link/collision 的平面位移和旋转。遇到动态障碍、mesh、
非地面/太阳的外部 include、倾斜或自定义参考坐标系会报错，避免遗漏真实碰撞几何。
更换地图时还需同步出生点与边界配置。新模型记录完整观测规格和世界文件 SHA-256。

旧雷达基线使用 `--observation-mode lidar`：200 个整数毫米距离除以上界，网络内开平方，
只保留 0～49、151～199，其余为零；量化、列序和有效性规则保持原样。它不输入地图或
真实速度。雷达量程说明见 [主 README](../README.md#雷达输入协议)。

动作经共享公式转换为左右轮目标速度。默认每动作推进 **0.2 秒仿真时间**，物理步长
0.001 秒。特权教师的步长不受雷达频率影响；只有雷达基线会为等待新扫描增大动作时长。
其余时间暂停物理世界。电脑算力改变训练耗时，不改变速度的物理意义。

原生插件在一个动作内检查每个物理步的接触，并在推进结束时同步返回仿真时间、碰撞、
位姿和速度。特权观测与奖励使用这一份对应状态，完全不依赖异步雷达/里程计消息。
雷达基线额外等待本次动作后的新扫描和末尾里程计。缺少状态、时间不符或仿真退出时
停车报错，不能伪造全零或使用旧帧继续训练。控制指令有独立心跳。

## 回合与奖励

| 条件 | 行为 |
| --- | --- |
| 车身或左右轮接触障碍物 | `terminated=True`，碰撞罚分，停车 |
| 3 秒仿真时间内实际移动不足 0.02 m | `terminated=True`，卡住罚分，停车 |
| 3 秒内平均前进速度低于目标的 75% | `terminated=True`，低速失败 |
| 至少 6 秒、最近 12 秒轨迹范围小于 0.65 m，累计转角达到 5.5 rad | `terminated=True`，小范围绕圈失败 |
| 超出配置的道路边界 | `terminated=True`，越界失败 |
| 正常达到 750 步 | `truncated=True`，记录完成一段巡航，无额外失败罚分 |
| 调用 `reset()` | 停车、暂停、复位关节、恢复出生位置/朝向并清零速度，等待新的对应模式状态 |

地面及车辆自身接触不计作撞墙。复位保留单调递增的仿真时钟，但清空旧传感器
缓存和回合统计。正常 750 步约为 150 秒巡航，不含复位稳定时间。
所有标签都保持两轮向前，不提供停止、倒车或原地旋转动作。小范围绕圈检测使用
一段时间内的轨迹包围框和实际累计转角；沿整个环形道路正常跑圈不属于这种行为。

奖励项在 [reward.py](reward.py) 中，系数集中放在 [config.py](config.py) 顶部：

| 奖励项 | 默认规则 |
| --- | --- |
| 巡航距离 | 本步有效前进米数 × 10；同时受实际前进速度和目标速度上限约束 |
| 速度误差 | −3 × abs(实际速度 − 目标速度) × 仿真时间 |
| 近障碍 | 特权模式用地图到车体的保守间距；雷达基线用最小有效距离；低于 250 mm 时按接近程度扣分 |
| 转向变化 | −0.1 × abs(本次标签 − 上次标签) / (标签数量 − 1) |
| 前方风险 | 特权模式用地图计算沿正前方扫掠圆盘的可行距离；雷达基线用前方 ±20° 有效距离；小于 450 mm 时按接近程度平方扣分 |
| 持续大转向 | 按归一化转向幅度平方及仿真时间轻微扣分 |
| 失败 | 碰撞、卡住、低速、绕圈、越界均为 −20，并取消本步距离奖励；失败罚分不叠加 |

保持同一转向不会产生变化惩罚；第一步没有上一动作，也不扣这项。
接近障碍不是碰撞，巡航采样时限也不是失败。特权奖励的车体圆盘半径默认 0.085 m，
只用于保守间距塑形，实际碰撞始终由 Gazebo 的整车接触决定。

## 修改配置

| 文件 | 负责内容 |
| --- | --- |
| `my_gazebo_launch/vehicle_config.h` | 标签数量、目标速度、轮距、轮半径、雷达点数/量程/频率、物理步长 |
| `my_gazebo_launch/config.py` | 仿真资源、话题名称和默认出生位置 |
| `rl_training/config.py` | 默认观测模式、输入输出地址、domain/端口、回合时限、卡住阈值、奖励系数、PPO 参数 |
| `rl_training/privileged.py` | 教师距离场的网格、范围、归一化与奖励安全间距 |

这些文件开头都有集中配置区。改变标签或教师网格会改变特权网络尺寸，需要重训。雷达点数只影响雷达网络尺寸。
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

每次训练使用新目录 `rl_models/policies/PPO_privileged_时间_随机后缀/`（雷达基线为 `PPO_cruise_*`），保存共享配置、
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
  --evaluate rl_models/policies/PPO_privileged_具体运行目录/best_model.zip \
  --eval-episodes 8 --episode-steps 3000
```

上述命令从 8 个预定义起点/朝向各运行最多 600 秒仿真时间。评估只启动独立仿真，
保存逐集结束原因、距离、速度、合规率到 `evaluation_时间_后缀/evaluation.json`，
同时保存 `trajectories.npz` 和 `trajectories.png`。只有所有回合完成巡航且每集速度
合规率至少 95%，报告的 `passed` 才为真。这是指定场景的实测结果，不能代替任意
未知环境的保证。旧模型没有配置记录时会明确警告。

评估和续训会从模型记录自动识别观测模式；显式指定不匹配的模式会报错。特权模型
还会核对网格、范围和归一化定义，不能把雷达网络直接当作特权模型继续训练。

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
图形界面打开后等待 `GUI_STARTUP_SETTLE_SEC`（默认 3 秒）再开始复位，避免界面
初始化控制消息与首次物理步进冲突。评估报告同时记录仿真时长和实际运行耗时。

通过长时验证后，可冻结一份与验证报告严格对应的模型：

```bash
python3 rl_training/qualify_model.py --model rl_models/policies/具体运行目录/rl_policy_30000_steps.zip \
  --report rl_models/policies/具体评估目录/evaluation.json
```

该命令核对模型 SHA-256，要求每份报告至少 8 个场景、每集至少 300 秒、零失败、
每集速度合规率至少 95%。特权模型还必须以关闭雷达、相同观测规格验收。通过后保存为
`validated_privileged_步数_时间/model.zip`（雷达基线为 `validated_*`），同时保留
配套归一化状态、配置、源码快照和轨迹报告。`--report` 可以重复提供多份报告。
门槛集中在 `qualify_model.py` 开头；短程流程检查不会被当作合格驾驶模型。
特权模型更新本地符号链接 `rl_models/policies/latest_validated_privileged`；雷达基线更新
`rl_models/policies/latest_validated`。二者分别指向最近通过验证并冻结
的模型目录，保留之前的所有模型。训练本身不会更新这个入口。观看已验证模型：

```bash
python rl_training/train_rl.py --evaluate rl_models/policies/latest_validated_privileged/model.zip \
  --config-header rl_models/policies/latest_validated_privileged/vehicle_config.h \
  --gui --eval-episodes 1 --episode-steps 900
```

这会启动独立 Gazebo 窗口，最多运行 180 秒仿真时间（3 分钟），结束后保存评估报告并关闭。

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
| 模型 | 模型 SHA-256、步数、网络、特权/雷达观测与动作规格、训练设置、验收轨迹和完整训练图 |

撞墙和卡住的平均发生时间**只统计实际发生该类失败的回合**。正常达到回合时限
属于未观察到失败的截断记录，显示“≥ 时长”。因此 16 段各 1000 秒的零失败验收
意味着每段至少巡航 1000 秒，不能解释成单段连续运行 16000 秒，也不推算之后多久
会失败。训练统计默认最近 100 个已结束回合，独立验收单独统计。旧训练没有逐步
`live_state.json`，仍可显示已有回合与曲线；新训练会显示当前回合。

本地每 5 秒生成快照；GitHub 进度分支 `rl-progress-data` 约每 60 秒检查同步，
手机每 15 秒读取原始 JSON。只有数据变化时提交进度；`gh-pages` 只在网页代码或
网页配置变化、或指定 `--deploy` 时更新，避免每分钟重建 Pages。
新训练同时记录时间预算、剩余时间和预计结束训练时间；概览进度条按时间或步数中
先达到的预算显示，便于查看按小时运行的训练。剩余时间以最近快照为准。
这也避开 GitHub Pages 分支构建的[频率限制](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)。
GitHub 缓存和网络可能延长延迟，因此外网页面不承诺秒级实时。页面分别显示真实
快照时间、数据过期、备用快照和连接中断；已结束的训练显示历史结果。

发布器使用独立 detached worktree，既不切换源码分支，也不提交源码目录。
仅发布白名单统计数据、网页资源和验收图片，模型 zip、原始日志、源码快照和本机
绝对路径保留在本机。GitHub 认证沿用仓库现有 Git 配置，也尊重 `core.sshCommand`
中的端口等设置；浏览器内不放令牌。
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
- 数据等待超时：检查原生步进/接触插件；雷达模式还需检查扫描与里程计，不能使用旧帧继续训练。
- 奖励改善不代表能实车部署；应先用固定条件的独立长时评估检查碰撞率和巡航距离，
  再核对实车标定及模型部署链路。
