# RL 模型输出

持续避障巡航训练为每次运行建立独立目录，避免覆盖已有模型。生成产物由
`Gazebo_simulation/.gitignore` 排除，训练入口和配置见
[rl_training/README.md](../rl_training/README.md)。

## 已训练模型（2026-10-10）

推荐使用 [latest_validated/model.zip](policies/latest_validated/model.zip)，当前指向
`validated_257276_20261010_022505/model.zip`。PPO 累计采样 **257,276 步**，
策略仅接收 200 列雷达距离，输出 0～14 标签；7 为直行，目标前进速度为 0.1 m/s。
从 3 万步到 25 万步连续 15 次定期评估均完成巡航，近期奖励趋于稳定，随后主动
停止续训，对最终固定权重开展了更长的独立验证。原训练目录中的 `stopped` 表示
这次主动停止，具体原因保存在 `run_config.json`。

| 最终模型独立验收 | 实测结果 |
| --- | --- |
| 世界与场景 | `train_world3.sdf`；8 种位置/方向，每种使用两组随机扰动，共 16 段 |
| 起点扰动 | x、y 各在 ±0.06 m 内，朝向在 ±15° 内；种子 20261011 |
| 单段连续时长 | 每段 1,000 秒仿真时间（16 分 40 秒） |
| 累计测试 | 16,000 秒（4 小时 26 分 40 秒）、1,599.95 m |
| 碰撞 / 卡住 / 局部绕圈 / 低速 / 越界 | 全部为 0 |
| 速度合规率 | 每段均为 100%（目标速度 ±20%） |
| 实际动作采样速度范围 | 0.099723～0.100201 m/s，共 80,000 次采样 |
| 每段平均速度 | 0.099968～0.100031 m/s |
| 完整道路巡航 | 每段约 9.32～9.41 圈，轨迹覆盖整条环形道路 |

这是 16 段独立巡航的累计时长；每段测试前恢复对应起点，复位等待未计入时长。
这组实测覆盖记录中的静态地图、车辆参数和扫描噪声；更换地图、车速或实车参数后
需重新验证。0.1 m/s 的轮速目标由共享动作接口给出，RL 学习转向；验收检查实际
运动速度与物理接触，不只检查下发的目标指令。

- [验收汇总](policies/latest_validated/validation_summary.json)及[逐段报告](policies/latest_validated/validation_1/evaluation.json)
- [轨迹与实测速度](policies/latest_validated/validation_1/trajectories.png)
- [训练曲线](policies/latest_validated/training_curve.png)及[曲线数据](policies/latest_validated/training_curve_data.json)
- [轨迹统计](policies/latest_validated/trajectory_metrics.json)、配套配置、源码快照与续训状态均在同目录

另外保留了 [11 万步备选模型](policies/validated_110000_20261010_015928/model.zip)：
8 段 × 2,000 秒，每段约 18.6 圈，累计约 1.6 km，零失败。当前固定入口使用上面的
257,276 步模型。图形界面已完成 20 秒运行检查；ROS 构建、依赖和模型散列校验通过。

在本目录的上一级 `Gazebo_simulation` 中观看最终模型：

```bash
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.bash
source install/setup.bash
python3 rl_training/train_rl.py --evaluate rl_models/policies/latest_validated/model.zip \
  --gui --eval-episodes 1 --episode-steps 1500
```

这会以正常仿真时间运行 300 秒，结束后保存报告并关闭本次窗口。
模型文件和日志仅保存在本机，没有上传。手机进度页仅同步筛选后的统计数据和验收图片。
未生成随机森林训练集，也未修改实车模型头文件。

## 输出目录

```text
rl_models/
├── policies/
│   ├── PPO_cruise_时间_随机后缀/
│   │   ├── run_config.json           # 车辆配置、训练参数、运行状态、实际步数
│   │   ├── vehicle_config.h          # 本次共享宏快照
│   │   ├── training_config.py        # 本次训练配置快照
│   │   ├── training_status.json      # 约 5 秒更新，结束时补写最终状态
│   │   ├── vecnormalize.pkl          # 最终奖励归一化状态，续训需要
│   │   ├── best_vecnormalize.pkl     # 最佳评估模型对应的归一化状态
│   │   ├── source_snapshot/          # 此次训练的 Python/C++ 源码与散列
│   │   ├── best_model.zip            # 独立评估平均奖励最佳的模型
│   │   ├── rl_policy_final.zip       # 正常完成的最终模型
│   │   ├── rl_policy_interrupted.zip # 中断/失败时尝试保存；不一定存在
│   │   ├── rl_policy_*_steps.zip     # 定期检查点
│   │   ├── evaluation/
│   │   │   └── evaluations.npz       # 训练期间独立评估的奖励、长度、成功标记
│   │   ├── worker_*/                 # Gazebo/ROS/回合日志与逐步 live_state.json
│   │   └── evaluation_server/        # 独立评估仿真日志
│   ├── evaluation_时间_后缀/
│   │   ├── evaluation.json           # 单独评估的逐集结果及是否达标
│   │   ├── evaluation_status.json    # 已完成段数、逐段失败信息、评估状态
│   │   ├── live_state.json           # 当前这一段的仿真时间、速度与距离
│   │   ├── trajectories.npz          # 逐步轨迹、速度、动作和奖励
│   │   └── trajectories.png          # 道路轨迹与实测速度图
│   ├── validated_步数_时间/
│   │   ├── model.zip                 # 与通过的长时验证报告匹配的固定权重
│   │   ├── validation_summary.json   # 验证范围、累计距离、速度、失败次数
│   │   ├── validation_*/             # 完整验证报告与轨迹
│   │   └── ...                       # 配套配置、源码和 vecnormalize.pkl
│   ├── latest_validated -> validated_步数_时间/  # 最近通过长时验证的模型
│   └── ...                          # 修正前的旧实验仍保留
├── logs/
│   └── PPO_cruise_时间_随机后缀_1/  # TensorBoard 事件
├── progress/                      # 手机页面、白名单 data.json 和验收图片
├── runtime/publisher/             # 独立的 GitHub 页面/进度发布工作区
└── README.md
```

`.zip` 是 Stable-Baselines3 的策略网络，包含权重、超参数和观测/动作空间；
共享硬件配置另存于同目录 `run_config.json`，评估时一并校验，请保留在一起。
旧实验没有这些配置记录，碰撞、复位和评估逻辑也不同，不能直接比较为新训练成果。

在 `Gazebo_simulation` 内加载 ROS 和本工程环境后执行：

```bash
python3 rl_training/train_rl.py \
  --evaluate rl_models/policies/latest_validated/model.zip \
  --eval-episodes 8 --episode-steps 3000
```

报告中的正常时间截断表示完成设定时长的巡航；碰撞、卡住、持续低速、小范围
反复绕圈或越界表示失败。上述命令每集最多运行 600 秒仿真时间。
训练时“最佳”指评估奖励最高，仍需结合碰撞率、距离和持续时间判断质量。
`latest_validated` 只由独立验证通过后的冻结步骤更新，普通训练不会覆盖这个入口。
目录根部旧的 `best_model.zip` 来自之前的实验，当前应使用上述已验证模型入口。

RL 模型不直接烧录到 Teensy。后续由策略采集与实车格式一致的雷达/标签数据，
再通过 `software/MLcode/train_model.py` 训练并导出随机森林头文件。
自动采集 CSV 尚未实现，本目录不提供 `collect_data.py` 或 `eval_model.py`。
