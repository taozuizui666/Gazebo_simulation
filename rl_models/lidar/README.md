# 雷达输入模型

对应代码：[rl_training/lidar/](../../rl_training/lidar/README.md)。
固定模型入口：`policies/latest_validated/model.zip`。

## 当前已验证模型

目录：`policies/validated_257276_20261010_022505/`。

| 项目 | 记录 |
| --- | --- |
| 算法 / 输入 | PPO；200 列雷达距离 |
| 权重采样步数 | 257,276 |
| 目标前进速度 | 0.1 m/s |
| 动作 | 0～14，7 为直行 |
| 独立验收 | 16 段，每段 1,000 秒 |
| 总距离 / 仿真时间 | 1,599.946 m / 16,000 秒 |
| 碰撞、卡住、局部绕圈、低速、越界 | 均为 0 |
| 每段速度合规率 | 100% |

SHA-256：
`d72907a1218e6de2b9881747dd3f28562afc32f480a2840e2be6130c971928a1`。

证据：[验收汇总](policies/latest_validated/validation_summary.json)、
[逐回合报告](policies/latest_validated/validation_1/evaluation.json)、
[轨迹和速度图](policies/latest_validated/validation_1/trajectories.png)。
这些本地文件未上传 Git。上述结果是多段独立验收，不代表一段连续运行 16,000 秒。

在 `Gazebo_simulation` 目录加载 ROS 与项目环境后观看：

```bash
python rl_training/lidar/train.py \
  --evaluate rl_models/lidar/policies/latest_validated/model.zip \
  --gui --eval-episodes 1 --episode-steps 900
```

默认自动使用模型目录里的 0.1 m/s 车辆配置。模型复制或备份应保留整个已验证目录，
包括配套归一化状态、配置、报告及源码快照。

## 其他记录

`policies/PPO_cruise_*/` 是各轮训练；
`policies/evaluation_*/` 是独立评估；
其他 `validated_*/` 是之前冻结的版本；
`logs/` 保存 TensorBoard 和运行日志。

`policies/legacy/best_model.zip` 是更早的历史实验权重，没有本版巡航验收依据。
它在分类迁移前就因 `numpy._core.numeric` 依赖不兼容而无法在当前环境加载；
仅保留存档，不是上面可以正常加载的已验证模型。

验收范围限于记录中的地图、起点和车辆参数。新增地图须重新评估。
