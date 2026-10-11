# 特权环境场模型

对应代码：[rl_training/privileged/](../../rl_training/privileged/README.md)。
固定模型入口：`policies/latest_validated/model.zip`。

## 当前已验证教师

目录：`policies/validated_privileged_2392588_20261011_074049/`。

| 项目 | 记录 |
| --- | --- |
| 算法 / 输入 | PPO；17×17 距离场 + 3 维状态，共 292 维 |
| 雷达 | 训练与评估均关闭 |
| 选定权重采样步数 | 2,392,588 |
| 本轮训练最后累计步数 | 2,500,520 |
| 本轮训练预算 | 六小时；2026-10-11 01:25 至 07:25（北京时间） |
| 目标前进速度 | 0.2 m/s |
| 动作 | 0～14，7 为直行 |
| 独立验收 | 8 个扰动起点，每段 600 秒 |
| 起点扰动 / 种子 | ±0.06 m、±15°；20261012 |
| 总距离 / 仿真时间 | 959.902 m / 4,800 秒 |
| 碰撞、卡住、局部绕圈、低速、越界 | 均为 0 |
| 每段速度合规率 | 100% |

SHA-256：
`7171b17cb1e492ec0434887bbb0128e0246ae0ce0d15875f1854a14d246b3365`。

证据：[验收汇总](policies/latest_validated/validation_summary.json)、
[逐回合报告](policies/latest_validated/validation_1/evaluation.json)、
[轨迹和速度图](policies/latest_validated/validation_1/trajectories.png)、
[本轮训练曲线](policies/latest_validated/training_curve.png)。
这些本地文件未上传 Git。选定检查点经过验收后冻结，因此不一定是最后一步权重。

在 `Gazebo_simulation` 目录加载 ROS 与项目环境后观看 3 分钟：

```bash
python rl_training/privileged/train.py \
  --evaluate rl_models/privileged/policies/latest_validated/model.zip \
  --gui --eval-episodes 1 --episode-steps 900
```

默认自动使用模型目录里的车辆配置。已有 3 分钟 GUI 演示记录保存在
`policies/evaluation_20261011_102558_0b8f27/`：180 秒仿真时间、约 35.983 m、
平均 0.19994 m/s，零失败。该演示与长时独立验收分别保留。

## 其他记录与范围

`policies/PPO_privileged_20261011_012519_9587cf/` 是六小时续训批次；
`validated_privileged_42588_20261011_011849/` 是更早的教师，保留在 `policies/`。
`logs/` 保存 TensorBoard；`runtime/` 保存该次六小时任务脚本、状态和 GUI 诊断文件。
日常训练使用上述正式入口；历史任务状态中的路径可查上级迁移清单。

距离场由 SDF 静态障碍几何及物理真值生成。当前训练实例使用同一张
`train_world3.sdf`，已有验收也在这张地图上。因此尚未验证跨地图泛化；
通过本次验收不等于新地图永不碰撞，也不等于未来雷达学生或实车能复现该表现。
