# 特权环境场模型

对应代码：[rl_training/privileged](../../rl_training/privileged/README.md)。

## 当前已验证教师

固定入口：`policies/latest_validated/model.zip`。
冻结目录：`policies/validated_privileged_2764876_20261101_201259/`。
PPO，292 维局部环境场与状态，128×128 隐藏层，15 个动作，目标速度 0.2 m/s；训练与评估关闭雷达。

五图 × 两个新种子 × 八个正反向扰动起点 × 300 秒：**80/80 通过严格轨迹审计**，约 4,799.39 m、24,000 仿真秒，速度合规率 100%。

| 地图 | 合格回合 | 最低覆盖分箱 | 最小保守净空 |
| --- | --- | --- | --- |
| 直角矩形 | 16/16 | 100/100 | 1.87 cm |
| 真正 S 弯 | 16/16 | 100/100 | 8.59 cm |
| 圆角矩形 | 16/16 | 100/100 | 7.78 cm |
| 椭圆 | 16/16 | 100/100 | 8.54 cm |
| 8 字 | 16/16 | 100/100 | 5.37 cm |

8 字每回合 7 次正确对侧穿越及换环。采用中间检查点；同次训练 2,770,116 步最终模型发生矩形碰撞，不用于采集。

证据：[严格审计](policies/latest_validated/audit.json)、[逐回合统计](policies/latest_validated/audit_episodes.csv)、[协议](policies/latest_validated/protocol.json)、[冻结清单](policies/latest_validated/map_manifest.json)。
完整开发过程见 [8 字教师修复记录](../../../article/experiment_data/P0_FIGURE_EIGHT_ROUTE_README.md)。模型和原始评估数据保留在本地。

```bash
python rl_training/privileged/visualize.py --check
python rl_training/privileged/visualize.py
```

默认依次观看直角矩形、真正 S 弯、圆角矩形、椭圆和 8 字，每图最多 180 仿真秒。
这是五张简单图的有限协议验收，未验证组合图、随机森林或实车。随机森林采集配置仍需切换至本教师并先检查，不能假定旧采集配置已同步。

## 历史教师：2026-10-11 单图验收

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

证据：[验收汇总](policies/validated_privileged_2392588_20261011_074049/validation_summary.json)、
[逐回合报告](policies/validated_privileged_2392588_20261011_074049/validation_1/evaluation.json)、
[轨迹和速度图](../../results/training/privileged/policies/validated_privileged_2392588_20261011_074049/validation_1/trajectories.png)、
[本轮训练曲线](../../results/training/privileged/policies/validated_privileged_2392588_20261011_074049/training_curve.png)。
这些本地文件未上传 Git。选定检查点经过验收后冻结，因此不一定是最后一步权重。

在 `Gazebo_simulation` 目录加载 ROS 与项目环境后观看 3 分钟：

```bash
python rl_training/privileged/train.py \
  --evaluate rl_models/privileged/policies/validated_privileged_2392588_20261011_074049/model.zip \
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

距离场由 SDF 静态障碍几何及物理真值生成。该历史批次训练实例使用同一张
`train_world3.sdf`，已有验收也在这张地图上。因此尚未验证跨地图泛化；
通过本次验收不等于新地图永不碰撞，也不等于未来雷达学生或实车能复现该表现。
