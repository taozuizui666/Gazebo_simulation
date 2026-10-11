# 雷达输入 RL

这一类是最初的雷达 PPO 策略。网络读取 200 列规范雷达距离，按量程归一化，
再由 [features.py](features.py) 做平方根变换；不输入地图距离场。

- 入口：[train.py](train.py)。
- 模型、日志及运行路径：[config.py](config.py) 顶部。
- 雷达点数、采集扇区、距离单位与标签：共享 [vehicle_config.h](../../my_gazebo_launch/vehicle_config.h)。
- 历史模型与验收结果：[rl_models/lidar](../../rl_models/lidar/README.md)。

在 `Gazebo_simulation` 目录加载 ROS 与项目环境后：

```bash
python rl_training/lidar/train.py
python rl_training/lidar/train.py \
  --evaluate rl_models/lidar/policies/latest_validated/model.zip \
  --gui --eval-episodes 1 --episode-steps 900
```

当前已验证模型为 257,276 步、0.1 m/s。评估和续训自动使用模型随附车辆配置；
新训练使用共享头文件当前设置，当前是 0.2 m/s。若想重新训练 0.1 m/s 的雷达对照，
可显式指定 `--config-header rl_models/lidar/policies/latest_validated/vehicle_config.h`。

PPO、奖励与仿真配置共用 [common/config.py](../common/config.py)。
完整训练、续训、验收和网页命令见 [总说明](../README.md)。
