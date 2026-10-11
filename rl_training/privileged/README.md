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
更换地图还需调整共用配置中的出生点和工作边界，并重新验证泛化表现。

后续若用教师生成随机森林数据，仍需在同一车辆状态下配对雷达帧和教师动作；
本目录负责教师训练，当前没有启动这项数据采集。

PPO、奖励与仿真配置共用 [common/config.py](../common/config.py)。
完整训练、续训、验收和网页命令见 [总说明](../README.md)。
