# RL 模型结果索引

模型、日志和运行记录按输入类型分开保存：

| 类型 | 结果与证据 | 对应代码 | 当前已验证模型 |
| --- | --- | --- | --- |
| 雷达输入 | [lidar/](lidar/README.md) | [rl_training/lidar/](../rl_training/lidar/README.md) | 257,276 步，0.1 m/s |
| 特权环境场 | [privileged/](privileged/README.md) | [rl_training/privileged/](../rl_training/privileged/README.md) | 2,392,588 步，0.2 m/s |

每一类都包含：

```text
lidar/ 或 privileged/
├── README.md
├── policies/           模型检查点、独立评估、已验证模型
│   └── latest_validated -> 本类最近通过验收的模型
├── logs/               TensorBoard 与运行日志
└── runtime/            本类临时运行文件
```

训练权重、日志及运行文件留在本机，由 Git 忽略。说明文件可提交。
普通训练不会覆盖已验证模型，只有独立长时验收达标后才能更新固定入口。

结果图（含 PDF）集中在 [../results/training/](../results/training/)，保留类型、批次和验收编号。
共用网页快照位于 `../results/training/progress/`，GitHub 发布工作区位于 `runtime/publisher/`。
页面顶部可以切换模型类别，两类的统计、验收和图片分别展示。

2026-10-11 分类迁移保留了历史 `policies/`、`logs/` 文件的原始内容，
包括权重、车辆参数、报告及源码快照。旧 JSON 中的绝对路径是历史来源记录；
按最长目录前缀查本机 `runtime/layout_migration.json` 可找到新位置。
旧源码快照不作为当前入口运行。

当前训练、续训、评估命令见 [训练说明](../rl_training/README.md)。
两类已有验收均针对记录中的地图与车辆条件，尚不能作为新地图或实车泛化保证。
