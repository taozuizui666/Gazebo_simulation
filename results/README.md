# 结果图与报告

所有供查看的结果统一从这里进入，PNG 与 PDF 放在同一批次目录。

| 目录 | 内容 |
| --- | --- |
| [maps/simple/](maps/simple/) | 正式生成地图预览、Gazebo 截图和演示评估报告 |
| [maps/matched_preview/](maps/matched_preview/) | 早期临时地图的预览、截图和演示报告 |
| [training/lidar/policies/](training/lidar/policies/) | 雷达模型各批次训练/评估图 |
| [training/privileged/policies/](training/privileged/policies/) | 特权模型各批次训练/评估图 |
| `training/progress/` | 本地进度网页和展示图片，由发布器生成 |
| `training/external/` | 自定义模型输出目录对应的结果，保留绝对路径层级以避免重名 |
| `datasets/` | 每批次按地图合并多轮采集，raw/、dropout/ 分开存放；CSV 按相同结构复制到 dissertation/dataset/simulation，元信息与时序只留在此处 |

两类 `policies/latest_validated/` 指向当前已验证模型的结果批次。
权重、日志、JSON 报告及 NPZ 原始轨迹仍在 `rl_models/` 对应批次中。
地图 SDF/JSON 位于 `assets/worlds/`，临时旧地图位于 `archive/worlds/`。

地图生成器和独立评估同时保存 PNG/PDF；历史 PNG 原样保存，不伪造矢量 PDF。
截图仍为 PNG。生成图像由 Git 忽略，本索引纳入版本管理。
