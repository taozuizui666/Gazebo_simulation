# 历史素材存档

本目录保存旧资源原件，不加入正式 ROS 包资源安装或 Gazebo 模型搜索路径。
日常使用 [assets/](../assets/README.md) 中的资源。

| 目录 | 原位置及内容 |
| --- | --- |
| [vehicles/urdf/](vehicles/urdf) | 原 `urdf/simple_car.urdf`、`urdf/simple_car2.txt` |
| [vehicles/meshes/](vehicles/meshes) | 原 `meshes/`；包含旧车辆使用的 STL |
| [world_exports/O_size/](world_exports/O_size) | 原 `sdf/O_size/` 编辑器导出模型 |
| [world_exports/Untitled/](world_exports/Untitled) | 原 `sdf/Untitled/`，包含两个不同版本的导出素材 |
| [world_exports/test1/](world_exports/test1) | 原 `sdf/test1/` 导出场景 |
| [world_exports/test2/](world_exports/test2) | 原 `sdf/test2/` 导出场景 |

文件内容保持原样，包括旧 URDF 中记录的 `/home/zui/ros2_ws` 网格路径。
这类路径不适用于当前工程；若以后重新启用旧资源，需要先调整引用并验证车辆行为。
两个 `Untitled/model.sdf` 内容不同，因此都保留。

原来的两个启动转发文件和三个 ROS 模板测试已移除，不在此存放旧代码副本。
历史训练源码仍随各模型保存在 `rl_models/对应类型/policies/批次/source_snapshot/`。
