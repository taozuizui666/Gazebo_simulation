# 工程目录与迁移说明

返回 [项目首页](../README.md)。

## 按用途找文件

- 仿真运行代码：[my_gazebo_launch/](../my_gazebo_launch)，包括车辆宏、控制和雷达转换。
- 统一启动入口：[launch/train_world.launch.py](../launch/train_world.launch.py)。
- 正式地图、车辆模板、复用模型：[assets/](../assets/README.md)。
- 雷达 RL / 特权 RL：[rl_training/](../rl_training/README.md)。
- 按类型保存的模型与证据：[rl_models/](../rl_models/README.md)。
- 资源编辑工具：[tools/](../tools/README.md)。
- 旧车辆与历史导出素材：[archive/](../archive/README.md)。
- 环境和运行命令：[simulation.md](simulation.md)。
- 车辆参数、PID 轮速接口及雷达协议：[vehicle_and_lidar.md](vehicle_and_lidar.md)。

## 2026-10-11 资源迁移

| 原位置 | 当前位置 |
| --- | --- |
| `sdf/train*.sdf` | `assets/worlds/` 中的同名地图 |
| `urdf/simple_car1.urdf` | `assets/urdf/simple_car1.urdf` |
| `models/wall/` | `assets/models/wall/` |
| `sdf/wall.sdf` | 与墙体模型内容相同，删除重复副本 |
| `urdf/simple_car.urdf`、`urdf/simple_car2.txt` | `archive/vehicles/urdf/` |
| `meshes/` | `archive/vehicles/meshes/` |
| `sdf/O_size/`、`sdf/Untitled/`、`sdf/test1/`、`sdf/test2/` | `archive/world_exports/` 中的同名目录 |
| `scripts/adjust_wall_height.py` | `tools/adjust_wall_height.py`，输入 / 输出路径可配置 |
| `launch/gazebo_simulation.py`、`launch/spawn_car.launch.py` | 删除转发入口，统一使用 `train_world.launch.py` |
| `test/` 三个版权 / 格式模板检查 | 删除模板及包中对应的测试依赖声明 |

ROS 包名仍为 `my_gazebo_launch`；共享 `vehicle_config.h`、两类 RL 入口、模型目录、
网页服务入口的位置不变。现有命令中显式指定地图时，把 `sdf/文件名` 更新为
`assets/worlds/文件名`。

## 构建文件与历史结果

`setup.py` 定义 Python 包、节点入口及资源安装；`setup.cfg` 指定 ROS 节点安装目录；
`package.xml` 声明依赖；`resource/my_gazebo_launch` 是 ROS 包索引标记。这些文件留在包根目录。

`build/`、`install/`、`log/` 分别是构建中间文件、安装产物和构建日志，
`__pycache__/` 是 Python 缓存。它们按现有习惯由工具生成并被 Git 忽略。
其中 `build/rl_native/` 是 RL 物理步进插件的编译缓存。

资源迁移保持地图、车辆和历史素材内容一致。两类已训练权重、归一化状态、日志、
验收报告及训练源码快照保持原样。旧报告中的 `sdf/` 路径记录当时环境，
不会改写成新的路径；当前入口已使用 `assets/`。
之前 RL 文件分类的本机路径对照仍在 `rl_models/runtime/layout_migration.json`。
