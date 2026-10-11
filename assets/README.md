# 仿真资源

正式启动与 RL 都从这里加载资源。默认选择由
[my_gazebo_launch/config.py](../my_gazebo_launch/config.py) 顶部常量决定，
车辆运行参数由 [vehicle_config.h](../my_gazebo_launch/vehicle_config.h) 决定。

| 路径 | 用途 |
| --- | --- |
| [worlds/simple/train_world3.sdf](worlds/simple/train_world3.sdf) | 当前训练、评估与手动仿真的默认地图 |
| [worlds/simple/train_world3.json](worlds/simple/train_world3.json) | 当前地图的训练/评估出生点和工作边界 |
| `worlds/simple/` | 原手绘地图及全部简单地图的 SDF/JSON；预览位于 `../results/maps/simple/` |
| `worlds/complex/` | 预留复杂地图目录 |
| [urdf/simple_car1.urdf](urdf/simple_car1.urdf) | 当前车辆几何模板，启动时套用共享车辆参数 |
| [models/wall/](models/wall) | `model://wall` 所需的模型描述与 SDF |

`setup.py` 按同样相对路径安装本目录。启动入口把 `assets/models/` 加入
Gazebo 模型搜索路径；地面和光源使用本机 Gazebo 资源。

原手绘地图是 `train_world3.sdf`，旧启动代码和训练记录均使用它。
旧简单场景和临时预览地图已移至 `archive/worlds/`。
历史编辑器导出文件及旧车辆素材位于 [archive/](../archive/README.md)，不参与正式安装。

切换地图示例（在 `Gazebo_simulation` 目录）：

```bash
ros2 launch my_gazebo_launch train_world.launch.py \
  world:=assets/worlds/simple/ellipse_002.sdf
```

RL 通过 `--world` 选择地图，自动读取同名 JSON 的出生点和工作边界；默认地图在
`rl_training/common/config.py` 顶部配置。出生点、边界也可用命令行覆盖，详见
[RL 使用说明](../rl_training/README.md#选择地图与出生点)。
手动仿真未指定位置时读取该地图 JSON 的第一个训练出生点；没有 JSON 时显式传入 x/y/yaw。
批量生成地图参见 [工具说明](../tools/README.md#批量生成训练地图)，新地图需重新评估模型。
修改墙高参见 [工具说明](../tools/README.md)。
