# 仿真资源

正式启动与 RL 都从这里加载资源。默认选择由
[my_gazebo_launch/config.py](../my_gazebo_launch/config.py) 顶部常量决定，
车辆运行参数由 [vehicle_config.h](../my_gazebo_launch/vehicle_config.h) 决定。

| 路径 | 用途 |
| --- | --- |
| [worlds/train_world3.sdf](worlds/train_world3.sdf) | 当前训练、评估与手动仿真的默认地图 |
| [worlds/train_world2.sdf](worlds/train_world2.sdf) | 保留的备用地图版本，需显式选择 |
| [worlds/train_world.sdf](worlds/train_world.sdf) | 保留的早期地图，需显式选择 |
| [worlds/train.sdf](worlds/train.sdf) | 使用 `model://wall` 的简单场景 |
| [urdf/simple_car1.urdf](urdf/simple_car1.urdf) | 当前车辆几何模板，启动时套用共享车辆参数 |
| [models/wall/](models/wall) | `model://wall` 所需的模型描述与 SDF |

`setup.py` 按同样相对路径安装本目录。启动入口把 `assets/models/` 加入
Gazebo 模型搜索路径；地面和光源使用本机 Gazebo 资源。

这些地图和当前车辆模板在目录整理中仅移动位置，文件内容保持原样。
历史编辑器导出文件及旧车辆素材位于 [archive/](../archive/README.md)，不参与正式安装。

切换地图示例（在 `Gazebo_simulation` 目录）：

```bash
ros2 launch my_gazebo_launch train_world.launch.py \
  world:=assets/worlds/train_world2.sdf x:=0 y:=0 yaw:=0
```

不同地图需选择合适出生点。用于 RL 时还需更新训练出生点和工作边界，并重新评估。
修改墙高参见 [工具说明](../tools/README.md)。
