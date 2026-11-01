# 仿真资源

正式启动与 RL 都从这里加载资源。默认选择由
[my_gazebo_launch/config.py](../my_gazebo_launch/config.py) 顶部常量决定，
车辆运行参数由 [vehicle_config.h](../my_gazebo_launch/vehicle_config.h) 决定。

| 路径 | 用途 |
| --- | --- |
| `worlds/simple/` | 待生成的统一宽度六类简单地图及同名 JSON |
| `worlds/complex/` | 待由简单地图组合生成的统一宽度组合地图及同名 JSON |
| [urdf/simple_car1.urdf](urdf/simple_car1.urdf) | 当前车辆几何模板，启动时套用共享车辆参数 |
| [models/wall/](models/wall) | `model://wall` 所需的模型描述与 SDF |

`setup.py` 按同样相对路径安装本目录。启动入口把 `assets/models/` 加入
Gazebo 模型搜索路径；地面和光源使用本机 Gazebo 资源。

旧地图与预览已清除。下一轮会生成道路宽度统一的直线、直角弯、S 弯、圆角弯、椭圆弯和 8 字路口；组合地图只由这些场景构成。

RL 通过 `--world` 选择地图，并自动读取同名 JSON 的出生点和工作边界。新地图生成后，再将默认地图和训练配置指向对应文件。历史编辑器导出文件及旧车辆素材仍位于 [archive/](../archive/README.md)，不参与正式安装。
