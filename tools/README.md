# 资源编辑工具

[adjust_wall_height.py](adjust_wall_height.py) 修改地图中 `Wall*` 链接的 box 墙体高度，
同步调整碰撞体和外观的局部中心高度；保留模型与链接的位置、水平尺寸及朝向。

默认输入地图和墙高位于脚本顶部的 `DEFAULT_WORLD_FILE`、`DEFAULT_WALL_HEIGHT_M`。
命令行可覆盖配置。以下命令在 `Gazebo_simulation` 目录执行：

```bash
# 另存一份，便于检查
python tools/adjust_wall_height.py \
  --world assets/worlds/train_world3.sdf --height 0.4 \
  --output /tmp/train_world_height_04.sdf

# 省略 --output 时修改输入文件
python tools/adjust_wall_height.py --height 0.4
```

默认高度为 0.3 m，要求输入大于零的有限数。该工具处理当前编辑器导出的墙体格式，
不是通用 SDF 坐标变换工具；带有额外高度偏移或倾斜的墙体需要单独核对。
从脚本启动到另存都不需要 ROS。修改地图后按 [仿真说明](../docs/simulation.md) 重新加载，
RL 结果须对应实际测试过的地图。
