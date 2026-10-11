# 资源编辑工具

## 批量生成训练地图

[generate_maps.py](generate_maps.py) 支持圆角矩形、椭圆、直角矩形、开放交叉口的 8 字形及 T 字死胡同地图。
不需要 ROS；使用普通 `python`，预览图需要已安装的 Matplotlib。
以下命令在 `Gazebo_simulation` 目录执行：

```bash
# 默认生成 6 张，交替使用两种模板。
python tools/generate_maps.py

# 换种子生成另一批；不同批次使用不同输出目录。
python tools/generate_maps.py --count 20 --seed 123 --output-dir assets/worlds/simple/batch_123

# 固定尺寸，批量改变圆角半径。
python tools/generate_maps.py --templates rounded_rectangle --count 10 \
  --width 2.52 --height 3.64 --road-width 0.38 --output-dir assets/worlds/simple/rounded_batch
```

默认参数全部位于脚本顶部的大写常量区：数量、模板、种子、输出目录、长宽范围、
道路净宽、圆角半径、墙厚/墙高和曲线分段长度。长宽是道路中心线的跨度，单位为米。
命令行覆盖默认值；省略尺寸参数时从配置范围采样。同样的参数和种子可复现同一批地图。

默认中心线跨度为水平 2.4～3.0 m、垂直 3.4～4.2 m，道路净宽 0.35～0.40 m，
接近现有手绘地图（实测直道约 0.34～0.40 m）。圆角半径 0.45～0.65 m，
曲线分段长度 0.12 m；墙厚沿用原图的 0.15 m。
生成器自动调用 [adjust_wall_height.py](adjust_wall_height.py) 处理每张地图。
默认墙高读取该脚本顶部的 `DEFAULT_WALL_HEIGHT_M`（当前为 0.3 m）；
如需只改变生成地图的墙高，可修改生成器顶部的 `WALL_HEIGHT_M`。
墙体 link 的 z 为 0，高度偏移放在 collision/visual 内，确保后续调整墙高时底部仍贴地。
地图内还保存了朝向道路中心的默认相机，打开 Gazebo 即可查看全景。

每张地图输出以下同名文件（资源与结果分开保存），例如 `rounded_rectangle_001`：

- `.sdf`：实际 Gazebo 场景，墙体均为静态 box，兼容特权教师。
- `.json`：4 个训练出生点、8 个评估出生点、朝向（弧度）、工作边界及生成参数。
- `.png` / `.pdf`：俯视预览；橙色编号箭头表示训练出生点和朝向。

同一输出目录再次生成会覆盖同名文件，因此需要保留不同批次时请换输出目录。
生成地图及 JSON 默认位于 `assets/worlds/simple/`，不覆盖手绘地图。
PNG/PDF 默认位于 `results/maps/<地图输出目录名>/`，例如 `results/maps/simple/`。
顶部 `DEFAULT_PREVIEW_DIR` 配置预览根目录；`--preview-dir` 可指定本批预览目录。
若地图输出目录的末级名称相同，请用 `--preview-dir` 区分结果。
训练和评估直接指定地图即可自动读取同名 JSON：

```bash
python rl_training/privileged/train.py --world assets/worlds/simple/rounded_rectangle_001.sdf
python rl_training/lidar/train.py --world assets/worlds/simple/ellipse_002.sdf
```

仿真环境已加载时，可以直接预览新地图；手动启动默认使用该地图的第一个训练出生点：

```bash
ros2 launch my_gazebo_launch train_world.launch.py \
  world:=assets/worlds/simple/ellipse_002.sdf
```

不同地图目前按一次运行一张选择，并行车在该地图的不同起点训练。
这些示例地图用于搭建训练环境；旧模型在新地图上的巡航表现需要重新评估。

## 修改现有墙高

[adjust_wall_height.py](adjust_wall_height.py) 修改地图中 `Wall*` 链接的 box 墙体高度，
同步调整碰撞体和外观的局部中心高度；保留模型与链接的位置、水平尺寸及朝向。

默认输入地图和墙高位于脚本顶部的 `DEFAULT_WORLD_FILE`、`DEFAULT_WALL_HEIGHT_M`。
命令行可覆盖配置。以下命令在 `Gazebo_simulation` 目录执行：

```bash
# 另存一份，便于检查
python tools/adjust_wall_height.py \
  --world assets/worlds/simple/train_world3.sdf --height 0.4 \
  --output /tmp/train_world_height_04.sdf

# 省略 --output 时修改输入文件
python tools/adjust_wall_height.py --height 0.4
```

默认高度为 0.3 m，要求输入大于零的有限数。该工具处理当前编辑器导出的墙体格式，
不是通用 SDF 坐标变换工具；带有额外高度偏移或倾斜的墙体需要单独核对。
从脚本启动到另存都不需要 ROS。修改地图后按 [仿真说明](../docs/simulation.md) 重新加载，
RL 结果须对应实际测试过的地图。

## 直角矩形与 8 字形

```bash
python tools/generate_maps.py --templates rectangle figure_eight --count 2
```

输出 `assets/worlds/simple/rectangle_001.sdf` 和 `figure_eight_002.sdf`，
同目录配套 JSON；PNG/PDF 位于 `results/maps/simple/`。
直角矩形保留真正的 90 度拐角；8 字形中心为平面开放交叉口，无桥梁。
净宽仍为 0.35～0.40 m（交叉口更宽），墙高 0.3 m，自动调用降墙高工具。
窄直角弯需要重新验证车辆转弯能力；当前奖励不约束 8 字路线，车辆可能只绕单圈。
`assets/worlds/complex/` 预留给后续复杂地图，目前为空。

## 死胡同地图

```bash
python tools/generate_maps.py --templates dead_end --count 1 --width 2.8 --height 3.6 --road-width 0.38
ros2 launch my_gazebo_launch train_world.launch.py world:=assets/worlds/simple/dead_end_001.sdf
```

`dead_end_001.sdf/.json` 与其他简单地图一起保存在 `assets/worlds/simple/`，
PNG/PDF 位于 `results/maps/simple/`。T 字左右及下方三个末端均封闭，墙高 0.3 m。
三个训练出生点位于不同深度并面向封闭端，第四个位于横向通道；评估还包含反向朝向。
当前车体几何的旋转包络直径约 0.158 m，居中时可放入 0.38 m 净宽通道。
这是几何空间检查，尚未验证原地旋转控制或 RL 脱困策略。
后续需添加原地旋转动作、调整低速/卡住判定及脱困奖励；当前训练仍按原有巡航规则运行。
未来脱困成功应按车辆离开某条死胡同支路、返回 T 字交汇处判定，而不是驶出地图边界。
