#!/usr/bin/env python3
"""
快速调整墙体高度工具
"""

import argparse
import math
from pathlib import Path
import re

# ===================== 可修改配置区 =====================
DEFAULT_WORLD_FILE = Path(__file__).resolve().parents[1] / "assets/worlds/train_world3.sdf"
DEFAULT_WALL_HEIGHT_M = 0.3
# ===================== 配置区结束 =====================

def adjust_wall_height(world_file_path, new_height_meters=DEFAULT_WALL_HEIGHT_M, output_path=None):
    """
    调整所有墙体的高度，并确保墙体底部紧贴地面

    Args:
        world_file_path: 世界文件路径
        new_height_meters: 新的墙高，单位为米
        output_path: 另存路径；省略时修改输入文件
    """
    if not math.isfinite(new_height_meters) or new_height_meters <= 0:
        raise ValueError("墙高必须是大于零的有限数。")
    with open(world_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 计算新的 center Z 坐标（高度的一半，确保底部在地面 z=0）
    new_center_z = new_height_meters / 2.0

    modified_lines = []
    in_wall_link = False
    in_collision_or_visual = False

    for line in lines:
        # 检测是否在 wall 的 link 中
        if '<link name=' in line and 'Wall' in line:
            in_wall_link = True
            in_collision_or_visual = False

        if in_wall_link:
            # 检测进入 collision 或 visual 块
            if '<collision' in line or '<visual' in line:
                in_collision_or_visual = True

            # 检测离开 collision 或 visual 块
            if '</collision>' in line or '</visual>' in line:
                in_collision_or_visual = False

            # 1. 更新 box size 中的高度（第三个值）
            if '<size>' in line and '</size>' in line:
                match = re.search(r'<size>([\d\.]+) ([\d\.]+) ([\d\.]+)</size>', line)
                if match:
                    x, y, z = match.groups()
                    line = line.replace(match.group(0), f'<size>{x} {y} {new_height_meters}</size>')

            # 2. 只更新 collision 和 visual 内部的 pose（相对偏移的 z 坐标）
            #    不更新 link 级别的 pose
            if in_collision_or_visual and '<pose>' in line and '</pose>' in line:
                match = re.search(r'<pose>([\d\.\- ]+)</pose>', line)
                if match:
                    values = match.group(1).split()
                    if len(values) >= 3:
                        # 对于 collision/visual 的 pose，更新 z 为 height/2
                        values[2] = str(new_center_z)
                        new_pose = ' '.join(values)
                        line = line.replace(match.group(1), new_pose)

        modified_lines.append(line)

        # 检测到/link 结束 wall 部分
        if '</link>' in line:
            in_wall_link = False
            in_collision_or_visual = False

    # 写回文件
    destination = output_path or world_file_path
    with open(destination, 'w', encoding='utf-8') as f:
        f.writelines(modified_lines)

    print(f"✅ 已将墙体高度调整为 {new_height_meters} 米")
    print(f"📍 墙体中心 Z 坐标设置为 {new_center_z} 米（紧贴地面）")
    print(f"📁 文件已保存：{destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', type=Path, default=DEFAULT_WORLD_FILE, help='输入 SDF 地图')
    parser.add_argument('--height', type=float, default=DEFAULT_WALL_HEIGHT_M, help='墙高，单位 m')
    parser.add_argument('--output', type=Path, help='另存 SDF；省略时修改输入文件')
    args = parser.parse_args()
    adjust_wall_height(args.world, args.height, args.output)


if __name__ == '__main__':
    main()
