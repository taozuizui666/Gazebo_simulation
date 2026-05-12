#!/usr/bin/env python3
"""
快速调整墙体高度工具
"""

import re

def adjust_wall_height(world_file_path, new_height_meters=0.5):
    """
    调整所有墙体的高度，并确保墙体底部紧贴地面
    
    Args:
        world_file_path: 世界文件路径
        new_height_meters: 新的墙高（默认 0.5 米）
    """
    with open(world_file_path, 'r') as f:
        lines = f.readlines()
    
    # 计算新的 center Z 坐标（高度的一半，确保底部在地面 z=0）
    new_center_z = new_height_meters / 2.0
    
    modified_lines = []
    in_wall_link = False
    in_collision_or_visual = False
    
    for i, line in enumerate(lines):
        original_line = line
        
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
    with open(world_file_path, 'w') as f:
        f.writelines(modified_lines)
    
    print(f"✅ 已将墙体高度调整为 {new_height_meters} 米")
    print(f"📍 墙体中心 Z 坐标设置为 {new_center_z} 米（紧贴地面）")
    print(f"📁 文件已保存：{world_file_path}")


if __name__ == '__main__':
    import sys
    
    world_file = '/home/zui/ros2_ws/src/my_gazebo_launch/sdf/train_world3.sdf'
    
    # 默认调整为 1 米高
    target_height = 0.3
    
    if len(sys.argv) > 1:
        try:
            target_height = float(sys.argv[1])
        except ValueError:
            print(f"❌ 无效的高度值：{sys.argv[1]}")
            sys.exit(1)
    
    print(f"🔧 准备调整墙体高度到 {target_height} 米...")
    adjust_wall_height(world_file, target_height)
    print("\n💡 提示：可以在 Gazebo 中查看效果，如不满意可 Ctrl+Z 撤销或重新从 Building Editor 导出")
