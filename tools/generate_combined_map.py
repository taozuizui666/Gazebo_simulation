#!/usr/bin/env python3
"""生成含连续弯、S 弯、渐变净宽、直角弯和 8 字交叉口的组合地图。"""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely import union_all

from generate_maps import (SIMULATION_ROOT, WALL_THICKNESS_M, WALL_HEIGHT_M,
                           WORKSPACE_PADDING_M, make_world, wall_polygons, draw_preview)
from adjust_wall_height import adjust_wall_height

# ===================== 可修改配置区 =====================
DEFAULT_OUTPUT_DIR = SIMULATION_ROOT / "assets/worlds/complex"
DEFAULT_PREVIEW_DIR = SIMULATION_ROOT / "results/maps/complex"
DEFAULT_NAME = "combined_001"
CAMERA_HEIGHT_M = 15.0  # 默认俯视全景高度。
SEGMENT_LENGTH_M = 0.06  # 曲线与渐变道路的离散长度。
# 每段为 (名称, 贝塞尔控制点或直线端点, 起点净宽, 终点净宽)，单位 m。
# 相邻段共享端点；最后一段返回原点，中央两次穿越形成可选分支。
SECTIONS = (
    ("left_entry", ((0, 0), (-1.2, 1.2)), .55, .55),
    ("continuous_bend_1", ((-1.2, 1.2), (-1.8, 1.8), (-.9, 3.8), (-2.1, 3.8)), .55, .55),
    ("continuous_bend_2", ((-2.1, 3.8), (-3.4, 3.8), (-3.4, 2.7), (-3., 2.1)), .55, .55),
    ("s_bend_narrowing", ((-3., 2.1), (-2.6, 1.5), (-1.7, 1.), (-2.2, .3)), .55, .35),
    ("s_bend_narrow", ((-2.2, .3), (-2.7, -.4), (-3.4, -.8), (-3.1, -1.6)), .35, .35),
    ("left_return_bend", ((-3.1, -1.6), (-2.8, -2.4), (-3.5, -3.8), (-2.2, -3.8)), .35, .45),
    ("left_widening", ((-2.2, -3.8), (-1., -3.8), (-1.8, -1.8), (-1.2, -1.2)), .45, .55),
    ("crossing_up", ((-1.2, -1.2), (1.2, 1.2)), .55, .55),
    ("right_entry", ((1.2, 1.2), (1.2, 3.5)), .55, .60),
    ("right_angle_top", ((1.2, 3.5), (3.1, 3.5)), .60, .60),
    ("right_narrowing", ((3.1, 3.5), (3.1, 1.5)), .60, .40),
    ("right_straight", ((3.1, 1.5), (3.1, -.8)), .40, .40),
    ("right_widening", ((3.1, -.8), (3.1, -3.5)), .40, .60),
    ("right_angle_bottom", ((3.1, -3.5), (1.2, -3.5)), .60, .60),
    ("right_return", ((1.2, -3.5), (1.2, -1.2)), .60, .55),
    ("crossing_down", ((1.2, -1.2), (0, 0)), .55, .55),
)
SPAWN_SECTIONS = (0, 2, 4, 6, 8, 10, 12, 14)  # 避开交叉口与直角接头。
# ===================== 配置区结束 =====================


def build_geometry():
    points, widths, sections = [], [], []
    for name, controls, start_width, end_width in SECTIONS:
        control = np.asarray(controls, dtype=float)
        count = max(2, math.ceil(np.linalg.norm(np.diff(control, axis=0), axis=1).sum() / SEGMENT_LENGTH_M))
        t = np.linspace(0, 1, count + 1)[:, None]
        if len(control) == 2:
            curve = (1-t)*control[0] + t*control[1]
        else:
            curve = (1-t)**3*control[0] + 3*(1-t)**2*t*control[1] + 3*(1-t)*t*t*control[2] + t**3*control[3]
        # 平滑宽度插值，使渐变路段两端的宽度导数为零。
        blend = (3*t*t-2*t**3).ravel()
        section_widths = start_width + (end_width-start_width)*blend
        sections.append({"name": name, "start_index": len(points), "count": count,
                         "width_start_m": start_width, "width_end_m": end_width})
        points.extend(curve[:-1]); widths.extend(section_widths[:-1])
    points, widths = np.asarray(points), np.asarray(widths)
    vectors = np.roll(points, -1, axis=0)-points
    tangents = vectors/np.linalg.norm(vectors, axis=1)[:, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    bisectors = normals + np.roll(normals, 1, axis=0)
    offsets = bisectors / (bisectors*normals).sum(axis=1)[:, None] * widths[:, None]/2
    pieces = []
    for i in range(len(points)):
        j = (i+1)%len(points)
        n = normals[i]
        pieces.append(Polygon([points[i]+n*widths[i]/2, points[i]-n*widths[i]/2,
                               points[j]-n*widths[j]/2, points[j]+n*widths[j]/2]))
        for side in (-1, 1):
            pieces.append(Polygon([points[i], points[i]+side*normals[i-1]*widths[i]/2,
                                   points[i]+side*offsets[i], points[i]+side*n*widths[i]/2]))
    road = union_all(pieces, grid_size=1e-8)
    if road.geom_type == "Polygon":
        # 去掉浮点拼接形成的微小孔洞（小于 1 mm²），不作为真实障碍。
        road = Polygon(road.exterior, [ring for ring in road.interiors if Polygon(ring).area > 1e-6])
    if road.geom_type != "Polygon" or not road.is_valid or len(road.interiors) != 2:
        raise ValueError("道路必须连通，且恰好包含左右两个环岛；请检查控制点和净宽。")
    # 墙中心位于道路边缘外半个墙厚；尖角使用斜接，交叉口由并集自然封口。
    wall_region = road.buffer(WALL_THICKNESS_M/2, join_style=2).simplify(.0001, preserve_topology=True)
    if wall_region.geom_type != "Polygon" or len(wall_region.interiors) != 2:
        raise ValueError("墙厚使环岛消失或分裂，请扩大环岛。")
    boxes = []
    for ring in (wall_region.exterior, *wall_region.interiors):
        vertices = np.asarray(ring.coords[:-1])
        edges = np.roll(vertices, -1, axis=0)-vertices
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        for i, (edge, yaw) in enumerate(zip(edges, angles)):
            turns = [abs(math.atan2(math.sin(yaw-other), math.cos(yaw-other)))
                     for other in (angles[i-1], angles[(i+1)%len(angles)])]
            overlap = WALL_THICKNESS_M*math.tan(max(turns)/2)+.001
            center = vertices[i]+edge/2
            boxes.append((*center, float(np.linalg.norm(edge)+overlap), float(yaw)))
    poses = []
    for index in SPAWN_SECTIONS:
        section = sections[index]
        i = section['start_index'] + section['count']//2
        poses.append([*map(float, points[i]), 0., float(math.atan2(tangents[i, 1], tangents[i, 0]))])
    vertices = np.asarray(wall_polygons(boxes)).reshape(-1, 2)
    bounds = [vertices[:, 0].min()-WORKSPACE_PADDING_M, vertices[:, 0].max()+WORKSPACE_PADDING_M,
              vertices[:, 1].min()-WORKSPACE_PADDING_M, vertices[:, 1].max()+WORKSPACE_PADDING_M]
    settings = {"template": "combined", "parameters": {"road_width_min_m": float(widths.min()),
                "road_width_max_m": float(widths.max()), "wall_height_m": WALL_HEIGHT_M,
                "wall_thickness_m": WALL_THICKNESS_M},
                "train_spawn_poses": poses,
                "eval_spawn_poses": poses + [[x,y,z,math.atan2(math.sin(yaw+math.pi),math.cos(yaw+math.pi))]
                                             for x,y,z,yaw in poses],
                "workspace_bounds": list(map(float, bounds)), "sections": sections,
                "reference_centerline": points.tolist(), "reference_widths_m": widths.tolist()}
    return boxes, settings, road


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--preview-dir', type=Path, default=DEFAULT_PREVIEW_DIR)
    args = parser.parse_args()
    boxes, settings, _ = build_geometry()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    world = args.output_dir / f'{DEFAULT_NAME}.sdf'
    root = ET.fromstring(make_world(DEFAULT_NAME, boxes))
    root.find("world/gui/camera/pose").text = f"0 0 {CAMERA_HEIGHT_M} 0 {math.pi/2} {math.pi/2}"
    world.write_text('<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode") + "\n")
    adjust_wall_height(world, WALL_HEIGHT_M)
    world.with_suffix('.json').write_text(json.dumps(settings, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(9, 10), constrained_layout=True)
    draw_preview(axis, boxes, settings)
    axis.set_title('Combined track | 0.35–0.60 m clear width')
    for x,y,label in ((-2.1,2.8,'Continuous bends'),(-2.8,.1,'S bends'),
                      (2.1,3.85,'90 degree turns'),(3.75,.3,'Width transition'),(0,.5,'Branch choice')):
        axis.annotate(label, (x,y), fontsize=8, ha='center')
    for suffix in ('png','pdf'):
        fig.savefig(args.preview_dir/f'{DEFAULT_NAME}.{suffix}', dpi=160)
    plt.close(fig)
    print(f'地图：{world}；预览：{args.preview_dir}；墙段：{len(boxes)}')


if __name__ == '__main__':
    main()
