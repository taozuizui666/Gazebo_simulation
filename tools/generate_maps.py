#!/usr/bin/env python3
"""批量生成静态 box 训练地图、出生点配置和俯视预览；无需 ROS。"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from adjust_wall_height import DEFAULT_WALL_HEIGHT_M, adjust_wall_height

# ===================== 可修改配置区 =====================
SIMULATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = SIMULATION_ROOT / "assets/worlds/simple"
DEFAULT_PREVIEW_DIR = SIMULATION_ROOT / "results/maps"  # 按地图输出目录名分批保存 PNG/PDF。
BASE_WORLD_FILE = SIMULATION_ROOT / "assets/worlds/simple/train_world3.sdf"  # 复用光照、地面、物理配置。
DEFAULT_MAP_COUNT = 6
DEFAULT_TEMPLATES = ("rounded_rectangle", "ellipse")
DEFAULT_SEED = 42  # 相同参数和种子生成相同地图。
WIDTH_RANGE_M = (2.4, 3.0)  # 道路中心线的水平跨度；接近现有手绘地图。
HEIGHT_RANGE_M = (3.4, 4.2)  # 道路中心线的垂直跨度。
ROAD_WIDTH_RANGE_M = (0.35, 0.40)  # 两侧墙之间的道路净宽；原地图约 0.34～0.40 m。
CORNER_RADIUS_RANGE_M = (0.45, 0.65)  # 圆角矩形中心线的弯道半径。
WALL_HEIGHT_M = DEFAULT_WALL_HEIGHT_M  # 默认跟随降墙高脚本（当前 0.3 m）。
WALL_THICKNESS_M = 0.15
CURVE_SEGMENT_LENGTH_M = 0.12  # 较短曲线段让窄道路的弯道接缝更平滑。
WORKSPACE_PADDING_M = 0.1
# ===================== 配置区结束 =====================


def centerline(template, width, height, radius):
    """返回逆时针道路中心线上的点及其外法向量。"""
    a, b = width / 2, height / 2
    if template == "rectangle":
        return [((a, b), (1, 1)), ((-a, b), (-1, 1)),
                ((-a, -b), (-1, -1)), ((a, -b), (1, -1))]
    if template == "figure_eight":
        count = max(120, math.ceil(4 * math.pi * max(a, b) / CURVE_SEGMENT_LENGTH_M))
        result = []
        for i in range(count):
            t = 2 * math.pi * i / count
            dx, dy = 2 * a * math.cos(2*t), b * math.cos(t)
            norm = math.hypot(dx, dy)
            result.append(((a * math.sin(2*t), b * math.sin(t)), (dy/norm, -dx/norm)))
        return result
    if template == "ellipse":
        count = max(24, math.ceil(2 * math.pi * max(a, b) / CURVE_SEGMENT_LENGTH_M))
        result = []
        for i in range(count):
            angle = 2 * math.pi * i / count
            c, s = math.cos(angle), math.sin(angle)
            norm = math.hypot(c / a, s / b)
            result.append(((a * c, b * s), (c / a / norm, s / b / norm)))
        return result
    count = max(3, math.ceil(math.pi * radius / 2 / CURVE_SEGMENT_LENGTH_M))
    result = []
    for corner, (x, y) in enumerate(((a-radius, b-radius), (-a+radius, b-radius),
                                    (-a+radius, -b+radius), (a-radius, -b+radius))):
        for i in range(count + 1):
            angle = (corner + i / count) * math.pi / 2
            c, s = math.cos(angle), math.sin(angle)
            result.append(((x + radius * c, y + radius * s), (c, s)))
    return result


def wall_boxes(points, road_width):
    boxes = []
    offset = (road_width + WALL_THICKNESS_M) / 2
    for side in (-1, 1):
        boundary = [(x + side * offset * nx, y + side * offset * ny)
                    for (x, y), (nx, ny) in points]
        segments = list(zip(boundary, boundary[1:] + boundary[:1]))
        angles = [math.atan2(y2-y1, x2-x1) for (x1,y1),(x2,y2) in segments]
        for index, ((x1, y1), (x2, y2)) in enumerate(segments):
            # 只重叠封住接缝所需的长度，避免长墙端侵入窄弯道。
            yaw = angles[index]
            turn = max(abs(math.atan2(math.sin(yaw-neighbor), math.cos(yaw-neighbor)))
                       for neighbor in (angles[index-1], angles[(index+1) % len(angles)]))
            overlap = WALL_THICKNESS_M * math.tan(turn / 2) + .001
            boxes.append(((x1+x2)/2, (y1+y2)/2,
                          math.hypot(x2-x1, y2-y1) + overlap, yaw))
    return boxes


def open_crossing(boxes, points, road_width):
    """去除 8 字交叉口内的墙段；按中心线距离裁剪，不封堵另一路。"""
    import numpy as np

    line = np.asarray([point for point, _ in points])
    vectors = np.roll(line, -1, axis=0) - line
    squared = (vectors * vectors).sum(axis=1)
    offset = (road_width + WALL_THICKNESS_M) / 2
    result = []
    for index, (x, y, length, yaw) in enumerate(boxes):
        # 5 mm 栅格只用于裁剪交叉口端部；连续保留部分仍合并成一个 box。
        count = max(1, math.ceil(length / .005))
        step = length / count
        positions = -length/2 + (np.arange(count) + .5) * step
        samples = np.column_stack((x + positions*math.cos(yaw), y + positions*math.sin(yaw)))
        delta = samples[:, None, :] - line
        fractions = np.clip((delta * vectors).sum(axis=2) / squared, 0, 1)
        distances = np.linalg.norm(delta - fractions[:, :, None]*vectors, axis=2)
        # 本支路相邻线段不参与裁剪，避免曲线离散误差切断弯道墙体。
        neighbors = (np.arange(len(line)) - index % len(line) + len(line)//2) % len(line) - len(line)//2
        distances[:, np.abs(neighbors) <= len(line)//8] = np.inf
        distance = distances.min(axis=1)
        keep = distance >= offset - .001
        edges = np.flatnonzero(np.diff(np.r_[False, keep, False]))
        for start, end in zip(edges[::2], edges[1::2]):
            middle = -length/2 + (start+end)*step/2
            result.append((x + middle*math.cos(yaw), y + middle*math.sin(yaw),
                           (end-start)*step, yaw))
    # 找出裁剪后真正外露的墙端，包含恰好落在原分段端点的切口。
    centers = np.asarray([box[:2] for box in result])
    lengths = np.asarray([box[2] for box in result])
    directions = np.asarray([(math.cos(box[3]), math.sin(box[3])) for box in result])
    cut_ends = []
    for index, (center, length, direction) in enumerate(zip(centers, lengths, directions)):
        for side in (-1, 1):
            endpoint = center + side*length/2*direction
            delta = endpoint + side*.0001*direction - centers
            along = np.abs((delta*directions).sum(axis=1))
            across = np.abs(delta[:, 0]*directions[:, 1] - delta[:, 1]*directions[:, 0])
            covered = ((along < lengths/2) & (across < WALL_THICKNESS_M/2)
                       & (np.abs(directions @ direction) > .95))
            covered[index] = False
            if not covered.any():
                cut_ends.append((endpoint, -side*direction))
    # 每对裁剪端点属于同一墙角。补一个沿角平分线横向的短 box，
    # 封住平切墙端之间的 V 口；形成平头接缝，不向行驶通道延伸尖角。
    while cut_ends:
        first, d1 = cut_ends.pop()
        nearest = min(range(len(cut_ends)), key=lambda i: np.linalg.norm(cut_ends[i][0]-first))
        second, d2 = cut_ends.pop(nearest)
        if float(np.dot(d1, d2)) < -.95:
            continue  # 原有相邻共线墙段的重叠端点，不是交叉口墙角。
        along = np.linalg.solve(np.column_stack((d1, -d2)), second-first)[0]
        joint = first + along*d1
        inward = (d1+d2) / np.linalg.norm(d1+d2)
        cosine = float(np.dot(d1, inward))
        sine = math.sqrt(max(0., 1-cosine*cosine))
        center = joint + inward * (WALL_THICKNESS_M/2 * (1-sine))
        result.append((*center, WALL_THICKNESS_M*cosine,
                       math.atan2(inward[1], inward[0]) + math.pi/2))
    return result


def dead_end_boxes(width, height, road_width):
    """T 形通道：左右及下方三个末端均封闭。"""
    a, b = width/2, height/2
    offset = (road_width + WALL_THICKNESS_M)/2
    bottom = -b - WALL_THICKNESS_M/2
    segments = [((-a, b+offset), (a, b+offset)),
                ((-a, b-offset), (-offset, b-offset)),
                ((offset, b-offset), (a, b-offset)),
                ((-offset, bottom), (-offset, b-offset)),
                ((offset, bottom), (offset, b-offset)),
                ((-offset, bottom), (offset, bottom)),
                ((-a, b-offset), (-a, b+offset)),
                ((a, b-offset), (a, b+offset))]
    return [((x1+x2)/2, (y1+y2)/2,
             math.hypot(x2-x1, y2-y1)+WALL_THICKNESS_M, math.atan2(y2-y1, x2-x1))
            for (x1, y1), (x2, y2) in segments]


def wall_polygons(boxes):
    polygons = []
    for x, y, length, yaw in boxes:
        c, s = math.cos(yaw), math.sin(yaw)
        polygons.append([(x + c * dx - s * dy, y + s * dx + c * dy)
                         for dx, dy in ((-length/2, -WALL_THICKNESS_M/2),
                                        (length/2, -WALL_THICKNESS_M/2),
                                        (length/2, WALL_THICKNESS_M/2),
                                        (-length/2, WALL_THICKNESS_M/2))])
    return polygons


def make_world(name, boxes):
    root = ET.Element("sdf", version="1.7")
    world = ET.SubElement(root, "world", name=name)
    base = ET.parse(BASE_WORLD_FILE).getroot().find("world")
    for child in base:
        if child.tag in ("include", "physics"):
            world.append(deepcopy(child))
    # 默认相机朝向道路中心，打开 Gazebo 即可看到整张地图。
    extent = max(abs(v) for polygon in wall_polygons(boxes) for point in polygon for v in point)
    camera = ET.SubElement(ET.SubElement(world, "gui"), "camera", name="user_camera")
    ET.SubElement(camera, "pose").text = (
        f"{2.4*extent} {-3.2*extent} {3.4*extent} 0 {math.atan(.85)} {math.atan2(1.6, -1.2)}")
    ET.SubElement(camera, "view_controller").text = "orbit"
    model = ET.SubElement(world, "model", name="track_walls")
    ET.SubElement(model, "static").text = "true"
    for index, (x, y, length, yaw) in enumerate(boxes):
        link = ET.SubElement(model, "link", name=f"Wall_{index}")
        ET.SubElement(link, "pose").text = f"{x:.9f} {y:.9f} 0 0 0 {yaw:.9f}"
        for tag in ("collision", "visual"):
            element = ET.SubElement(link, tag, name=f"wall_{index}_{tag}")
            # 与手绘地图一致：link 在地面，高度偏移位于 collision/visual 中。
            ET.SubElement(element, "pose").text = f"0 0 {WALL_HEIGHT_M/2} 0 0 0"
            box = ET.SubElement(ET.SubElement(element, "geometry"), "box")
            ET.SubElement(box, "size").text = f"{length:.9f} {WALL_THICKNESS_M} {WALL_HEIGHT_M}"
            if tag == "visual":
                material = ET.SubElement(element, "material")
                for color in ("ambient", "diffuse"):
                    ET.SubElement(material, color).text = "0.4 0.46 0.52 1"
    ET.indent(root)
    return '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def map_settings(template, width, height, road_width, radius, seed, boxes):
    a, b = width / 2, height / 2
    forward = [(a, 0., 0., math.pi/2), (0., b, 0., math.pi),
               (-a, 0., 0., -math.pi/2), (0., -b, 0., 0.)]
    if template == "figure_eight":
        forward = []
        for t in (math.pi/4, 3*math.pi/4, 5*math.pi/4, 7*math.pi/4):
            forward.append((a*math.sin(2*t), b*math.sin(t), 0.,
                            math.atan2(b*math.cos(t), 2*a*math.cos(2*t))))
    if template == "dead_end":
        forward = [(0., -b+road_width, 0., -math.pi/2),
                   (0., -b+height*.35, 0., -math.pi/2),
                   (0., -b+height*.65, 0., -math.pi/2),
                   (-a/2, b, 0., 0.)]
    reverse = [(x, y, z, math.atan2(math.sin(yaw+math.pi), math.cos(yaw+math.pi)))
               for x, y, z, yaw in forward]
    vertices = [point for polygon in wall_polygons(boxes) for point in polygon]
    xs, ys = zip(*vertices)
    return {
        "template": template, "seed": seed,
        "parameters": {"width_m": width, "height_m": height, "road_width_m": road_width,
                       "corner_radius_m": radius if template == "rounded_rectangle" else None,
                       "wall_height_m": WALL_HEIGHT_M, "wall_thickness_m": WALL_THICKNESS_M},
        "train_spawn_poses": forward if template == "dead_end" else [forward[0], reverse[1], forward[2], reverse[3]],
        "eval_spawn_poses": forward + reverse,
        "workspace_bounds": [min(xs)-WORKSPACE_PADDING_M, max(xs)+WORKSPACE_PADDING_M,
                             min(ys)-WORKSPACE_PADDING_M, max(ys)+WORKSPACE_PADDING_M],
    }


def draw_preview(axis, boxes, settings):
    from matplotlib.collections import PolyCollection
    axis.add_collection(PolyCollection(wall_polygons(boxes), facecolor="#687787"))
    for index, (x, y, _, yaw) in enumerate(settings["train_spawn_poses"]):
        axis.arrow(x, y, .3*math.cos(yaw), .3*math.sin(yaw), width=.02,
                   head_width=.13, color="#e66b36", length_includes_head=True)
        axis.annotate(str(index+1), (x, y), xytext=(5, 5), textcoords="offset points")
    xmin, xmax, ymin, ymax = settings["workspace_bounds"]
    axis.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x (m)", ylabel="y (m)", aspect="equal")
    axis.grid(alpha=.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_MAP_COUNT)
    parser.add_argument("--templates", nargs="+", choices=("rounded_rectangle", "ellipse", "rectangle", "figure_eight", "dead_end"),
                        default=DEFAULT_TEMPLATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preview-dir", type=Path, help="预览输出目录；默认 results/maps/<地图输出目录名>")
    for name in ("width", "height", "road-width", "corner-radius"):
        parser.add_argument(f"--{name}", type=float, help="固定此参数（米）；省略时从顶部范围随机取值")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count 必须大于零。")
    constants = (WALL_HEIGHT_M, WALL_THICKNESS_M, CURVE_SEGMENT_LENGTH_M, WORKSPACE_PADDING_M)
    ranges = (WIDTH_RANGE_M, HEIGHT_RANGE_M, ROAD_WIDTH_RANGE_M, CORNER_RADIUS_RANGE_M)
    if (any(not math.isfinite(v) or v <= 0 for v in constants)
            or any(len(r) != 2 or not all(math.isfinite(v) and v > 0 for v in r)
                   or r[0] > r[1] for r in ranges)):
        parser.error("顶部尺寸与采样范围必须是有效的正数。")
    randomizer = random.Random(args.seed)
    maps = []
    for index in range(args.count):
        template = args.templates[index % len(args.templates)]
        width, height, road_width, radius = [
            value if value is not None else randomizer.uniform(*limits)
            for value, limits in zip((args.width, args.height, args.road_width, args.corner_radius), ranges)]
        if not all(math.isfinite(v) and v > 0 for v in (width, height, road_width, radius)):
            parser.error("地图尺寸必须是有限正数。")
        offset = (road_width + WALL_THICKNESS_M) / 2
        if template == "rounded_rectangle":
            if not offset < radius < min(width, height) / 2:
                parser.error("圆角半径必须大于半道路宽加半墙厚，且小于半地图短边。")
        elif template == "ellipse" and offset >= min(width, height)**2 / (2 * max(width, height)):
            parser.error("道路过宽，椭圆内侧边界会自交；请增大短边或减小道路宽度。")
        elif template == "rectangle" and min(width, height) <= 2*offset:
            parser.error("矩形尺寸过小，内墙会相交。")
        elif template == "figure_eight":
            # 验证采样曲率半径，防止内侧偏移曲线在窄弯处折返。
            for i in range(720):
                t = 2*math.pi*i/720
                dx, dy = width*math.cos(2*t), height/2*math.cos(t)
                ddx, ddy = -2*width*math.sin(2*t), -height/2*math.sin(t)
                curvature = abs(dx*ddy-dy*ddx) / (dx*dx+dy*dy)**1.5
                if offset*curvature >= 1:
                    parser.error("8 字弯道过窄；请增大地图尺寸或减小道路宽度。")
        if template == "dead_end":
            if width <= 4*offset or height <= 3*road_width:
                parser.error("死胡同地图过小；宽度须大于 2×(道路净宽+墙厚)，高度须大于三倍道路净宽。")
            boxes = dead_end_boxes(width, height, road_width)
        else:
            points = centerline(template, width, height, radius)
            boxes = wall_boxes(points, road_width)
            if template == "figure_eight":
                boxes = open_crossing(boxes, points, road_width)
        name = f"{template}_{index+1:03d}"
        settings = map_settings(template, width, height, road_width, radius, args.seed, boxes)
        maps.append((name, boxes, settings))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.preview_dir or DEFAULT_PREVIEW_DIR / args.output_dir.resolve().name
    preview_dir.mkdir(parents=True, exist_ok=True)
    for name, boxes, settings in maps:
        path = args.output_dir / name
        path.with_suffix(".sdf").write_text(make_world(name, boxes), encoding="utf-8")
        adjust_wall_height(path.with_suffix(".sdf"), WALL_HEIGHT_M)
        path.with_suffix(".json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        draw_preview(axis, boxes, settings)
        axis.set_title(f"{name} | road width {settings['parameters']['road_width_m']:.2f} m")
        fig.savefig(preview_dir / f"{name}.png", dpi=140)
        fig.savefig(preview_dir / f"{name}.pdf")
        plt.close(fig)
        print(path.with_suffix(".sdf"))
    print(f"地图及配置：{args.output_dir}；PNG/PDF 俯视图：{preview_dir}")


if __name__ == "__main__":
    main()
