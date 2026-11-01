"""Deterministic normal/yaw/near-wall start plans for student recovery tests."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

from my_gazebo_launch.config import load_config
from my_gazebo_launch.world_config import load_world_settings
from rl_training.privileged.observation import GroundTruthMap

YAW_OFFSETS_DEG = (15.0, 25.0, -15.0, -25.0, 20.0)
NEAR_WALL_CLEARANCE_M = (0.04, 0.08)
NEAR_WALL_FRONT_M = (0.15, 0.40)
NEAR_WALL_OFFSET_M = (0.02, 0.07)
MARGIN_M = 0.01
STEP_SECONDS = 0.2
ATTEMPTS = 400


def safe_spawn(world, geometry, pose, margin=MARGIN_M):
    radius = geometry.settings.body_clearance_radius_m + margin
    x, y = pose[:2]
    if world.workspace_bounds:
        xmin, xmax, ymin, ymax = world.workspace_bounds
        if not (xmin + radius < x < xmax - radius and ymin + radius < y < ymax - radius):
            return False
    return float(geometry.signed_distance(np.array([[x, y]]))[0]) > radius


def swept_safe(world, geometry, pose, vehicle, side, margin=MARGIN_M):
    """Check the strongest corrective arc for one control step."""
    omega = -side * 2 * vehicle.forward_speed_mps * vehicle.max_wheel_speed_ratio / vehicle.wheel_separation_m
    if omega == 0:
        return True
    times = np.linspace(0, STEP_SECONDS, math.ceil(STEP_SECONDS / 0.005) + 1)
    angles = pose[3] + omega * times
    radius = vehicle.forward_speed_mps / omega
    x = pose[0] + radius * (np.sin(angles) - math.sin(pose[3]))
    y = pose[1] - radius * (np.cos(angles) - math.cos(pose[3]))
    return all(safe_spawn(world, geometry, (px, py, pose[2], 0.0), margin) for px, py in zip(x, y))


def _clearance(geometry, pose):
    clearance, front = geometry.safety_distances(dict(x=pose[0], y=pose[1], yaw=pose[3]))
    return float(clearance), float(front)


def build_start_plan(world, geometry, vehicle, seed=20261029, per_category=5):
    if per_category < 1:
        raise ValueError("per_category 必须为正整数。")
    bases = list(world.eval_spawn_poses)
    if len(bases) < per_category:
        raise ValueError(f"{world.world.name} 的评估出生点不足 {per_category} 个。")
    rng = random.Random(seed)
    categories = {"normal": [], "yaw": [], "near_wall": []}

    for index in range(per_category):
        base = bases[index]
        for _ in range(ATTEMPTS):
            pose = (base[0] + rng.uniform(-0.02, 0.02), base[1] + rng.uniform(-0.02, 0.02),
                    base[2], base[3] + math.radians(rng.uniform(-3.0, 3.0)))
            if not safe_spawn(world, geometry, pose):
                continue
            clearance, front = _clearance(geometry, pose)
            categories["normal"].append({"base_index": index, "pose": list(pose),
                "clearance_m": clearance, "front_clearance_m": front})
            break
        else:
            raise ValueError(f"{world.world.name} 第 {index + 1} 个 normal 起点生成失败。")

    for index in range(per_category):
        base = bases[index]
        for _ in range(ATTEMPTS):
            pose = (base[0], base[1], base[2], base[3] + math.radians(YAW_OFFSETS_DEG[index % len(YAW_OFFSETS_DEG)]))
            if not safe_spawn(world, geometry, pose):
                continue
            clearance, front = _clearance(geometry, pose)
            categories["yaw"].append({"base_index": index, "pose": list(pose),
                "clearance_m": clearance, "front_clearance_m": front})
            break
        else:
            raise ValueError(f"{world.world.name} 第 {index + 1} 个 yaw 起点生成失败。")

    for index in range(per_category):
        base = bases[index]
        for _ in range(ATTEMPTS):
            side = rng.choice((-1, 1))
            across = rng.uniform(*NEAR_WALL_OFFSET_M)
            heading = base[3] + side * math.radians(rng.uniform(10.0, 20.0))
            nx, ny = -math.sin(base[3]), math.cos(base[3])
            pose = (base[0] + side * across * nx, base[1] + side * across * ny, base[2], heading)
            if not safe_spawn(world, geometry, pose) or not swept_safe(world, geometry, pose, vehicle, side):
                continue
            clearance, front = _clearance(geometry, pose)
            if not (NEAR_WALL_CLEARANCE_M[0] <= clearance <= NEAR_WALL_CLEARANCE_M[1]
                    and NEAR_WALL_FRONT_M[0] <= front <= NEAR_WALL_FRONT_M[1]):
                continue
            categories["near_wall"].append({"base_index": index, "pose": list(pose),
                "clearance_m": clearance, "front_clearance_m": front, "side": side})
            break
        else:
            raise ValueError(f"{world.world.name} 第 {index + 1} 个 near_wall 起点生成失败。")

    return {"world": str(world.world), "world_sha256": geometry.sha256, "seed": seed,
            "per_category": per_category, "margin_m": MARGIN_M,
            "body_clearance_radius_m": geometry.settings.body_clearance_radius_m,
            "categories": categories}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True, help="SDF 路径；相对 Gazebo_simulation")
    parser.add_argument("--output", required=True, help="start_plan.json 输出路径")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--config-header", default="my_gazebo_launch/vehicle_config.h")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    world_path = Path(args.world)
    if not world_path.is_absolute():
        world_path = root / world_path
    world = load_world_settings(world_path)
    geometry = GroundTruthMap(world.world)
    vehicle = load_config(Path(args.config_header) if Path(args.config_header).is_absolute() else root / args.config_header)
    plan = build_start_plan(world, geometry, vehicle, args.seed, args.per_category)
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"写入 {output}")
    for name, entries in plan["categories"].items():
        print(f"{name}: {len(entries)} 个起点；净空 {min(e['clearance_m'] for e in entries):.3f}–"
              f"{max(e['clearance_m'] for e in entries):.3f} m")


if __name__ == "__main__":
    main()
