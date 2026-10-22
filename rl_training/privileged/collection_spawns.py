"""Reproducible collection starts; geometry checks do not guarantee PPO recovery."""

import math
import random

import numpy as np

from ..common.multi_map import safe_spawn


def build_spawn_plan(world, geometry, vehicle, rounds, seed, settings):
    for key in ("position_jitter_m", "yaw_jitter_deg", "margin_m", "recovery_max_offset_m"):
        if not math.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f"出生配置 {key} 必须为非负有限数。")
    fraction = settings["recovery_fraction"]
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("RECOVERY_FRACTION 必须在 [0, 1]。")
    if type(settings["attempts"]) is not int or settings["attempts"] < 1:
        raise ValueError("SPAWN_ATTEMPTS 必须为正整数。")
    for key in ("recovery_clearance_m", "recovery_yaw_deg", "recovery_front_m"):
        limits = settings[key]
        if (len(limits) != 2 or not all(math.isfinite(v) for v in limits)
                or not 0 < limits[0] <= limits[1]):
            raise ValueError(f"出生配置 {key} 必须为两个递增的正有限数。")
    if settings["recovery_yaw_deg"][1] >= 90:
        raise ValueError("近墙偏航必须小于 90 度，保留原来的行驶方向。")
    if fraction and (settings["recovery_clearance_m"][0] <= settings["margin_m"]
                     or settings["recovery_max_offset_m"] <= 0):
        raise ValueError("近墙净空下限必须大于出生安全余量，横移上限必须为正。")

    rng = random.Random(seed)
    # A fixed count avoids having no recovery rounds by chance in a small batch.
    recovery_rounds = set(rng.sample(range(rounds), math.floor(rounds * fraction + 0.5)))
    plan = []
    for index in range(rounds):
        base_index = index % len(world.eval_spawn_poses)
        x, y, z, yaw = world.eval_spawn_poses[base_index]
        recovery = index in recovery_rounds
        for _ in range(settings["attempts"]):
            jitter = settings["position_jitter_m"]
            if recovery:
                side = rng.choice((-1, 1))
                along = rng.uniform(-jitter, jitter)
                across = side * rng.uniform(0, settings["recovery_max_offset_m"])
                heading = yaw + side * math.radians(rng.uniform(*settings["recovery_yaw_deg"]))
                pose = (x + along * math.cos(yaw) - across * math.sin(yaw),
                        y + along * math.sin(yaw) + across * math.cos(yaw), z, heading)
            else:
                pose = (x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter), z,
                        yaw + math.radians(rng.uniform(-settings["yaw_jitter_deg"],
                                                       settings["yaw_jitter_deg"])))
            if not safe_spawn(world, geometry, pose, settings["margin_m"]):
                continue
            clearance, front = geometry.safety_distances(dict(x=pose[0], y=pose[1], yaw=pose[3]))
            if recovery:
                low, high = settings["recovery_clearance_m"]
                front_low, front_high = settings["recovery_front_m"]
                if not (low <= clearance <= high and front_low <= front <= front_high):
                    continue
                # Check a full strongest corrective action, sampled every <=5 ms.
                # This only rejects impossible starts; the real teacher picks all actions.
                omega = -side * 2 * vehicle.forward_speed_mps * vehicle.max_wheel_speed_ratio / vehicle.wheel_separation_m
                if omega == 0:
                    raise ValueError("近墙场景需要能够转向的车辆配置。")
                times = np.linspace(0, settings["step_seconds"],
                                    math.ceil(settings["step_seconds"] / 0.005) + 1)
                angles = pose[3] + omega * times
                radius = vehicle.forward_speed_mps / omega
                turn = np.column_stack((pose[0] + radius * (np.sin(angles) - math.sin(pose[3])),
                                        pose[1] - radius * (np.cos(angles) - math.cos(pose[3]))))
                if not all(safe_spawn(world, geometry, (px, py, z, 0), settings["margin_m"])
                           for px, py in turn):
                    continue
            kind = "near_wall" if recovery else ("jitter" if jitter or settings["yaw_jitter_deg"] else "fixed")
            plan.append({"kind": kind,
                         "base_spawn_index": base_index, "pose": list(pose),
                         "clearance_m": clearance, "front_clearance_m": front})
            break
        else:
            raise ValueError(f"{world.world.name} 第 {index + 1} 轮无法生成"
                             f"{'近墙' if recovery else '扰动'}出生点；请调整扰动范围/净空或地图出生点。")
    return plan
