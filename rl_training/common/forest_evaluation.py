"""Repeatable RF closed-loop evaluation. Ground truth is used only for scoring."""
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
from uuid import uuid4

import numpy as np


def validate_ring_world(world):
    metadata = json.loads(world.world.with_suffix(".json").read_text())
    if metadata.get("template") not in ("ellipse", "rounded_rectangle", "rectangle"):
        raise ValueError("检查点评分仅支持简单环路；此地图需先定义专用路线和成功条件。")


def corrupt_scan(scan, scenario, rng, active):
    scan = np.asarray(scan, dtype=np.float32).copy()
    if scenario in ("dropout05", "dropout10"):
        probability = .05 if scenario == "dropout05" else .10
        scan[active[rng.random(len(active)) < probability]] = 0
    elif scenario == "sector":
        # A contiguous 20-bin (36 degree) blind sector inside the front hemisphere.
        ordered = np.r_[active[active > 3 * len(scan) // 4], active[active < len(scan) // 4]]
        start = int(rng.integers(len(ordered) - 20 + 1))
        scan[ordered[start:start + 20]] = 0
    elif scenario != "normal":
        raise ValueError(f"Unknown scenario: {scenario}")
    return scan


class CheckpointProgress:
    """Ordered checkpoints for the project's simple convex ring roads.

    Credits chord distances only after reaching the next checkpoint. This is a
    conservative progress proxy, not measured road arc length or odometry.
    """
    def __init__(self, poses, position, yaw, radius):
        points = np.unique(np.asarray(poses)[:, :2], axis=0)
        if len(points) < 3:
            raise ValueError("评分需要至少三个环路检查点（train_spawn_poses）。")
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        self.points = points[np.argsort(angles)]
        nearest = int(np.argmin(np.linalg.norm(self.points - position, axis=1)))
        tangent = self.points[(nearest + 1) % len(points)] - self.points[(nearest - 1) % len(points)]
        self.direction = 1 if np.dot(tangent, [math.cos(yaw), math.sin(yaw)]) >= 0 else -1
        self.last = nearest
        self.next = (nearest + self.direction) % len(points)
        self.started = np.linalg.norm(self.points[nearest] - position) <= radius
        self.radius, self.distance, self.crossings = radius, 0., 0

    def update(self, position):
        if np.linalg.norm(self.points[self.next] - position) <= self.radius:
            if self.started:
                self.distance += float(np.linalg.norm(self.points[self.next] - self.points[self.last]))
            self.started = True
            self.last = self.next
            self.next = (self.next + self.direction) % len(self.points)
            self.crossings += 1
        return self.distance


def summarize(episodes, planned, minimum_score=90., minimum_success=.95):
    valid = [e for e in episodes if e["status"] == "complete"]
    groups = {}
    for e in valid:
        groups.setdefault((e["world"], e["scenario"]), []).append(e)
    cells = []
    for (world, scenario), entries in groups.items():
        mean = lambda key: float(np.mean([e[key] for e in entries]))
        components = {"collision_free": 40 * mean("collision_free"),
                      "success": 30 * mean("success"),
                      "progress": 20 * mean("progress_ratio"),
                      "clearance": 10 * mean("safe_time_ratio")}
        cells.append({"world": world, "scenario": scenario, "episodes": len(entries),
                      "components": components, "score": sum(components.values()),
                      "success_rate": mean("success")})
    complete = len(valid) == planned
    score = float(np.mean([g["score"] for g in cells])) if cells else None
    distance = sum(e["distance_m"] for e in valid)
    collisions = sum(not e["collision_free"] for e in valid)
    return {"total_distance_m": distance, "collision_episodes": collisions,
            "collision_events_per_100m": 100 * collisions / distance if distance else None,
            "termination_counts": {reason: sum(e["reason"] == reason for e in valid)
                                   for reason in ("time_limit", "collision", "stuck", "looping", "slow", "out_of_bounds")},
            "planned_episodes": planned, "completed_episodes": len(valid),
            "system_failures": sum(e["status"] == "system_error" for e in episodes),
            "complete": complete, "cells": cells, "score": score,
            "passed": bool(complete and score >= minimum_score and all(e["collision_free"] for e in valid)
                           and all(g["success_rate"] >= minimum_success for g in cells)),
            "score_is_provisional": not complete}


def evaluate(args, runner, classifier, indices, vehicle):
    from . import config
    from .rl_env import AicarGazeboEnv
    from .evaluation import save_trajectory_plot
    from ..privileged.observation import GroundTruthMap
    from my_gazebo_launch.world_config import load_world_settings

    worlds = [args.world] if args.world else runner.EVAL_WORLDS
    scenarios = args.scenario or runner.EVAL_SCENARIOS
    planned = len(worlds) * len(scenarios) * args.episodes
    directory = runner.EVAL_OUTPUT_DIR / (datetime.now().strftime("evaluation_%Y%m%d_%H%M%S_") + uuid4().hex[:6])
    directory.mkdir(parents=True)
    model = args.model.expanduser().resolve()
    report = {"model": str(model), "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
              "vehicle": asdict(vehicle), "seed": runner.SEED, "seconds": args.eval_seconds,
              "scenarios": scenarios, "episodes_per_cell": args.episodes,
              "acceptance": {"minimum_score": runner.MIN_EVAL_SCORE, "minimum_success_rate": runner.MIN_SUCCESS_RATE, "zero_collisions": True},
              "safe_clearance_m": runner.SAFE_CLEARANCE_M, "checkpoint_radius_m": runner.CHECKPOINT_RADIUS_M,
              "progress_definition": "ordered ring-checkpoint chord distance / (target speed * duration), capped at 1",
              "clearance_definition": "map obstacle distance minus conservative body disk radius; sampled each control step",
              "execution": "paused Gazebo lidar stepping, not asynchronous GUI driver timing",
              "episodes": [], "status": "running"}

    def save():
        report["summary"] = summarize(report["episodes"], planned, runner.MIN_EVAL_SCORE, runner.MIN_SUCCESS_RATE)
        report["summary"]["passed"] &= report["status"] == "complete"
        temporary = directory / "report.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(directory / "evaluation.json")

    active = np.array([i for i in range(vehicle.lidar_point_count)
                       if not vehicle.lidar_front_sector_only or i < vehicle.lidar_point_count // 4
                       or i > 3 * vehicle.lidar_point_count // 4])
    save()
    print(f"评估输出：{directory}", flush=True)
    try:
        for world_name in worlds:
            world = load_world_settings(world_name)
            validate_ring_world(world)
            geometry = GroundTruthMap(world.world)
            period = max(config.STEP_SIM_TIME_SEC, 1 / vehicle.lidar_scan_hz + vehicle.sim_time_step_sec)
            settings = replace(config.EnvironmentSettings(), max_steps=math.ceil(args.eval_seconds / period),
                               workspace_bounds=world.workspace_bounds)
            for scenario in scenarios:
                for episode in range(args.episodes):
                    seed = runner.SEED + episode
                    pose = world.eval_spawn_poses[episode % len(world.eval_spawn_poses)]
                    run_dir = directory / world.world.stem / scenario / f"episode_{episode + 1:03d}"
                    run_dir.mkdir(parents=True)
                    entry = {"world": world.world.stem, "world_sha256": hashlib.sha256(world.world.read_bytes()).hexdigest(),
                             "scenario": scenario, "episode": episode + 1, "seed": seed, "spawn": pose,
                             "status": "running", "body_clearance_radius_m": geometry.settings.body_clearance_radius_m,
                             "settings": asdict(settings)}
                    env = None
                    trace, clearances, latencies = [], [], []
                    safe_time = switches = 0
                    previous = None
                    rng = np.random.default_rng(seed)
                    try:
                        progress = CheckpointProgress(world.train_spawn_poses, pose[:2], pose[3], runner.CHECKPOINT_RADIUS_M)
                        env = AicarGazeboEnv(role="eval", domain_id=runner.ROS_DOMAIN, port=runner.GAZEBO_PORT,
                                            header=runner.CONFIG_HEADER, world=world.world, spawn_pose=pose,
                                            seed=seed, log_dir=run_dir, settings=settings, gui=False,
                                            observation_mode="lidar", capture_lidar=True)
                        _, info = env.reset(seed=seed)
                        trace.append([0., *info["position"], info["yaw"], 0., vehicle.straight_label, 0.])
                        with (run_dir / "steps.jsonl").open("w") as steps:
                            while True:
                                scan = corrupt_scan(info["lidar_mm"], scenario, rng, active)
                                started = time.perf_counter()
                                action = int(classifier.predict(scan[indices].reshape(1, -1))[0])
                                latencies.append((time.perf_counter() - started) * 1000)
                                switches += int(previous is not None and action != previous)
                                previous = action
                                _, reward, terminated, truncated, info = env.step(action)
                                x, y = info["position"]
                                clearance, _ = geometry.safety_distances(dict(x=x, y=y, yaw=info["yaw"]))
                                clearances.append(clearance)
                                safe_time += info["sim_dt"] * (clearance >= runner.SAFE_CLEARANCE_M)
                                progress.update((x, y))
                                trace.append([info["episode_sim_time_sec"], x, y, info["yaw"], info["linear_velocity"], action, reward])
                                steps.write(json.dumps({"time": info["episode_sim_time_sec"], "x": x, "y": y,
                                                       "yaw": info["yaw"], "action": action, "clearance_m": clearance,
                                                       "collided": info["collided"], "progress_m": progress.distance}) + "\n")
                                if terminated or truncated:
                                    break
                        duration = info["episode_sim_time_sec"]
                        entry.update(status="complete", reason=info["termination_reason"],
                                     collision_free=not info["collided"], success=bool(truncated and not terminated),
                                     sim_duration_sec=duration, distance_m=info["episode_distance_m"],
                                     first_collision_sec=duration if info["collided"] else None,
                                     progress_m=progress.distance, checkpoint_crossings=progress.crossings,
                                     progress_ratio=min(1., progress.distance / (vehicle.forward_speed_mps * args.eval_seconds)),
                                     safe_time_ratio=safe_time / duration, clearance_min_m=min(clearances),
                                     clearance_p05_m=float(np.percentile(clearances, 5)),
                                     action_switches=switches, inference_p95_ms=float(np.percentile(latencies, 95)),
                                     speed_compliance=info["speed_compliance"])
                        print(f"{world.world.stem}/{scenario}/{episode+1}: {entry['reason']}, {duration:.1f}s, "
                              f"{entry['distance_m']:.2f}m, 净空最小 {entry['clearance_min_m']:.3f}m", flush=True)
                    except BaseException as error:
                        entry.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "system_error", error_type=type(error).__name__, error=str(error))
                        raise
                    finally:
                        report["episodes"].append(entry)
                        try:
                            np.save(run_dir / "trajectory.npy", np.asarray(trace))
                            save()
                        finally:
                            if env is not None:
                                env.close()
                    save_trajectory_plot(run_dir / "trajectory.png", [trace], [entry], world.world, settings)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        save()
    print(f"评分：{report['summary']['score']:.2f}/100；通过：{report['summary']['passed']}；{directory}", flush=True)
    return directory
