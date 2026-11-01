#!/usr/bin/env python3
"""Matched normal/sector diagnostic traces for two frozen student models."""
from pathlib import Path
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
import os
import statistics
import sys
import uuid
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DISSERTATION = ROOT.parent
if os.environ.get("P1_DIAG_ENV_READY") != "1":
    os.execvpe("bash", ["bash", "-c",
        'set -e; source /opt/ros/humble/setup.bash; '
        'source /usr/share/gazebo/setup.bash; source "$1/install/setup.bash"; '
        'export P1_DIAG_ENV_READY=1; shift; exec "$@"',
        "diagnose_student_sector", str(ROOT), sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        os.environ)

import numpy as np

sys.path.insert(0, str(ROOT))
from my_gazebo_launch.config import load_config
from my_gazebo_launch.world_config import load_world_settings
from rl_training.common import config
from rl_training.common.rl_env import AicarGazeboEnv
from rl_training.common.forest_evaluation import corrupt_scan
from rl_training.privileged.observation import GroundTruthMap
from run_random_forest import load_model

CONFIG_HEADER = ROOT / "my_gazebo_launch/vehicle_config.h"
WORLD = ROOT / "assets/worlds/student_test/rounded_rectangle_001.sdf"
START_PLAN = DISSERTATION / "article/experiment_data/r2_start_plans/rounded_rectangle_001_starts.json"
DEFAULT_MODELS = (DISSERTATION / "software/ML_model/simulation_tree6/randomForest.joblib",
                  DISSERTATION / "software/ML_model/simulation_both/randomForest.joblib")
STARTS = (1, 2, 5)
SEEDS = {1: 20261034, 2: 20261035, 5: 20261038}
CONDITIONS = ("normal", "sector")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in (DISSERTATION, ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (DISSERTATION / path).resolve()


def corrupt_scan_with_indices(scan, scenario, rng, active):
    scan = np.asarray(scan, dtype=np.float32).copy()
    if scenario == "normal":
        return scan, np.empty(0, dtype=np.int64)
    if scenario != "sector":
        raise ValueError(f"unsupported scenario: {scenario}")
    ordered = np.r_[active[active > 3 * len(scan) // 4], active[active < len(scan) // 4]]
    start = int(rng.integers(len(ordered) - 20 + 1))
    indices = ordered[start:start + 20].astype(np.int64)
    scan[indices] = 0
    return scan, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--start", action="append", type=int, dest="starts")
    parser.add_argument("--condition", action="append", choices=CONDITIONS, dest="conditions")
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--domain", type=int, default=95)
    parser.add_argument("--port", type=int, default=11555)
    parser.add_argument("--output-dir", default=str(ROOT / "results/student_diagnostics"))
    parser.add_argument("--check", action="store_true", help="只检查模型/地图/起点和遮挡函数，不启动仿真")
    args = parser.parse_args()
    models = [resolve_path(m) for m in (args.models or DEFAULT_MODELS)]
    starts = args.starts or list(STARTS)
    conditions = args.conditions or list(CONDITIONS)
    if not all(s in SEEDS for s in starts):
        raise SystemExit(f"starts 必须是 {sorted(SEEDS)}；未知起点没有固定诊断种子。")
    if args.seconds <= 0 or not math.isfinite(args.seconds):
        raise SystemExit("seconds 必须为正有限数。")
    vehicle = load_config(CONFIG_HEADER)
    world = load_world_settings(WORLD)
    plan = json.loads(START_PLAN.read_text(encoding="utf-8"))
    if plan.get("world_sha256") != GroundTruthMap(world.world).sha256:
        raise SystemExit("start plan 与地图 SHA-256 不一致。")
    entries = plan["categories"]["normal"]
    if any(not 1 <= start <= len(entries) for start in starts):
        raise SystemExit("start 超出 normal 起点范围。")
    loaded = []
    for model_path in models:
        classifier, indices = load_model(model_path, vehicle)
        loaded.append((model_path, model_path.parent.name, classifier, indices))
    active = np.array([i for i in range(vehicle.lidar_point_count)
                       if not vehicle.lidar_front_sector_only or i < vehicle.lidar_point_count // 4
                       or i > 3 * vehicle.lidar_point_count // 4])
    sample = np.arange(200, dtype=np.float32) + 1
    expected = corrupt_scan(sample, "sector", np.random.default_rng(123), active)
    actual, mask = corrupt_scan_with_indices(sample, "sector", np.random.default_rng(123), active)
    if not np.array_equal(expected, actual) or len(mask) != 20:
        raise SystemExit("诊断遮挡函数与 forest_evaluation.corrupt_scan 不一致。")
    print(f"模型：{[str(m) for m in models]}\n地图：{world.world}\n起点：{starts}；条件：{conditions}\n"
          f"矩阵：{len(models)} × {len(starts)} × {len(conditions)} = {len(models) * len(starts) * len(conditions)} 回合；"
          f"{args.seconds:g} 秒/回合", flush=True)
    if args.check:
        print("遮挡函数一致；配置与试推理检查通过；未启动仿真。")
        return 0

    directory = Path(args.output_dir) / (datetime.now(ZoneInfo("Asia/Shanghai")).strftime("diagnostic_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    manifest = {
        "status": "running", "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "world": str(world.world), "world_sha256": GroundTruthMap(world.world).sha256,
        "start_plan": str(START_PLAN), "start_plan_sha256": sha256(START_PLAN),
        "starts": starts, "seeds": {str(s): SEEDS[s] for s in starts}, "conditions": conditions,
        "vehicle": asdict(vehicle), "models": {name: {"path": str(path), "sha256": sha256(path)} for path, name, _, _ in loaded},
        "source_sha256": {name: sha256(ROOT / name) for name in (
            "diagnose_student_sector.py", "evaluate_student_recovery.py", "run_random_forest.py",
            "rl_training/common/rl_env.py", "rl_training/common/forest_evaluation.py",
            "my_gazebo_launch/vehicle_config.h")},
        "versions": {"python": sys.version, "numpy": np.__version__, "sklearn": __import__("sklearn").__version__,
                     "gymnasium": __import__("gymnasium").__version__, "stable_baselines3": __import__("stable_baselines3").__version__},
        "command": sys.argv,
    }
    (directory / "diagnostic_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_rows = []
    try:
        step_seconds = max(config.EnvironmentSettings().step_sim_sec, 1 / vehicle.lidar_scan_hz + vehicle.sim_time_step_sec)
        settings = replace(config.EnvironmentSettings(), max_steps=math.ceil(args.seconds / step_seconds),
                           workspace_bounds=world.workspace_bounds)
        geometry = GroundTruthMap(world.world)
        for model_path, model_label, classifier, indices in loaded:
            env = None
            try:
                env = AicarGazeboEnv(role="eval", domain_id=args.domain, port=args.port, header=CONFIG_HEADER,
                                     world=world.world, spawn_pose=entries[0]["pose"], seed=SEEDS[starts[0]],
                                     log_dir=directory / model_label, settings=settings, gui=False,
                                     real_time_factor=0.0, observation_mode="lidar", capture_lidar=True)
                for start in starts:
                    seed = SEEDS[start]
                    start_pose = entries[start - 1]["pose"]
                    for condition in conditions:
                        run_dir = directory / model_label / f"start_{start}" / condition
                        run_dir.mkdir(parents=True)
                        rng = np.random.default_rng(seed)
                        env.spawn_pose = tuple(start_pose)
                        obs, info = env.reset(seed=seed)
                        initial_clearance, initial_front = geometry.safety_distances(dict(
                            x=info["position"][0], y=info["position"][1], yaw=info["yaw"]))
                        steps = []
                        scan_sequence = 0
                        last_stamp = None
                        switches = 0
                        previous_action = None
                        first_occlusion = None
                        first_collision = None
                        while True:
                            raw_scan = np.asarray(info["lidar_mm"], dtype=np.float32)
                            model_scan, occluded = corrupt_scan_with_indices(raw_scan, condition, rng, active)
                            action = int(classifier.predict(model_scan[indices].reshape(1, -1))[0])
                            if previous_action is not None and action != previous_action:
                                switches += 1
                            previous_action = action
                            stamp = int(info["lidar_stamp_ns"])
                            if stamp != last_stamp:
                                scan_sequence += 1
                                last_stamp = stamp
                            decision_stamp = int(info["odom_stamp_ns"])
                            row = {
                                "step": int(info.get("step", 0)),
                                "sim_time_before": float(info.get("episode_sim_time_sec", 0.0)),
                                "decision_stamp_ns": decision_stamp,
                                "lidar_stamp_ns": stamp,
                                "observation_age_ns": decision_stamp - stamp,
                                "scan_sequence": scan_sequence,
                                "raw_scan_mm": raw_scan.astype(int).tolist(),
                                "model_scan_mm": model_scan.astype(int).tolist(),
                                "occluded_indices": occluded.astype(int).tolist(),
                                "pre_x": float(info["position"][0]), "pre_y": float(info["position"][1]),
                                "pre_yaw": float(info["yaw"]), "pre_speed": float(info.get("linear_velocity", 0.0)),
                                "action": action,
                            }
                            if occluded.size and first_occlusion is None:
                                first_occlusion = row["sim_time_before"]
                            obs, reward, terminated, truncated, info = env.step(action)
                            clearance, front = geometry.safety_distances(dict(
                                x=info["position"][0], y=info["position"][1], yaw=info["yaw"]))
                            if info.get("collided") and first_collision is None:
                                first_collision = float(info["episode_sim_time_sec"])
                            row.update({
                                "post_x": float(info["position"][0]), "post_y": float(info["position"][1]),
                                "post_yaw": float(info["yaw"]), "post_speed": float(info["linear_velocity"]),
                                "clearance_m": float(clearance), "front_clearance_m": float(front),
                                "collided": bool(info.get("collided")),
                                "termination_reason": info.get("termination_reason", ""), "reward": float(reward),
                            })
                            steps.append(row)
                            if terminated or truncated:
                                break
                        success = bool(truncated and not terminated and info.get("termination_reason") == "time_limit")
                        duration = float(info["episode_sim_time_sec"])
                        clearances = [initial_clearance] + [s["clearance_m"] for s in steps]
                        episode = {
                            "model": model_label, "start": start, "condition": condition, "seed": seed,
                            "spawn": start_pose, "initial_clearance_m": initial_clearance,
                            "initial_front_clearance_m": initial_front, "spawn_yaw": start_pose[3],
                            "reset_yaw": float(steps[0]["pre_yaw"]) if steps else None,
                            "success": success, "collided": bool(info.get("collided")),
                            "reason": info.get("termination_reason"), "duration_sec": duration,
                            "distance_m": float(info["episode_distance_m"]), "mean_speed_mps": float(info["episode_mean_speed_mps"]),
                            "min_clearance_m": min(clearances), "p05_clearance_m": float(np.percentile(clearances, 5)),
                            "action_switches": switches, "steps": len(steps), "scan_sequences": scan_sequence,
                            "occluded_steps": sum(1 for s in steps if s["occluded_indices"]),
                            "first_occlusion_time_sec": first_occlusion, "first_collision_time_sec": first_collision,
                        }
                        with (run_dir / "steps.jsonl").open("w", encoding="utf-8") as handle:
                            for row in steps:
                                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        (run_dir / "episode.json").write_text(json.dumps(episode, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                        np.save(run_dir / "trajectory.npy", np.asarray(
                            [[s["sim_time_before"], s["pre_x"], s["pre_y"], s["pre_yaw"], s["pre_speed"],
                              s["action"], s["post_x"], s["post_y"], s["post_yaw"], s["post_speed"],
                              s["clearance_m"], len(s["occluded_indices"])] for s in steps], dtype=np.float64))
                        summary_rows.append(episode)
                        print(f"{model_label}/start_{start}/{condition}: {episode['reason']}, success={success}, "
                              f"t={duration:.1f}s, min_clear={episode['min_clearance_m']:.3f}, occluded_steps={episode['occluded_steps']}", flush=True)
            finally:
                if env is not None:
                    env.close()
        summary_rows.sort(key=lambda e: (e["model"], e["start"], e["condition"]))
        with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            import csv
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader(); writer.writerows(summary_rows)
        (directory / "summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.update(status="complete", ended_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                        episodes=len(summary_rows), summary=str(directory / "summary.csv"))
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        ended_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                        error=f"{type(error).__name__}: {error}")
        (directory / "diagnostic_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    (directory / "diagnostic_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"完成：{directory}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
