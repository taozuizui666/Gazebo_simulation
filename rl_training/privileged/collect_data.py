#!/usr/bin/env python3
"""PPO 教师驾驶采集雷达/动作，生成随机置零副本，并复制到 dataset/simulation。"""
from pathlib import Path
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime
from io import BytesIO
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

# ===================== 可修改配置区 =====================
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "rl_models/privileged/policies/latest_validated/model.zip"
WORLD_DIR = ROOT / "assets/worlds/simple"
WORLDS = [WORLD_DIR / name for name in (
    "train_world3.sdf",
    "rounded_rectangle_001.sdf",
    "ellipse_002.sdf",
    "rounded_rectangle_003.sdf",
)]
OUTPUT_DIR = ROOT / "results/datasets"          # 原始采集、增强数据、时序与日志
DATASET_DIR = ROOT.parent / "dataset/simulation"  # 后续随机森林读取的副本
COLLECTION_SECONDS = 300                      # 每轮最多采集多少仿真秒
COLLECTION_ROUNDS = 5                         # 每张地图采集多少轮，依次使用 JSON 中的评估出生点
GUI = True                                  # True：显示优先；False：无窗口、不限速采集
GUI_REAL_TIME_FACTOR = 1.0                   # 仅 GUI 生效；1 为实时，0.5 为半速，必须 > 0
SEED = 42                                    # 仿真种子
DROPOUT_FRACTION = 0.05                       # 每行非零雷达点中约 5% 随机置零
DROPOUT_SEED = 42                             # 增强种子；原始/增强行一一对应，数量 1:1
ROS_DOMAIN = 92
GAZEBO_PORT = 11552                           # 采集使用独立的 ROS/Gazebo 实例
# 雷达点数、扇区、距离单位及控制标签从 MODEL 同目录 vehicle_config.h 读取。
# ===================== 配置区结束 =====================


def write_metadata(directory, metadata):
    temporary = directory / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "metadata.json")


def augment_file(source, destination, vehicle, fraction, rng):
    """每行抽取 round(非零点数 * fraction) 个位置；不修改标签或原始文件。"""
    from my_gazebo_launch.lidar_processing import training_row

    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("置零比例必须在 [0, 1]。")
    rows = nonzero = masked = 0
    with source.open(encoding="utf-8") as original, destination.open("x", encoding="utf-8") as output:
        for line in original:
            values = list(map(int, line.strip().split(",")))
            distances, label = values[:-1], values[-1]
            training_row(distances, label, vehicle)  # 校验格式，不过滤/清理样本
            indices = [i for i, value in enumerate(distances) if value > 0]
            count = int(len(indices) * fraction + 0.5)
            for index in rng.sample(indices, count):
                distances[index] = 0
            output.write(training_row(distances, label, vehicle))
            rows += 1
            nonzero += len(indices)
            masked += count
    return {"rows": rows, "nonzero_points_before": nonzero, "masked_points": masked}


def publish_dataset(directory, dataset_dir, filenames):
    """只复制训练 CSV；核对成功后发布批次，元信息和时序留在采集目录。"""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    destination = dataset_dir / directory.name
    if destination.exists():
        raise FileExistsError(destination)
    # 在 dataset 之外暂存，避免训练器递归读入尚未复制完整的 CSV。
    with tempfile.TemporaryDirectory(prefix=".rl_dataset_", dir=dataset_dir.parent.parent) as temporary:
        staging = Path(temporary) / directory.name
        staging.mkdir()
        for filename in filenames:
            if not filename.endswith(".csv"):
                continue
            source, target = directory / filename, staging / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if source.read_bytes() != target.read_bytes():
                raise IOError(f"复制校验失败：{filename}")
        staging.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="检查模型/地图/配置，不启动采集")
    parser.add_argument("--headless", action="store_true", help="关闭 Gazebo 图形窗口")
    parser.add_argument("--seconds", type=float, default=COLLECTION_SECONDS, help="每轮仿真时间上限（秒）")
    parser.add_argument("--rounds", type=int, default=COLLECTION_ROUNDS, help="每张地图的采集轮数")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args()
    if os.environ.get("RL_COLLECT_ENV_READY") != "1":
        os.execvpe("bash", ["bash", "-c",
            'set -e; source /opt/ros/humble/setup.bash; '
            'source /usr/share/gazebo/setup.bash; source "$1/install/setup.bash"; '
            'export RL_COLLECT_ENV_READY=1; shift; exec "$@"',
            "collect_data", str(ROOT), sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            os.environ)
    sys.path.insert(0, str(ROOT))
    from my_gazebo_launch.lidar_processing import training_row
    from my_gazebo_launch.world_config import load_world_settings
    from rl_training.common import config
    from rl_training.common.model_io import load_model
    from rl_training.common.observations import observation_specification
    from rl_training.common.rl_env import AicarGazeboEnv, SimulationDataError
    from rl_training.privileged.observation import GroundTruthMap
    import torch

    if not WORLDS or type(args.rounds) is not int or args.rounds < 1:
        raise ValueError("地图列表不能为空，轮数必须为正整数。")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("每轮采集秒数必须为正的有限数。")
    if not math.isfinite(DROPOUT_FRACTION) or not 0 <= DROPOUT_FRACTION <= 1:
        raise ValueError("DROPOUT_FRACTION 必须在 [0, 1]。")
    model_path = MODEL.expanduser().resolve(strict=True)
    saved = json.loads((model_path.parent / "run_config.json").read_text())
    header = model_path.parent / "vehicle_config.h"
    vehicle = config.vehicle_config(header)
    spec = observation_specification("privileged", vehicle)
    model_bytes = model_path.read_bytes()
    model = load_model(BytesIO(model_bytes), device=config.DEVICE)
    if (saved.get("observation_mode") != "privileged" or saved["vehicle"] != asdict(vehicle)
            or saved.get("observation_spec") != spec
            or model.observation_space.shape != (spec["dimension"],)
            or model.action_space.n != vehicle.label_count):
        raise ValueError("教师模型与观测/车辆/动作配置不一致。")
    settings = config.EnvironmentSettings(**saved["environment"]).validate()
    step_seconds = math.ceil(settings.step_sim_sec / vehicle.sim_time_step_sec) * vehicle.sim_time_step_sec
    # 只采集完整控制步；小容差消除 0.6 / 0.2 等浮点运算的舍入误差。
    max_steps = math.floor(args.seconds / step_seconds + 1e-9)
    if max_steps < 1:
        raise ValueError(f"每轮至少需要一个控制步，即 {step_seconds:g} 秒。")
    settings = replace(settings, max_steps=max_steps)
    if step_seconds < 1 / vehicle.lidar_scan_hz + vehicle.sim_time_step_sec:
        raise ValueError("教师控制周期不足以获得新雷达帧。")
    worlds = [load_world_settings(path) for path in WORLDS]
    if len({world.world.stem for world in worlds}) != len(worlds):
        raise ValueError("地图文件名不能重复；每张地图按文件名保存一份训练集。")
    for world in worlds:
        GroundTruthMap(world.world)
    output_dir = args.output_dir.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if output_dir.is_relative_to(dataset_dir.parent):
        raise ValueError("原始采集输出应放在 dataset 目录之外，避免重复读入。")
    gui = GUI and not args.headless
    if gui and (not math.isfinite(GUI_REAL_TIME_FACTOR) or GUI_REAL_TIME_FACTOR <= 0):
        raise ValueError("GUI_REAL_TIME_FACTOR 必须为正的有限数。")
    real_time_factor = GUI_REAL_TIME_FACTOR if gui else 0.0
    mode = (f"GUI 显示优先（最高 {real_time_factor:g} 倍速，关闭阴影）" if gui
            else "无 GUI，采集优先（不限速）")
    print(f"教师：{model_path}\n地图：{', '.join(w.world.name for w in worlds)}\n"
          f"运行模式：{mode}\n"
          f"每图 {args.rounds} 轮，每轮最多 {max_steps * step_seconds:g} 仿真秒"
          f"（配置 {args.seconds:g} 秒，{max_steps} 步）；"
          f"随机置零 {DROPOUT_FRACTION:.0%}\n采集目录：{output_dir}\n训练副本：{dataset_dir}", flush=True)
    if args.check:
        print("配置检查通过；未启动采集。")
        return
    torch.set_num_threads(config.TORCH_THREADS)
    batch = f"collection_{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    directory = output_dir / batch
    directory.mkdir(parents=True)
    (directory / "raw").mkdir()
    (directory / "dropout").mkdir()
    metadata = {
        "schema_version": 2, "batch": batch, "status": "collecting",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_directory": str(directory), "dataset_directory": str(dataset_dir / batch),
        "model": str(model_path), "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "model_training_timesteps": saved.get("training_timesteps"),
        "observation_spec": spec, "vehicle": asdict(vehicle), "seed": SEED, "gui": gui,
        "collection": {"seconds_per_round": args.seconds, "rounds_per_map": args.rounds,
                       "real_time_factor": real_time_factor,
                       "step_seconds": step_seconds, "steps_per_round": max_steps,
                       "actual_max_seconds_per_round": max_steps * step_seconds},
        "format": {"columns": vehicle.lidar_point_count + 1, "header": False,
                   "distance_unit": "mm", "last_column": "action", "zero": "invalid_or_unobserved"},
        "labels": {"values": list(vehicle.labels), "straight": vehicle.straight_label,
                   "minimum": "maximum_left", "maximum": "maximum_right"},
        "pairing": "latest fresh lidar at decision boundary -> action executed next; timestamps in ns",
        "dropout": {"fraction": DROPOUT_FRACTION, "seed": DROPOUT_SEED,
                    "copies_per_raw_row": 1, "sampling": "nonzero points, without replacement",
                    "count": "floor(nonzero_count * fraction + 0.5)"},
        "split_group_rule": "Keep each map's raw/dropout files together; episode row ranges are 1-based inclusive.",
        "maps": [], "episodes": [],
    }
    write_metadata(directory, metadata)
    filenames = []
    env = None
    try:
        for world in worlds:
            stem = world.world.stem
            raw_name = f"raw/{stem}_raw.csv"
            dropout_name = f"dropout/{stem}_dropout.csv"
            timing_name = f"{stem}_timing.jsonl"
            map_rows = 0
            map_counts = Counter()
            env = AicarGazeboEnv(
                role="eval", domain_id=ROS_DOMAIN, port=GAZEBO_PORT, header=header,
                world=world.world, spawn_pose=world.eval_spawn_poses[0], seed=SEED,
                settings=replace(settings, workspace_bounds=world.workspace_bounds),
                reward_settings=config.RewardSettings(**saved["reward"]),
                log_dir=directory / f"runtime_{stem}",
                gui=gui, real_time_factor=real_time_factor,
                observation_mode="privileged", capture_lidar=True)
            if gui:
                time.sleep(config.GUI_STARTUP_SETTLE_SEC)
            with (directory / raw_name).open("x", encoding="utf-8") as raw, \
                    (directory / timing_name).open("x", encoding="utf-8") as timing:
                for index in range(args.rounds):
                    number = len(metadata["episodes"]) + 1
                    env.spawn_pose = tuple(world.eval_spawn_poses[index % len(world.eval_spawn_poses)])
                    obs, current = env.reset(seed=SEED + number - 1)
                    round_started = time.monotonic()
                    counts = Counter()
                    previous_stamp = -1
                    row_start = map_rows + 1
                    while True:
                        stamp = current["lidar_stamp_ns"]
                        decision_ns = current["odom_stamp_ns"]  # exact teacher physics time
                        if not previous_stamp < stamp <= decision_ns:
                            raise SimulationDataError("雷达时间戳重复、倒退或晚于动作决策。")
                        action, _ = model.predict(obs, deterministic=True)
                        action = int(action)
                        row = training_row(current["lidar_mm"].tolist(), action, vehicle)
                        obs, _, terminated, truncated, following = env.step(action)
                        if following["action"] != action:
                            raise SimulationDataError("记录动作与执行动作不同。")
                        raw.write(row)
                        map_rows += 1
                        timing.write(json.dumps({
                            "row": map_rows, "round": index + 1, "step": following["step"],
                            "lidar_stamp_ns": stamp,
                            "decision_stamp_ns": decision_ns, "action": action,
                            "action_end_stamp_ns": following["odom_stamp_ns"],
                        }) + "\n")
                        counts[action] += 1
                        previous_stamp, current = stamp, following
                        if following["step"] % 100 == 0:
                            raw.flush()
                            timing.flush()
                            print(f"{stem} 第 {index + 1}/{args.rounds} 轮：{following['step']} 条，"
                                  f"{following['episode_sim_time_sec']:.1f} 仿真秒", flush=True)
                        if terminated or truncated:
                            break
                    raw.flush()
                    timing.flush()
                    wall_seconds = time.monotonic() - round_started
                    map_counts.update(counts)
                    metadata["episodes"].append({
                        "group_id": f"{batch}/{stem}", "round": index + 1, "world": str(world.world),
                        "world_sha256": env.ground_truth_map.sha256,
                        "spawn_pose": env.spawn_pose, "settings": asdict(env.settings),
                        "raw_file": raw_name, "dropout_file": dropout_name, "timing_file": timing_name,
                        "row_start": row_start, "row_end": map_rows,
                        "rows": following["step"], "action_counts": dict(sorted(counts.items())),
                        "termination_reason": following["termination_reason"],
                        "success": following["is_success"],
                        "sim_seconds": following["episode_sim_time_sec"],
                        "wall_seconds": wall_seconds,
                        "distance_m": following["episode_distance_m"],
                        "mean_speed_mps": following["episode_mean_speed_mps"],
                    })
                    write_metadata(directory, metadata)
                    print(f"{stem} 第 {index + 1}/{args.rounds} 轮结束："
                          f"{following['termination_reason']}，{following['step']} 条，"
                          f"本地图累计 {map_rows} 条，实际采集耗时 {wall_seconds:.1f} 秒"
                          f"（{following['step'] / wall_seconds:.1f} 条/秒）。", flush=True)
            metadata["maps"].append({
                "group_id": f"{batch}/{stem}", "world": str(world.world),
                "world_sha256": env.ground_truth_map.sha256, "rounds": args.rounds,
                "raw_file": raw_name, "dropout_file": dropout_name, "timing_file": timing_name,
                "rows": map_rows, "action_counts": dict(sorted(map_counts.items())),
            })
            filenames.extend([raw_name, timing_name])
            write_metadata(directory, metadata)
            env.close()
            env = None
        rng = random.Random(DROPOUT_SEED)
        for entry in metadata["maps"]:
            name = entry["dropout_file"]
            entry["augmentation"] = augment_file(
                directory / entry["raw_file"], directory / name, vehicle, DROPOUT_FRACTION, rng)
            filenames.append(name)
        metadata["raw_rows"] = sum(e["rows"] for e in metadata["maps"])
        metadata["dropout_rows"] = sum(e["augmentation"]["rows"] for e in metadata["maps"])
        metadata["status"] = "complete"
        metadata["completed_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        metadata["file_sha256"] = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in filenames}
        write_metadata(directory, metadata)
        destination = publish_dataset(directory, dataset_dir, filenames)
        print(f"完成：原始 {metadata['raw_rows']} 条 + 增强 {metadata['dropout_rows']} 条。\n"
              f"采集结果：{directory}\n训练副本：{destination}", flush=True)
    except BaseException as error:
        metadata["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        write_metadata(directory, metadata)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
