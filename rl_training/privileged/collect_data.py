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
import re
import shutil
import sys
import tempfile
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

# ===================== 可修改配置区 =====================
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "rl_models/privileged/policies/latest_validated/model.zip"
LIDAR_CONFIG = ROOT / "my_gazebo_launch/vehicle_config.h"  # 采集雷达的扫描频率和原始点数
WORLD_DIR = ROOT / "assets/worlds/simple"
WORLDS = [WORLD_DIR / name for name in (
    "train_world3.sdf",
    "rounded_rectangle_001.sdf",
    "ellipse_002.sdf",
    "rounded_rectangle_003.sdf",
)]
OUTPUT_DIR = ROOT / "results/datasets"          # 原始采集、增强数据、时序与日志
DATASET_DIR = ROOT.parent / "dataset/simulation"  # 后续随机森林读取的副本
COLLECTION_SECONDS = 3600                     # 每轮最多采集多少仿真秒
COLLECTION_ROUNDS = 20                        # 每张地图采集多少轮，以 JSON 中的评估出生点为基准
SPAWN_JITTER_ENABLED = True                  # 普通轮次随机出生姿态；False 使用原出生姿态
SPAWN_POSITION_JITTER_M = 0.03                # 普通轮次 x/y 各 ±3 cm；近墙轮次用于沿路方向
SPAWN_YAW_JITTER_DEG = 10.0                   # 普通轮次朝向 ±10 度
RECOVERY_ENABLED = True                      # 部分轮次从偏向墙壁的姿态开始，由原教师纠正
RECOVERY_FRACTION = 0.25                      # 每图近墙轮数 = floor(总轮数 × 比例 + 0.5)
RECOVERY_MAX_OFFSET_M = 0.12                  # 相对基准出生点，向左/右横移的最大值
RECOVERY_CLEARANCE_M = (0.04, 0.08)           # 近墙时车体保守圆到墙的净空范围，不是雷达距离
RECOVERY_YAW_DEG = (10.0, 20.0)               # 向横移一侧偏航的角度范围
RECOVERY_FRONT_M = (0.15, 0.40)               # 前进方向保守车体首次接触障碍前的距离范围
SPAWN_MARGIN_M = 0.01                        # 所有出生点、近墙纠正转弯路径的额外安全余量
SPAWN_ATTEMPTS = 200                         # 候选姿态尝试上限；失败明确报错，不退回固定起点
LIDAR_WAIT_SEC = 0.2                          # 每次等待新扫描的现实秒数；超时后继续驾驶，不生成该行
MAX_CONSECUTIVE_LIDAR_MISSES = 5              # 连续多少个决策点缺帧后报错停止，避免一直空采
GUI = False                                  # True：显示优先；False：无窗口、不限速采集
GUI_REAL_TIME_FACTOR = 1.0                   # 仅 GUI 生效；1 为实时，0.5 为半速，必须 > 0
SEED = 42                                    # 仿真种子
DROPOUT_FRACTION = 0.05                       # 每行非零雷达点中约 5% 随机置零
DROPOUT_SEED = 42                             # 增强种子；原始/增强行一一对应，数量 1:1
ROS_DOMAIN = 92
GAZEBO_PORT = 11552                           # 采集使用独立的 ROS/Gazebo 实例
# 仅扫描频率、原始点数读取 LIDAR_CONFIG；200 桶格式、车辆及动作仍沿用模型配置。
# ===================== 配置区结束 =====================


def write_metadata(directory, metadata):
    temporary = directory / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "metadata.json")


def count_rows(path):
    with path.open("rb") as rows:
        return sum(1 for _ in rows)


def spawn_settings(step_seconds):
    return {"position_jitter_m": SPAWN_POSITION_JITTER_M if SPAWN_JITTER_ENABLED else 0.0,
            "yaw_jitter_deg": SPAWN_YAW_JITTER_DEG if SPAWN_JITTER_ENABLED else 0.0,
            "recovery_fraction": RECOVERY_FRACTION if RECOVERY_ENABLED else 0.0,
            "recovery_max_offset_m": RECOVERY_MAX_OFFSET_M,
            "recovery_clearance_m": list(RECOVERY_CLEARANCE_M),
            "recovery_yaw_deg": list(RECOVERY_YAW_DEG), "recovery_front_m": list(RECOVERY_FRONT_M),
            "margin_m": SPAWN_MARGIN_M, "attempts": SPAWN_ATTEMPTS, "step_seconds": step_seconds}


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
    parser.add_argument("--resume", type=Path, help="继续一个失败或中断的批次目录，跳过其中已完成的地图")
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
    from rl_training.privileged.collection_spawns import build_spawn_plan
    import torch

    if not WORLDS or type(args.rounds) is not int or args.rounds < 1:
        raise ValueError("地图列表不能为空，轮数必须为正整数。")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("每轮采集秒数必须为正的有限数。")
    if not math.isfinite(DROPOUT_FRACTION) or not 0 <= DROPOUT_FRACTION <= 1:
        raise ValueError("DROPOUT_FRACTION 必须在 [0, 1]。")
    if not math.isfinite(LIDAR_WAIT_SEC) or LIDAR_WAIT_SEC <= 0:
        raise ValueError("LIDAR_WAIT_SEC 必须为正的有限数。")
    if type(MAX_CONSECUTIVE_LIDAR_MISSES) is not int or MAX_CONSECUTIVE_LIDAR_MISSES < 1:
        raise ValueError("MAX_CONSECUTIVE_LIDAR_MISSES 必须为正整数。")
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
    teacher_vehicle = vehicle
    lidar_config = config.vehicle_config(LIDAR_CONFIG)
    vehicle = replace(vehicle, lidar_scan_hz=lidar_config.lidar_scan_hz,
                      lidar_raw_samples=lidar_config.lidar_raw_samples).validate()
    # 教师只读地图/物理状态；只替换采集传感器的两个参数，不修改模型目录。
    collection_header = header.read_text(encoding="utf-8")
    for field in ("lidar_scan_hz", "lidar_raw_samples"):
        collection_header, count = re.subn(
            rf"^([ \t]*#define[ \t]+AICAR_{field.upper()}[ \t]+)[^\r\n]*",
            lambda match: f"{match[1]}{getattr(vehicle, field)}", collection_header, flags=re.M)
        if count != 1:
            raise ValueError(f"教师配置中的 AICAR_{field.upper()} 必须恰好定义一次。")
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
    spawn_config = spawn_settings(step_seconds)
    spawn_plans = {}
    for index, world in enumerate(worlds):
        geometry = GroundTruthMap(world.world)
        spawn_plans[world.world.stem] = {
            "world": str(world.world), "world_sha256": geometry.sha256, "seed": SEED + index,
            "rounds": build_spawn_plan(world, geometry, vehicle, args.rounds, SEED + index, spawn_config)}
    output_dir = args.output_dir.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if output_dir.is_relative_to(dataset_dir.parent):
        raise ValueError("原始采集输出应放在 dataset 目录之外，避免重复读入。")
    resume_directory = args.resume.expanduser().resolve() if args.resume else None
    completed_maps, pending_worlds = [], worlds
    if resume_directory is not None:
        if not resume_directory.is_relative_to(output_dir):
            raise ValueError("续采目录必须位于采集输出目录内。")
        metadata_path = resume_directory / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"续采目录缺少 metadata.json：{resume_directory}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") not in ("failed", "interrupted"):
            raise ValueError("只能续采 failed 或 interrupted 状态的批次。")
        if (metadata.get("model") != str(model_path)
                or metadata.get("model_sha256") != hashlib.sha256(model_bytes).hexdigest()):
            raise ValueError("续采批次的教师模型与当前模型不一致。")
        if metadata.get("vehicle") != asdict(vehicle) or metadata.get("observation_spec") != spec:
            raise ValueError("续采批次的车辆/雷达或观测配置与当前配置不一致；请新建采集批次。")
        if metadata.get("spawn_randomization") != spawn_config or metadata.get("spawn_plans") != spawn_plans:
            raise ValueError("续采批次的出生场景/地图/种子与当前配置不一致；请新建采集批次。")
        saved_collection = metadata.get("collection", {})
        expected_collection = {"seconds_per_round": args.seconds, "rounds_per_map": args.rounds,
                               "step_seconds": step_seconds, "steps_per_round": max_steps,
                               "lidar_wait_sec": LIDAR_WAIT_SEC,
                               "max_consecutive_lidar_misses": MAX_CONSECUTIVE_LIDAR_MISSES}
        if any(saved_collection.get(key) != value for key, value in expected_collection.items()):
            raise ValueError("续采批次的采集参数（包括缺帧处理）与当前配置不一致；请新建批次。")
        saved_dropout = metadata.get("dropout", {})
        if (saved_dropout.get("fraction"), saved_dropout.get("seed")) != (DROPOUT_FRACTION, DROPOUT_SEED):
            raise ValueError("续采批次的随机置零参数与当前配置不一致。")
        completed_maps = metadata.get("maps", [])
        if len(completed_maps) > len(worlds) or any(
                Path(entry["world"]).resolve() != worlds[index].world
                for index, entry in enumerate(completed_maps)):
            raise ValueError("续采批次的已完成地图与当前地图列表不一致。")
        for entry in completed_maps:
            if entry.get("rounds") != args.rounds:
                raise ValueError("续采批次每张地图的轮数与当前配置不一致。")
            for key in ("raw_file", "timing_file"):
                path = resume_directory / entry[key]
                if not path.is_file() or count_rows(path) != entry["rows"]:
                    raise ValueError(f"续采批次文件缺失或行数不符：{path}")
            if not (resume_directory / entry["lidar_events_file"]).is_file():
                raise ValueError("续采批次缺少雷达事件日志。")
        pending_worlds = worlds[len(completed_maps):]
    gui = GUI and not args.headless
    if gui and (not math.isfinite(GUI_REAL_TIME_FACTOR) or GUI_REAL_TIME_FACTOR <= 0):
        raise ValueError("GUI_REAL_TIME_FACTOR 必须为正的有限数。")
    real_time_factor = GUI_REAL_TIME_FACTOR if gui else 0.0
    mode = (f"GUI 显示优先（最高 {real_time_factor:g} 倍速，关闭阴影）" if gui
            else "无 GUI，采集优先（不限速）")
    pending_names = ", ".join(w.world.name for w in pending_worlds) or "无（仅生成增强并发布）"
    resume_note = (f"续采批次：{resume_directory}（已完成 {len(completed_maps)} 张）\n"
                   if resume_directory is not None else "")
    print(f"{resume_note}教师：{model_path}\n地图：{pending_names}\n"
          f"运行模式：{mode}\n"
          f"出生扰动：位置 ±{spawn_config['position_jitter_m']:g} m，朝向 ±{spawn_config['yaw_jitter_deg']:g}°；"
          f"每图 {sum(s['kind'] == 'near_wall' for s in next(iter(spawn_plans.values()))['rounds'])} 轮近墙纠正\n"
          f"采集雷达：{vehicle.lidar_scan_hz:g} Hz，{vehicle.lidar_raw_samples} 原始点/圈，"
          f"输出 {vehicle.lidar_point_count} 桶（来源：{LIDAR_CONFIG}）\n"
          f"缺帧处理：最多等待 {LIDAR_WAIT_SEC:g} 现实秒，继续驾驶但不生成该行；"
          f"连续 {MAX_CONSECUTIVE_LIDAR_MISSES} 个决策点缺帧时停止\n"
          f"每图 {args.rounds} 轮，每轮最多 {max_steps * step_seconds:g} 仿真秒"
          f"（配置 {args.seconds:g} 秒，{max_steps} 步）；"
          f"随机置零 {DROPOUT_FRACTION:.0%}\n采集目录：{output_dir}\n训练副本：{dataset_dir}", flush=True)
    if args.check:
        print("配置检查通过；未启动采集。")
        return
    torch.set_num_threads(config.TORCH_THREADS)
    if resume_directory is not None:
        directory = resume_directory
        batch = metadata["batch"]
        metadata["status"] = "collecting"
        metadata.pop("error", None)
        metadata["resumed_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        filenames = [name for entry in completed_maps
                     for name in (entry["raw_file"], entry["timing_file"], entry["lidar_events_file"])]
        write_metadata(directory, metadata)
    else:
        batch = f"collection_{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
        directory = output_dir / batch
        directory.mkdir(parents=True)
        (directory / "raw").mkdir()
        (directory / "dropout").mkdir()
        metadata = {
            "schema_version": 4, "batch": batch, "status": "collecting",
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "source_directory": str(directory), "dataset_directory": str(dataset_dir / batch),
            "model": str(model_path), "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "model_training_timesteps": saved.get("training_timesteps"),
            "teacher_vehicle": asdict(teacher_vehicle), "lidar_config_source": str(LIDAR_CONFIG.resolve()),
            "observation_spec": spec, "vehicle": asdict(vehicle), "seed": SEED, "gui": gui,
            "spawn_randomization": spawn_config, "spawn_plans": spawn_plans,
            "collection": {"seconds_per_round": args.seconds, "rounds_per_map": args.rounds,
                           "real_time_factor": real_time_factor,
                           "step_seconds": step_seconds, "steps_per_round": max_steps,
                           "lidar_wait_sec": LIDAR_WAIT_SEC,
                           "max_consecutive_lidar_misses": MAX_CONSECUTIVE_LIDAR_MISSES,
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
        header = directory / "vehicle_config.h"
        header.write_text(collection_header, encoding="utf-8")
        for world in pending_worlds:
            stem = world.world.stem
            raw_name = f"raw/{stem}_raw.csv"
            dropout_name = f"dropout/{stem}_dropout.csv"
            timing_name = f"{stem}_timing.jsonl"
            events_name = f"{stem}_lidar_events.jsonl"
            map_rows = 0
            map_steps = map_skipped = missing_streak = 0
            map_counts = Counter()
            env = AicarGazeboEnv(
                role="eval", domain_id=ROS_DOMAIN, port=GAZEBO_PORT, header=header,
                world=world.world, spawn_pose=spawn_plans[stem]["rounds"][0]["pose"], seed=SEED,
                settings=replace(settings, workspace_bounds=world.workspace_bounds),
                reward_settings=config.RewardSettings(**saved["reward"]),
                log_dir=directory / f"runtime_{stem}",
                gui=gui, real_time_factor=real_time_factor,
                observation_mode="privileged", capture_lidar=True, lidar_wait_sec=LIDAR_WAIT_SEC)
            if gui:
                time.sleep(config.GUI_STARTUP_SETTLE_SEC)
            with (directory / raw_name).open("x", encoding="utf-8") as raw, \
                    (directory / timing_name).open("x", encoding="utf-8") as timing, \
                    (directory / events_name).open("x", encoding="utf-8") as events:
                for index in range(args.rounds):
                    number = len(metadata["episodes"]) + 1
                    scenario = spawn_plans[stem]["rounds"][index]
                    env.spawn_pose = tuple(scenario["pose"])
                    obs, current = env.reset(seed=SEED + number - 1)
                    print(f"{stem} 第 {index + 1}/{args.rounds} 轮开始：{scenario['kind']}，"
                          f"车体净空 {scenario['clearance_m']:.3f} m，"
                          f"前方净空 {scenario['front_clearance_m']:.3f} m。", flush=True)
                    round_started = time.monotonic()
                    counts = Counter()
                    previous_stamp = -1
                    row_start = map_rows + 1
                    skipped_rows = 0
                    while True:
                        stamp = current["lidar_stamp_ns"]
                        decision_ns = current["odom_stamp_ns"]  # exact teacher physics time
                        ready = current["lidar_ready"]
                        if ready and not previous_stamp < stamp <= decision_ns:
                            raise SimulationDataError("雷达时间戳重复、倒退或晚于动作决策。")
                        if not ready or missing_streak:
                            missing_streak = missing_streak + 1 if not ready else 0
                            events.write(json.dumps({
                                "event": "missing" if not ready else "recovered",
                                "round": index + 1, "decision_step": current["step"],
                                "decision_stamp_ns": decision_ns, "lidar_stamp_ns": stamp,
                                "consecutive_misses": missing_streak,
                            }) + "\n")
                            events.flush()
                            if not ready:
                                print(f"{stem} 第 {index + 1} 轮：决策点 {current['step']} 缺少新雷达，"
                                      f"连续 {missing_streak}/{MAX_CONSECUTIVE_LIDAR_MISSES} 次。", flush=True)
                            else:
                                print(f"{stem} 雷达恢复，继续记录。", flush=True)
                        if missing_streak >= MAX_CONSECUTIVE_LIDAR_MISSES:
                            raise SimulationDataError(
                                f"雷达连续 {missing_streak} 个决策点缺少新鲜帧，已停止采集并保留已有数据；"
                                f"查看 {directory / events_name}")
                        action, _ = model.predict(obs, deterministic=True)
                        action = int(action)
                        row = training_row(current["lidar_mm"].tolist(), action, vehicle) if ready else None
                        obs, _, terminated, truncated, following = env.step(action)
                        if following["action"] != action:
                            raise SimulationDataError("记录动作与执行动作不同。")
                        if row is not None:
                            raw.write(row)
                            map_rows += 1
                            timing.write(json.dumps({
                                "row": map_rows, "round": index + 1, "step": following["step"],
                                "lidar_stamp_ns": stamp,
                                "decision_stamp_ns": decision_ns, "action": action,
                                "action_end_stamp_ns": following["odom_stamp_ns"],
                            }) + "\n")
                            counts[action] += 1
                            previous_stamp = stamp
                        else:
                            skipped_rows += 1
                        current = following
                        if following["step"] % 100 == 0:
                            raw.flush()
                            timing.flush()
                            print(f"{stem} 第 {index + 1}/{args.rounds} 轮：{following['step']} 步，"
                                  f"{map_rows - row_start + 1} 条，缺帧未记录 {skipped_rows} 步，"
                                  f"{following['episode_sim_time_sec']:.1f} 仿真秒", flush=True)
                        if terminated or truncated:
                            break
                    raw.flush()
                    timing.flush()
                    wall_seconds = time.monotonic() - round_started
                    rows = map_rows - row_start + 1
                    map_steps += following["step"]
                    map_skipped += skipped_rows
                    map_counts.update(counts)
                    metadata["episodes"].append({
                        "group_id": f"{batch}/{stem}", "round": index + 1, "world": str(world.world),
                        "world_sha256": env.ground_truth_map.sha256,
                        "spawn_pose": env.spawn_pose, "settings": asdict(env.settings),
                        "spawn_scenario": scenario,
                        "raw_file": raw_name, "dropout_file": dropout_name, "timing_file": timing_name,
                        "row_start": row_start if rows else None, "row_end": map_rows if rows else None,
                        "rows": rows, "control_steps": following["step"], "skipped_rows": skipped_rows,
                        "action_counts": dict(sorted(counts.items())),
                        "termination_reason": following["termination_reason"],
                        "success": following["is_success"],
                        "sim_seconds": following["episode_sim_time_sec"],
                        "wall_seconds": wall_seconds,
                        "distance_m": following["episode_distance_m"],
                        "mean_speed_mps": following["episode_mean_speed_mps"],
                    })
                    write_metadata(directory, metadata)
                    print(f"{stem} 第 {index + 1}/{args.rounds} 轮结束："
                          f"{following['termination_reason']}，{following['step']} 步、{rows} 条，"
                          f"缺帧未记录 {skipped_rows} 步，"
                          f"本地图累计 {map_rows} 条，实际采集耗时 {wall_seconds:.1f} 秒"
                          f"（{rows / wall_seconds:.1f} 条/秒）。", flush=True)
            if map_rows == 0:
                raise SimulationDataError(f"{stem} 没有取得可记录的雷达帧，不发布空训练集。")
            metadata["maps"].append({
                "group_id": f"{batch}/{stem}", "world": str(world.world),
                "world_sha256": env.ground_truth_map.sha256, "rounds": args.rounds,
                "raw_file": raw_name, "dropout_file": dropout_name, "timing_file": timing_name,
                "lidar_events_file": events_name, "control_steps": map_steps, "skipped_rows": map_skipped,
                "rows": map_rows, "action_counts": dict(sorted(map_counts.items())),
            })
            filenames.extend([raw_name, timing_name, events_name])
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
