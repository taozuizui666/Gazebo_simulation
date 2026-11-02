#!/usr/bin/env python3
"""冻结 RF 的五图/定向闭环验收：有符号净空、路线覆盖和 8 字换环。"""
from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from generate_maps import centerline
from audit_teacher_suite import track_metrics
from rl_training.common.figure_eight_route import FigureEightRoute
from my_gazebo_launch.config import load_config
from my_gazebo_launch.world_config import load_world_settings
from rl_training.common import config
from rl_training.common.multi_map import safe_spawn
from rl_training.common.rl_env import AicarGazeboEnv
from rl_training.privileged.observation import GroundTruthMap
from run_random_forest import load_model

MAPS = tuple(f'assets/worlds/simple/{name}.sdf' for name in (
    'rectangle_001', 'rounded_rectangle_003', 'ellipse_004', 'figure_eight_005'))
S_MAP = 'assets/worlds/simple_s_v2/s_bend_001.sdf'
DEV_MAPS = ('assets/worlds/simple/figure_eight_005.sdf', 'assets/worlds/simple/rectangle_001.sdf')
MODEL = ROOT.parent / 'software/ML_model/five_map_rf_validated_20261101/randomForest.joblib'
CONFIG_HEADER = ROOT / 'my_gazebo_launch/vehicle_config.h'
OUTPUT_ROOT = ROOT / 'results/random_forest_acceptance'
DOMAIN = 95
PORT = 11555
EPISODES = 8
SECONDS = 300.0
JITTER_M = 0.03
JITTER_DEG = 10.0
FORMAL_SEEDS = (20261110, 20261111)
CLEARANCE_M = 0.01
COVERAGE_BINS = 100
MIN_COVERAGE_BINS = 85  # 与冻结教师协议一致；报告仍保存实际覆盖
MIN_SPEED_COMPLIANCE = .95
MIN_ROUTE_SWITCHES = 3
ROUTE_SETTINGS = config.RewardSettings(figure_eight_route_enabled=True)


def centerline_geometry(world_path):
    settings = json.loads(Path(world_path).with_suffix('.json').read_text())
    p = settings['parameters']
    if 'reference_centerline' in settings:
        pts = np.asarray(settings['reference_centerline'], dtype=float)
    else:
        if settings['template'] == 's_bend':
            raise ValueError('S 弯必须使用带 reference_centerline 的真正 S 弯地图。')
        pts = np.array([q for q, _ in centerline(settings['template'], p['width_m'],
                                               p['height_m'], p.get('corner_radius_m'))])
    if (pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3
            or not np.isfinite(pts).all()
            or (np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1) <= 0).any()):
        raise ValueError('中心线必须包含至少三个有限点，且无零长度线段。')
    return settings, pts


def route_metrics(world_path, trace, geometry, speed):
    """复用教师的插值几何审计和路线门点；失败后不让后续通过覆盖首次失败。"""
    trace = np.asarray(trace, dtype=float)
    if len(trace) < 2 or not np.isfinite(trace).all():
        raise ValueError('轨迹不完整或含非有限值。')
    meta, pts = centerline_geometry(world_path)
    metrics = track_metrics(trace, pts, geometry)
    route = FigureEightRoute.for_world(world_path, ROUTE_SETTINGS, speed)
    failure, failure_time, switches = '', None, 0
    if route is not None:
        route.reset(trace[0, 1:3], trace[0, 3])
        for index, row in enumerate(trace[1:], 1):
            info = route.update(float(row[0]), row[1:3], final=index == len(trace)-1)
            if info['route_failed']:
                failure, failure_time = info['route_reason'], float(row[0])
                break
        switches = route.switches
        elements_ok = not failure and switches >= MIN_ROUTE_SWITCHES
    else:
        elements_ok = abs(metrics['net_laps']) >= 1
        if meta['template'] == 'rectangle':
            p = meta['parameters']
            corners = np.array([[sx*p['width_m']/2, sy*p['height_m']/2]
                                for sx in (-1, 1) for sy in (-1, 1)])
            elements_ok = elements_ok and bool((np.linalg.norm(
                trace[:, None, 1:3]-corners, axis=2).min(axis=0) < .4).all())
    return dict(**metrics, route_switches=switches, route_failure=failure,
                route_failure_time_sec=failure_time,
                route_qualified=bool(metrics['coverage_bins'] >= MIN_COVERAGE_BINS and elements_ok))


def qualification(metrics, info, reached_time_limit):
    duration_ok = info['episode_sim_time_sec'] >= SECONDS - 1e-6
    speed_ok = info['speed_compliance'] >= MIN_SPEED_COMPLIANCE
    clearance_ok = metrics['min_clearance_m'] >= CLEARANCE_M
    return dict(duration_qualified=bool(duration_ok), speed_qualified=bool(speed_ok),
                clearance_qualified=bool(clearance_ok),
                fully_qualified=bool(reached_time_limit and duration_ok and speed_ok
                                     and clearance_ok and metrics['route_qualified']))


def acceptance_exit_code(episodes, expected):
    return 0 if expected > 0 and len(episodes) == expected and all(
        e['fully_qualified'] for e in episodes) else 1


def signed_clearance(geometry, x, y, yaw):
    # 与既有教师净空指标一致：车体中心的有符号距离再减去保守车体半径；
    # 前方射线只用于 front clearance，不把整条射线算进车体净空。
    return float(geometry.signed_distance(np.array([[x, y]]))[0]
                 - geometry.settings.body_clearance_radius_m)


def starts_for(world, geometry, seed, count):
    poses = []
    rng = np.random.default_rng(seed)
    for index in range(count):
        base = world.eval_spawn_poses[index % len(world.eval_spawn_poses)]
        for _ in range(200):
            pose = (base[0] + rng.uniform(-JITTER_M, JITTER_M),
                    base[1] + rng.uniform(-JITTER_M, JITTER_M), base[2],
                    base[3] + math.radians(rng.uniform(-JITTER_DEG, JITTER_DEG)))
            if safe_spawn(world, geometry, pose, .01):
                poses.append(pose)
                break
        else:
            raise ValueError(f'{world.world.stem} 起点 {index} 无法生成安全扰动')
    return poses


def evaluate_map(map_path, model, classifier, indices, vehicle, seeds, output, development):
    world = load_world_settings(map_path)
    geometry = GroundTruthMap(world.world)
    stem = world.world.stem
    settings = replace(config.EnvironmentSettings(), max_steps=math.ceil(SECONDS / 0.2),
                       workspace_bounds=world.workspace_bounds)
    map_dir = output / stem
    server_dir = map_dir / '_server'
    server_dir.mkdir(parents=True)
    episodes = []
    env = None
    try:
        for seed in seeds:
            poses = (world.eval_spawn_poses if development else starts_for(world, geometry, seed, EPISODES))
            for index, pose in enumerate(poses):
                run_dir = map_dir / f'seed_{seed}' / f'episode_{index + 1:03d}'
                run_dir.mkdir(parents=True)
                if env is None:
                    env = AicarGazeboEnv(role='eval', domain_id=DOMAIN, port=PORT,
                                         header=CONFIG_HEADER, world=world.world, spawn_pose=pose,
                                         seed=seed, log_dir=server_dir, settings=settings, gui=False,
                                         observation_mode='lidar', capture_lidar=True)
                env.spawn_pose = pose
                obs, info = env.reset(seed=seed)
                x, y = info['position']
                yaw = info['yaw']
                signed = signed_clearance(geometry, x, y, yaw)
                trace = [[0.0, x, y, yaw, 0.0, vehicle.straight_label, signed]]
                clearances = [signed]
                latencies = []
                switches = 0
                previous = None
                while True:
                    started = time.perf_counter()
                    action = int(classifier.predict(info['lidar_mm'][indices].reshape(1, -1))[0])
                    latencies.append((time.perf_counter() - started) * 1000)
                    switches += int(previous is not None and action != previous)
                    previous = action
                    obs, reward, terminated, truncated, info = env.step(action)
                    x, y = info['position']
                    yaw = info['yaw']
                    signed = signed_clearance(geometry, x, y, yaw)
                    clearances.append(signed)
                    trace.append([info['episode_sim_time_sec'], x, y, yaw,
                                  info.get('linear_velocity', 0.0), action, signed])
                    if terminated or truncated:
                        break
                np.save(run_dir / 'trajectory.npy', np.asarray(trace))
                metrics = route_metrics(world.world, trace, geometry, vehicle.forward_speed_mps)
                coverage, lobe_switches = metrics['coverage_bins'], metrics['route_switches']
                no_collision = bool(truncated and not terminated and info['termination_reason'] == 'time_limit')
                gates = qualification(metrics, info, no_collision)
                entry = {
                    'world': stem, 'seed': seed, 'episode': index + 1, 'spawn': list(pose),
                    'reason': info['termination_reason'], 'no_collision_to_time_limit': no_collision,
                    'distance_m': info['episode_distance_m'], 'sim_duration_sec': info['episode_sim_time_sec'],
                    'mean_speed_mps': info['episode_mean_speed_mps'], 'speed_compliance': info['speed_compliance'],
                    'min_signed_clearance_m': metrics['min_clearance_m'],
                    'sampled_min_signed_clearance_m': min(clearances),
                    'coverage_bins': coverage, 'net_laps': metrics['net_laps'],
                    'route_switches': lobe_switches, 'lobe_switches': lobe_switches,
                    'route_failure': metrics['route_failure'],
                    'route_failure_time_sec': metrics['route_failure_time_sec'],
                    'route_qualified': metrics['route_qualified'], **gates,
                    'action_switches': switches, 'inference_p95_ms': float(np.percentile(latencies, 95)),
                }
                episodes.append(entry)
                (run_dir / 'episode.json').write_text(json.dumps(entry, indent=2) + '\n')
                print(f"{stem} seed={seed} ep={index + 1}: {entry['reason']} "
                      f"no_collision={no_collision} coverage={coverage} switches={lobe_switches} "
                      f"signed_clear={metrics['min_clearance_m']:.3f} qualified={entry['fully_qualified']}", flush=True)
    finally:
        if env is not None:
            env.close()
    cells = []
    for seed in seeds:
        sub = [e for e in episodes if e['seed'] == seed]
        cells.append({
            'world': stem, 'seed': seed, 'episodes': len(sub),
            'no_collision_to_time_limit': sum(e['no_collision_to_time_limit'] for e in sub),
            'clearance_qualified': sum(e['clearance_qualified'] for e in sub),
            'route_qualified': sum(e['route_qualified'] for e in sub),
            'fully_qualified': sum(e['fully_qualified'] for e in sub),
            'collisions': sum(e['reason'] == 'collision' for e in sub),
            'min_signed_clearance_m': min(e['min_signed_clearance_m'] for e in sub),
            'coverage_min': min(e['coverage_bins'] for e in sub),
            'lobe_switches_min': min(e['lobe_switches'] for e in sub),
        })
    return episodes, cells


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--mode', choices=('development', 'formal'), default='development')
    parser.add_argument('--maps', nargs='+', default=None)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    model_path = args.model.expanduser().resolve(strict=True)
    vehicle = load_config(CONFIG_HEADER)
    classifier, indices = load_model(model_path, vehicle)
    map_paths = args.maps or (DEV_MAPS if args.mode == 'development' else MAPS + (S_MAP,))
    output = args.output_dir / f'{args.mode}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    seeds = (args.seed,) if args.seed is not None else ((0,) if args.mode == 'development' else FORMAL_SEEDS)
    print(f'model={model_path} mode={args.mode} maps={map_paths} seeds={seeds}', flush=True)
    # 与实际评估使用同一地图解析、中心线、路线门点及起点检查。
    worlds = [load_world_settings(path) for path in map_paths]
    if len({str(w.world) for w in worlds}) != len(worlds) or len({w.world.stem for w in worlds}) != len(worlds):
        raise ValueError('地图路径及名称不能重复。')
    for world in worlds:
        geometry = GroundTruthMap(world.world)
        centerline_geometry(world.world)
        FigureEightRoute.for_world(world.world, ROUTE_SETTINGS, vehicle.forward_speed_mps)
        if len(world.eval_spawn_poses) != EPISODES:
            raise ValueError(f'{world.world}: 必须配置 {EPISODES} 个评估起点。')
        if args.mode == 'formal':
            for seed in seeds:
                starts_for(world, geometry, seed, EPISODES)
        print('ok', world.world, flush=True)
    if args.check:
        return 0
    output.mkdir(parents=True)
    all_episodes, all_cells = [], []
    for path in map_paths:
        print('===', path, flush=True)
        episodes, cells = evaluate_map(path, model_path, classifier, indices, vehicle,
                                       seeds, output, args.mode == 'development')
        all_episodes.extend(episodes)
        all_cells.extend(cells)
    expected = len(worlds) * len(seeds) * EPISODES
    exit_code = acceptance_exit_code(all_episodes, expected)
    report = {'passed': exit_code == 0, 'expected_episodes': expected,
              'vehicle': asdict(vehicle), 'route_settings': asdict(ROUTE_SETTINGS),
              'worlds': [{'path': str(w.world),
                          'sha256': hashlib.sha256(w.world.read_bytes()).hexdigest(),
                          'metadata_sha256': hashlib.sha256(w.world.with_suffix('.json').read_bytes()).hexdigest()}
                         for w in worlds],
              'model': str(model_path), 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'mode': args.mode, 'protocol': {'episodes_per_map': EPISODES, 'seconds': SECONDS,
              'jitter_m': 0 if args.mode == 'development' else JITTER_M,
              'jitter_deg': 0 if args.mode == 'development' else JITTER_DEG,
              'seeds': list(seeds), 'clearance_m': CLEARANCE_M, 'coverage_bins': COVERAGE_BINS,
              'minimum_coverage_bins': MIN_COVERAGE_BINS,
              'minimum_speed_compliance': MIN_SPEED_COMPLIANCE,
              'figure8_min_switches': MIN_ROUTE_SWITCHES,
              'clearance_interpolation_max_m': .01},
              'cells': all_cells, 'episodes': all_episodes}
    (output / 'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('report', output / 'acceptance.json', 'passed=', exit_code == 0, flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
