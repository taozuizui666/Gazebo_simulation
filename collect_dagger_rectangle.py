#!/usr/bin/env python3
"""用 RF 行驶到偏离状态，由冻结教师给动作标签，采集矩形纠偏示范。"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from my_gazebo_launch.config import load_config
from my_gazebo_launch.lidar_processing import training_row
from my_gazebo_launch.world_config import load_world_settings
from rl_training.common import config
from rl_training.common.model_io import load_model
from rl_training.common.rl_env import AicarGazeboEnv
from run_random_forest import load_model as load_rf

RF_MODEL = ROOT.parent / 'software/ML_model/five_map_rf_targeted_cap2/randomForest.joblib'
TEACHER = ROOT / 'rl_models/privileged/policies/validated_privileged_2764876_20261101_201259/model.zip'
CONFIG_HEADER = ROOT / 'my_gazebo_launch/vehicle_config.h'
STARTS = ROOT / 'results/targeted_collection_inputs_20261101/targeted_starts.json'
OUTPUT = ROOT / 'results/datasets_dagger_20261102'
DATASET = ROOT.parent / 'dataset/simulation_five_20261101/dagger_rectangle'
DOMAIN = 97
PORT = 11557
SECONDS = 300.0
SEED = 42
DROPOUT = .05


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--rf-model', type=Path, default=RF_MODEL)
    parser.add_argument('--teacher', type=Path, default=TEACHER)
    parser.add_argument('--world', default='assets/worlds/simple/rectangle_001.sdf')
    parser.add_argument('--starts', type=Path, default=STARTS)
    parser.add_argument('--seconds', type=float, default=SECONDS)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--dataset-dir', type=Path, default=DATASET)
    args = parser.parse_args()
    vehicle = load_config(CONFIG_HEADER)
    rf, indices = load_rf(args.rf_model.expanduser().resolve(), vehicle)
    teacher = load_model(args.teacher.expanduser().resolve(), device=config.DEVICE)
    starts = json.loads(args.starts.expanduser().read_text())['rectangle_001']
    world = load_world_settings(args.world)
    settings = replace(config.EnvironmentSettings(), max_steps=math.ceil(args.seconds / 0.2),
                       workspace_bounds=world.workspace_bounds)
    batch = f'collection_{datetime.now():%Y%m%d_%H%M%S}_dagger'
    directory = args.output_dir / batch
    (directory / 'raw').mkdir(parents=True)
    (directory / 'dropout').mkdir()
    raw_path = directory / 'raw' / f'{world.world.stem}_dagger_raw.csv'
    drop_path = directory / 'dropout' / f'{world.world.stem}_dagger_dropout.csv'
    rng = random.Random(SEED)
    episodes = []
    env = None
    try:
        env = AicarGazeboEnv(role='eval', domain_id=DOMAIN, port=PORT, header=CONFIG_HEADER,
                             world=world.world, spawn_pose=starts[0], seed=SEED,
                             settings=settings, gui=False, observation_mode='privileged',
                             capture_lidar=True, lidar_wait_sec=.2)
        with raw_path.open('w', newline='') as raw_file, drop_path.open('w', newline='') as drop_file:
            rows = 0
            for index, pose in enumerate(starts):
                start_rows = rows
                env.spawn_pose = tuple(pose)
                obs, info = env.reset(seed=SEED + index)
                step = 0
                while True:
                    rf_action = int(rf.predict(info['lidar_mm'][indices].reshape(1, -1))[0])
                    teacher_action = int(teacher.predict(obs, deterministic=True)[0])
                    distances = list(map(int, info['lidar_mm']))
                    raw_file.write(training_row(distances, teacher_action, vehicle))
                    masked = distances[:]
                    nonzero = [i for i, value in enumerate(masked) if value > 0]
                    count = int(len(nonzero) * DROPOUT + .5)
                    for i in rng.sample(nonzero, count):
                        masked[i] = 0
                    drop_file.write(training_row(masked, teacher_action, vehicle))
                    rows += 1
                    obs, reward, terminated, truncated, info = env.step(rf_action)
                    step += 1
                    if terminated or truncated:
                        break
                episodes.append({'round': index + 1, 'spawn': pose, 'rows': rows - start_rows,
                                 'steps': step, 'reason': info['termination_reason'],
                                 'distance_m': info['episode_distance_m'],
                                 'sim_seconds': info['episode_sim_time_sec']})
                print(f"{index + 1}/{len(starts)} {info['termination_reason']} "
                      f"{info['episode_distance_m']:.2f}m rows={rows}", flush=True)
    finally:
        if env is not None:
            env.close()
    metadata = {'batch': batch, 'status': 'complete', 'rf_model': str(args.rf_model),
                'rf_sha256': hashlib.sha256(args.rf_model.read_bytes()).hexdigest(),
                'teacher': str(args.teacher), 'teacher_sha256': hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
                'world': str(world.world), 'world_sha256': hashlib.sha256(world.world.read_bytes()).hexdigest(),
                'rows': rows, 'episodes': episodes, 'dropout_fraction': DROPOUT,
                'format': {'columns': vehicle.lidar_point_count + 1, 'last_column': 'action'}}
    (directory / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    destination = args.dataset_dir / 'raw'
    destination.mkdir(exist_ok=True)
    for source, target in ((raw_path, destination / raw_path.name), (drop_path, args.dataset_dir / 'dropout' / drop_path.name)):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    (args.dataset_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print('done', directory, 'rows', rows, flush=True)


if __name__ == '__main__':
    main()
