"""Run held-out, fixed-weight acceptance after training and promote only on pass."""

from argparse import Namespace
import json
import math
from pathlib import Path

import numpy as np

from my_gazebo_launch.world_config import load_world_settings
from rl_training.common import config
from rl_training.common.evaluation import evaluate_model
from rl_training.common.multi_map import safe_spawn
from rl_training.common.progress_state import atomic_json, utc_now
from rl_training.common.qualify_model import qualify
from .observation import GroundTruthMap


def validation_worlds(args):
    settings = args.validation
    if not settings['enabled']:
        return []
    if settings['model_name'] not in ('best_model.zip', 'rl_policy_final.zip'):
        raise ValueError('VALIDATION_MODEL_NAME 必须是 best_model.zip 或 rl_policy_final.zip。')
    episodes, scenarios = settings['episodes_per_map'], settings['min_scenarios']
    if (not isinstance(episodes, int) or not isinstance(scenarios, int)
            or not 1 <= scenarios <= episodes
            or not math.isfinite(settings['episode_seconds']) or settings['episode_seconds'] <= 0
            or not math.isfinite(settings['min_speed_compliance'])
            or not 0 < settings['min_speed_compliance'] <= 1):
        raise ValueError('验收回合数、起点数、时长或速度合规率配置无效。')
    if any(not math.isfinite(settings[k]) or settings[k] < 0 for k in ('jitter_m', 'jitter_deg')):
        raise ValueError('验收出生点扰动必须为非负有限数。')
    worlds = [load_world_settings(p) for p in settings['worlds']]
    if not worlds or len({w.world for w in worlds}) != len(worlds):
        raise ValueError('VALIDATION_WORLDS 必须非空且不重复。')
    used = {Path(p).resolve() for p in args.multi_map['train_worlds'] + args.multi_map['eval_worlds']}
    used_hashes = {GroundTruthMap(p).sha256 for p in used}
    for index, world in enumerate(worlds):
        geometry = GroundTruthMap(world.world)
        if world.world in used or geometry.sha256 in used_hashes:
            raise ValueError(f'{world.world.name} 已用于训练或选模，不能兼作独立验收地图。')
        poses = world.eval_spawn_poses
        if len(set(poses[:episodes])) < scenarios:
            raise ValueError(f'{world.world.name} 的实际验收出生点数少于 VALIDATION_MIN_SCENARIOS。')
        # Replay evaluation's exact sampling order so unsafe jitter fails before training.
        random = np.random.default_rng(settings['seed'] + index)
        for episode in range(episodes):
            x, y, z, yaw = poses[episode % len(poses)]
            pose = (x + random.uniform(-settings['jitter_m'], settings['jitter_m']),
                    y + random.uniform(-settings['jitter_m'], settings['jitter_m']), z,
                    yaw + math.radians(random.uniform(-settings['jitter_deg'], settings['jitter_deg'])))
            if not safe_spawn(world, geometry, pose, args.multi_map['spawn_margin_m']):
                raise ValueError(f'{world.world.name} 第 {episode+1} 个验收起点净空不足，请调整扰动/起点。')
    return worlds


def run_validation(args, directory):
    settings = args.validation
    if not settings['enabled']:
        return
    directory = Path(directory)
    metadata = json.loads((directory / 'run_config.json').read_text())
    state = dict(status='starting', started_at=utc_now(), settings=settings, reports=[])
    destination = directory / 'auto_validation.json'
    if metadata['status'] not in ('complete', 'time_budget_reached'):
        state.update(status='skipped', reason='训练被中断或主动停止，不自动启动验收。')
        atomic_json(destination, state)
        print(state['reason'], flush=True)
        return
    model = directory / settings['model_name']
    if not model.is_file() and settings['model_name'] == 'best_model.zip':
        model = directory / 'rl_policy_final.zip'
        print('本次尚未产生 best_model.zip，验收最终模型。', flush=True)
    state['model'] = str(model)
    atomic_json(destination, state)
    try:
        worlds = validation_worlds(args)
        vehicle = config.vehicle_config(directory / 'vehicle_config.h')
        step = math.ceil(metadata['environment']['step_sim_sec'] / vehicle.sim_time_step_sec) * vehicle.sim_time_step_sec
        for index, world in enumerate(worlds):
            state.update(status='running', world=str(world.world))
            atomic_json(destination, state)
            evaluation = Namespace(evaluate=model, config_header=directory / 'vehicle_config.h',
                world=str(world.world), eval_spawns=world.eval_spawn_poses,
                workspace_bounds=world.workspace_bounds, output_dir=args.output_dir,
                eval_domain=args.eval_domain, eval_port=args.eval_port, gui=False,
                episode_steps=math.ceil(settings['episode_seconds'] / step),
                eval_episodes=settings['episodes_per_map'], seed=settings['seed'] + index,
                eval_jitter_m=settings['jitter_m'], eval_jitter_deg=settings['jitter_deg'],
                min_speed_compliance=settings['min_speed_compliance'])
            _, report_dir = evaluate_model(evaluation)
            state['reports'].append(str(report_dir / 'evaluation.json'))
            atomic_json(destination, state)
        try:
            qualified = qualify(model, [Path(p) for p in state['reports']], output_dir=args.output_dir,
                min_episodes=settings['episodes_per_map'], min_scenarios=settings['min_scenarios'],
                min_episode_sim_seconds=settings['episode_seconds'],
                min_speed_compliance=settings['min_speed_compliance'])
        except ValueError as error:
            state.update(status='rejected', reason=str(error))
            print(f'验收未通过，保留原 latest_validated：{error}', flush=True)
        else:
            state.update(status='passed', validated_directory=str(qualified))
            print(f'自动验收通过，latest_validated 已更新：{qualified}', flush=True)
    except KeyboardInterrupt:
        state.update(status='interrupted', reason='用户中断验收；已保存的训练模型保留。')
        print(state['reason'], flush=True)
    except Exception as error:
        state.update(status='failed', reason=f'{type(error).__name__}: {error}')
        print(f'自动验收未完成，训练模型已保存：{error}', flush=True)
    finally:
        state['ended_at'] = utc_now()
        atomic_json(destination, state)
