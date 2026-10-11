"""One Gazebo server per worker, with episode-boundary map rotation."""
from dataclasses import replace
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from my_gazebo_launch.world_config import load_world_settings
from . import config
from .rl_env import AicarGazeboEnv, SimulationDataError
from ..privileged.observation import GroundTruthMap, PrivilegedSettings


def load_campaign(spec):
    groups = {role: [load_world_settings(p) for p in spec[role + '_worlds']]
              for role in ('train', 'eval')}
    for role, worlds in groups.items():
        if not worlds or len({w.world for w in worlds}) != len(worlds):
            raise ValueError(f'{role}_worlds 必须非空且不重复。')
    if len({w.world.name for worlds in groups.values() for w in worlds}) != sum(map(len, groups.values())):
        raise ValueError('所有地图文件名必须唯一，以便逐地图统计。')
    if {w.world for w in groups['train']} & {w.world for w in groups['eval']}:
        raise ValueError('训练地图和评估地图必须分开。')
    for key in ('episodes_per_map', 'eval_episodes_per_map', 'spawn_attempts'):
        if not isinstance(spec[key], int) or spec[key] < 1:
            raise ValueError(f'{key} 必须是正整数。')
    for key in ('train_jitter_m', 'train_jitter_deg', 'eval_jitter_m', 'eval_jitter_deg', 'spawn_margin_m'):
        if not math.isfinite(spec[key]) or spec[key] < 0:
            raise ValueError(f'{key} 必须是非负有限数。')
    for role, worlds in groups.items():
        for world in worlds:
            geometry = GroundTruthMap(world.world)
            poses = world.train_spawn_poses if role == 'train' else world.eval_spawn_poses
            for pose in poses:
                if not safe_spawn(world, geometry, pose, spec['spawn_margin_m']):
                    raise ValueError(f'{world.world.name} 的出生点 {pose} 没有足够车体净空。')
    return groups


def safe_spawn(world, geometry, pose, margin):
    radius = geometry.settings.body_clearance_radius_m + margin
    x, y = pose[:2]
    if world.workspace_bounds:
        xmin, xmax, ymin, ymax = world.workspace_bounds
        if not (xmin + radius < x < xmax - radius and ymin + radius < y < ymax - radius):
            return False
    return float(geometry.signed_distance(np.array([[x, y]]))[0]) > radius


class MultiMapEnv(gym.Env):
    """Training randomizes starts; evaluation replays a fixed, balanced suite."""
    def __init__(self, *, spec, rank, role, settings, **kwargs):
        self.campaign, self.rank, self.role = spec, rank, role
        self.worlds = [load_world_settings(p) for p in spec[role + '_worlds']]
        self.geometry = [GroundTruthMap(w.world) for w in self.worlds]
        self.settings, self.kwargs = settings, kwargs
        self.seed_value = kwargs['seed']
        self.rng = np.random.default_rng(self.seed_value)
        self.episode_index = 0
        self.env = None
        self.map_index = None
        self.records = []
        self.current = {}
        self.observation_space = spaces.Box(-1., 1., (PrivilegedSettings().dimension,), np.float32)
        self.action_space = spaces.Discrete(config.vehicle_config(kwargs['header']).label_count)

    def reset_sequence(self):
        self.episode_index = 0
        self.rng = np.random.default_rng(self.seed_value)
        self.records = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        train = self.role == 'train'
        period = self.campaign['episodes_per_map' if train else 'eval_episodes_per_map']
        index = ((self.rank if train else 0) + self.episode_index // period) % len(self.worlds)
        world, geometry = self.worlds[index], self.geometry[index]
        poses = world.train_spawn_poses if train else world.eval_spawn_poses
        spawn_index = int(self.rng.integers(len(poses))) if train else self.episode_index % period % len(poses)
        base = np.array(poses[spawn_index], dtype=float)
        jitter_m = self.campaign[self.role + '_jitter_m']
        jitter_rad = math.radians(self.campaign[self.role + '_jitter_deg'])
        if self.map_index != index:
            self.close()
            self.map_index = index
        for _ in range(self.campaign['spawn_attempts']):
            pose = base.copy()
            pose[:2] += self.rng.uniform(-jitter_m, jitter_m, 2)
            pose[3] += self.rng.uniform(-jitter_rad, jitter_rad)
            if not safe_spawn(world, geometry, pose, self.campaign['spawn_margin_m']):
                continue
            if self.env is None:
                self.env = AicarGazeboEnv(role=self.role, world=str(world.world),
                    spawn_pose=tuple(pose), settings=replace(self.settings, workspace_bounds=world.workspace_bounds),
                    **self.kwargs)
            self.env.spawn_pose = tuple(pose)
            try:
                observation, info = self.env.reset(seed=seed, options=options)
            except SimulationDataError as error:
                # The inner environment closes itself after a failed reset.
                self.env = None
                if '出生点与障碍物相碰' not in str(error):
                    raise
                continue
            self.episode_index += 1
            self.current = dict(world=world.world.name, spawn_index=spawn_index,
                                spawn_pose=pose.tolist())
            self.env.progress.update(force=True, episode_number=self.episode_index, **self.current)
            return observation, {**info, **self.current}
        raise ValueError(f'{world.world.name} 出生点 {spawn_index} 扰动重试耗尽；请调整出生点/扰动/净空。')

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info.update(self.current)
        if self.role == 'eval' and (terminated or truncated):
            self.records.append({key: info[key] for key in (
                'world', 'spawn_index', 'spawn_pose', 'termination_reason',
                'episode_distance_m', 'episode_sim_time_sec', 'episode_mean_speed_mps',
                'speed_compliance', 'is_success')})
        return observation, reward, terminated, truncated, info

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
