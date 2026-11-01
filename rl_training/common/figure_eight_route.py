"""Ordered figure-eight gates for training feedback, never policy inputs."""
import json
import math
from pathlib import Path

import numpy as np


class FigureEightRoute:
    def __init__(self, world, settings, speed):
        self.settings = settings
        if (not all(math.isfinite(x) and x > 0 for x in (speed, settings.route_crossing_radius_m,
                settings.route_timeout_factor, settings.route_progress_per_m, settings.route_switch_bonus))
                or not math.isfinite(settings.route_failure_penalty) or settings.route_failure_penalty >= 0):
            raise ValueError('路线距离/时间/奖励必须为正，失败惩罚必须为负。')
        meta = json.loads(Path(world).with_suffix('.json').read_text())
        p = meta['parameters']
        self.a, self.b = p['width_m']/2, p['height_m']/2
        self.radius = settings.route_crossing_radius_m
        self.far_y = .55*self.b
        t = np.linspace(0, 2*np.pi, 1001)
        xy = np.column_stack((self.a*np.sin(2*t), self.b*np.sin(t)))
        half_lap_seconds = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()/2/speed
        self.timeout = settings.route_timeout_factor*half_lap_seconds
        if not 0 < self.radius < self.far_y:
            raise ValueError('8 字交叉口检查范围必须小于远端检查区域距离。')

    @classmethod
    def for_world(cls, world, settings, speed):
        path = Path(world).with_suffix('.json')
        if not settings.figure_eight_route_enabled or not path.exists():
            return None
        if json.loads(path.read_text()).get('template') != 'figure_eight':
            return None
        return cls(world, settings, speed)

    def reset(self, position, yaw):
        self.previous = np.asarray(position, dtype=float)
        self.initial_heading = np.array([math.cos(yaw), math.sin(yaw)])
        side = np.sign(position[1]) if abs(position[1]) > .05 else -np.sign(math.sin(yaw))
        self.target_side = -int(side or 1)
        self.direction = None
        self.crossed = False
        self.switches = 0
        self.last_progress = 0.
        self.failed = False

    def update(self, seconds, position, *, final=False):
        p = np.asarray(position, dtype=float)
        reward = 0.
        reason = ''
        radius = float(np.linalg.norm(p))
        if self.direction is None and not self.crossed and radius <= self.radius:
            # The entrance determines the opposite branch, not the chosen yaw.
            directions = np.array([[2*self.a, self.target_side*self.b],
                                   [-2*self.a, self.target_side*self.b]])
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            approach = -self.previous if np.linalg.norm(self.previous) > .1 else self.initial_heading
            self.direction = directions[np.argmax(directions @ approach)]
        if self.direction is not None:
            reward += self.settings.route_progress_per_m*float((p-self.previous) @ self.direction)
            if radius >= self.radius + .10:
                if p[1]*self.target_side <= 0 or p[0]*self.direction[0] <= 0:
                    reason = 'route_wrong_exit'
                else:
                    self.crossed = True
                self.direction = None
        if self.crossed and p[1]*self.target_side >= self.far_y:
            self.switches += 1
            self.target_side *= -1
            self.crossed = False
            self.last_progress = seconds
            reward += self.settings.route_switch_bonus
        if seconds-self.last_progress > self.timeout and not reason:
            reason = 'route_no_progress'
        if final and self.switches == 0 and not reason:
            reason = 'route_incomplete'
        self.failed = bool(reason)
        if self.failed:
            reward = self.settings.route_failure_penalty
        self.previous = p.copy()
        return dict(route_reward=reward, route_failed=self.failed,
                    route_reason=reason, route_switches=self.switches,
                    route_target_side=self.target_side)
