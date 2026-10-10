"""持续避障巡航奖励；接触信息用于奖励，不进入雷达观测。"""

import math
import numpy as np

from .config import RewardSettings


def compute_reward(lidar_mm, *, linear_velocity, distance_moved, sim_dt,
                   action, previous_action, vehicle, collided=False, stuck=False,
                   looping=False, slow=False, out_of_bounds=False, settings=None):
    """Return total reward and individual terms for auditable local logs."""
    settings = settings or RewardSettings()
    terms = {
        "travel": min(distance_moved, max(0., linear_velocity) * sim_dt,
                      vehicle.forward_speed_mps * sim_dt * 1.2) * settings.distance_per_m,
        "speed": -abs(linear_velocity - vehicle.forward_speed_mps)
                 * settings.speed_error_per_sec * sim_dt,
        "clearance": 0.0,
        "steering_change": 0.0,
        "turn": -settings.turn_per_sec * sim_dt
                * ((action - vehicle.straight_label) / vehicle.straight_label) ** 2,
        "front_risk": 0.0,
        "collision": settings.collision if collided else 0.0,
        "stuck": settings.stuck if stuck and not collided else 0.0,
        "loop": settings.loop if looping and not (collided or stuck) else 0.0,
        "slow": settings.slow if slow and not (collided or stuck or looping) else 0.0,
        "out_of_bounds": settings.out_of_bounds if out_of_bounds
                         and not (collided or stuck or looping or slow) else 0.0,
    }
    distances = np.asarray(lidar_mm)
    valid = distances[distances > 0]
    if valid.size:
        proximity = max(0.0, 1.0 - float(valid.min()) / settings.near_distance_mm)
        terms["clearance"] = -settings.near_penalty_per_sec * proximity * sim_dt
    angles = np.arange(len(distances)) * 360 / len(distances)
    front = distances[(np.minimum(angles, 360 - angles) <= settings.front_half_angle_deg)
                      & (distances > 0)]
    if front.size:
        risk = max(0., 1 - float(front.min()) / settings.front_distance_mm)
        terms["front_risk"] = -settings.front_risk_per_sec * risk ** 2 * sim_dt
    if previous_action is not None:
        change = abs(action - previous_action) / (vehicle.label_count - 1)
        terms["steering_change"] = -settings.steering_change_penalty * change
    if collided or stuck or looping or slow or out_of_bounds:
        # Failure must not gain a positive travel bonus from an impact/teleport.
        terms["travel"] = 0.0
    total = float(sum(terms.values()))
    if not math.isfinite(total):
        raise ValueError("奖励出现非有限值。")
    return total, terms
