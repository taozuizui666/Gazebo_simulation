"""World-state checks for rewards/evaluation only; never policy observations."""

from collections import deque
import math


class MotionMonitor:
    def __init__(self, settings, target_speed):
        self.settings = settings
        self.target_speed = target_speed
        self.history = deque()

    def reset(self, timestamp, position, yaw):
        self.history.clear()
        self.yaw = yaw
        self.turn = self.distance = self.forward_distance = 0.0
        self.speed_samples = self.in_speed_samples = 0
        self.history.append((timestamp, *position, self.turn, self.distance, self.forward_distance))

    def update(self, timestamp, position, yaw, linear_velocity, distance, dt):
        self.turn += abs(math.atan2(math.sin(yaw - self.yaw), math.cos(yaw - self.yaw)))
        self.yaw = yaw
        self.distance += distance
        self.forward_distance += max(0., linear_velocity) * dt
        self.speed_samples += 1
        self.in_speed_samples += abs(linear_velocity / self.target_speed - 1) <= self.settings.speed_tolerance_ratio
        self.history.append((timestamp, *position, self.turn, self.distance, self.forward_distance))
        cutoff = timestamp - self.settings.loop_window_sec
        while len(self.history) > 1 and self.history[1][0] <= cutoff:
            self.history.popleft()
        speed_start = self.history[0]
        for entry in self.history:
            if entry[0] <= timestamp - self.settings.stuck_window_sec + 1e-9:
                speed_start = entry
            else:
                break
        speed_elapsed = timestamp - speed_start[0]
        window_ready = speed_elapsed >= self.settings.stuck_window_sec - 1e-9
        mean_speed = (self.forward_distance - speed_start[5]) / max(speed_elapsed, 1e-9)
        stuck = window_ready and self.distance - speed_start[4] < self.settings.stuck_min_travel_m
        slow = window_ready and mean_speed < self.target_speed * self.settings.min_cruise_speed_ratio
        xs, ys = [p[1] for p in self.history], [p[2] for p in self.history]
        span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        turn = self.turn - self.history[0][3]
        looping = (timestamp - self.history[0][0] >= self.settings.loop_min_duration_sec - 1e-9
                   and turn >= self.settings.loop_min_turn_rad and span < self.settings.loop_max_span_m)
        outside = False
        if self.settings.workspace_bounds is not None:
            xmin, xmax, ymin, ymax = self.settings.workspace_bounds
            outside = not (xmin <= position[0] <= xmax and ymin <= position[1] <= ymax)
        return {
            "stuck": bool(stuck), "slow": bool(slow), "looping": bool(looping),
            "out_of_bounds": outside, "window_mean_speed_mps": mean_speed,
            "window_span_m": span, "window_turn_rad": turn,
            "speed_compliance": float(self.in_speed_samples / self.speed_samples),
        }
