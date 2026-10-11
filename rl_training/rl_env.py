"""Gymnasium continuous cruising with isolated, paused Gazebo stepping."""

from collections import deque
from contextlib import suppress
import math
import os
from pathlib import Path
import threading
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.clock import Clock as RosClock
from rclpy.clock import ClockType
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import GetModelList, SetEntityState
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray, Int32
from std_srvs.srv import Empty

from my_gazebo_launch.config import (
    CONTACT_TOPIC, DEFAULT_ENTITY_NAME, LABEL_TOPIC, RAW_SCAN_TOPIC, WHEEL_TARGET_TOPIC,
)
from my_gazebo_launch.control import wheel_targets
from my_gazebo_launch.lidar_processing import scan_to_distances
from . import config
from .reward import compute_reward
from .motion import MotionMonitor
from .simulator import SimulationProcess
from .progress_state import LiveState, utc_now
from .privileged import GroundTruthMap


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class SimulationDataError(RuntimeError):
    """Sensor/service failure: stop training instead of inventing a transition."""


class AicarGazeboNode(Node):
    def __init__(self, context, vehicle, settings, name, enable_lidar=True):
        super().__init__(name, context=context, use_global_arguments=False,
                         parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.vehicle, self.settings = vehicle, settings
        self.changed = threading.Condition()
        self.sim_ns = None
        self.scans = deque(maxlen=32)
        self.odometry = deque(maxlen=1024)
        self.contacts_ns = {}
        self.contacts_current = {}
        self.impacts = deque(maxlen=256)
        self.latest_wheels = None
        self.wheel_sequence = 0
        self.fault = None
        self.command = vehicle.stop_command
        self.action_publisher = self.create_publisher(Int32, LABEL_TOPIC, 10)
        self.sensor_subscriptions = [
            self.create_subscription(Clock, "/clock", self.receive_clock, qos_profile_sensor_data),
            self.create_subscription(Odometry, "/odom", self.receive_odom, qos_profile_sensor_data),
            self.create_subscription(ContactsState, CONTACT_TOPIC, self.receive_contacts, qos_profile_sensor_data),
            self.create_subscription(Float64MultiArray, WHEEL_TARGET_TOPIC, self.receive_wheels, 10),
        ]
        if enable_lidar:
            self.sensor_subscriptions.append(self.create_subscription(
                LaserScan, RAW_SCAN_TOPIC, self.receive_scan, qos_profile_sensor_data))
        self.heartbeat = self.create_timer(
            settings.command_heartbeat_sec, self.publish_current,
            clock=RosClock(clock_type=ClockType.STEADY_TIME))

    def receive_clock(self, message):
        with self.changed:
            value = stamp_ns(message.clock)
            if self.sim_ns is not None and value < self.sim_ns:
                self.fault = "仿真时钟倒退，当前采样已失效。"
            self.sim_ns = value
            self.changed.notify_all()

    def receive_scan(self, message):
        try:
            if len(message.ranges) != self.vehicle.lidar_raw_samples:
                raise ValueError("原始雷达点数与共享配置不一致。")
            # Same exact integer-mm/bin/sector conversion as /lidar_distances.
            # Read the timestamped source to reject pre-action/reset frames.
            data = scan_to_distances(
                message.ranges, message.angle_min, message.angle_increment,
                message.range_min, message.range_max, self.vehicle)
            frame = (stamp_ns(message.header.stamp), np.asarray(data, dtype=np.uint16))
            with self.changed:
                self.scans.append(frame)
                self.changed.notify_all()
        except (ValueError, OverflowError) as error:
            with self.changed:
                self.fault = str(error)
                self.changed.notify_all()

    def receive_odom(self, message):
        position, twist = message.pose.pose.position, message.twist.twist
        q = message.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        frame = (stamp_ns(message.header.stamp), {
            "x": position.x, "y": position.y, "linear_x": twist.linear.x,
            "angular_z": twist.angular.z, "yaw": yaw})
        with self.changed:
            if not all(math.isfinite(v) for v in frame[1].values()):
                self.fault = "里程计包含非有限值。"
            self.odometry.append(frame)
            self.changed.notify_all()

    def receive_contacts(self, message):
        timestamp = stamp_ns(message.header.stamp)
        collisions = []
        for state in message.states:
            names = (state.collision1_name, state.collision2_name)
            own = [DEFAULT_ENTITY_NAME in name.split("::")[:-1] for name in names]
            if sum(own) != 1:
                continue  # self contact / a different model
            other = names[1] if own[0] else names[0]
            if any(name in other.split("::")[:-1] for name in config.IGNORED_CONTACT_MODELS):
                continue
            collisions.append(names)
        with self.changed:
            self.contacts_ns[message.header.frame_id] = timestamp
            self.contacts_current[message.header.frame_id] = bool(collisions)
            if collisions:
                self.impacts.append((timestamp, collisions))
            self.changed.notify_all()

    def receive_wheels(self, message):
        with self.changed:
            self.latest_wheels = tuple(message.data)
            self.wheel_sequence += 1
            self.changed.notify_all()

    def publish_current(self):
        with self.changed:
            command = self.command
        message = Int32()
        message.data = command
        self.action_publisher.publish(message)

    def set_command(self, command):
        with self.changed:
            self.command = int(command)
        self.publish_current()

    def clear_frames(self):
        with self.changed:
            self.scans.clear()
            self.odometry.clear()
            self.contacts_ns.clear()
            self.contacts_current.clear()
            self.impacts.clear()
            self.fault = None


class AicarGazeboEnv(gym.Env):
    """Selectable lidar baseline or map/physics-only privileged teacher."""

    metadata = {"render_modes": []}

    def __init__(self, *, role="train", domain_id=None, port=None, header=None,
                 world=None, spawn_pose=None, seed=config.SEED, log_dir=None,
                 settings=None, reward_settings=None, gui=False, real_time_factor=0.0,
                 observation_mode="lidar", privileged_settings=None):
        super().__init__()
        if role not in ("train", "eval"):
            raise ValueError("role 必须为 train 或 eval。")
        self.vehicle = config.vehicle_config(header)
        self.settings = (settings or config.EnvironmentSettings()).validate()
        self.reward_settings = reward_settings or config.RewardSettings()
        self.role = role
        self.domain_id = domain_id if domain_id is not None else (
            config.TRAIN_ROS_DOMAIN if role == "train" else config.EVAL_ROS_DOMAIN)
        self.port = port if port is not None else (
            config.TRAIN_GAZEBO_PORT if role == "train" else config.EVAL_GAZEBO_PORT)
        self.spawn_pose = tuple(spawn_pose or config.SPAWN_POSE)
        if observation_mode not in ("lidar", "privileged"):
            raise ValueError("observation_mode 必须为 lidar 或 privileged。")
        self.observation_mode = observation_mode
        self.ground_truth_map = (GroundTruthMap(world or config.WORLD, privileged_settings)
                                 if observation_mode == "privileged" else None)
        self.observation_space = (spaces.Box(-1.,1.,(self.ground_truth_map.settings.dimension,),np.float32)
                                  if self.ground_truth_map else
                                  spaces.Box(0.,1.,(self.vehicle.lidar_point_count,),np.float32))
        self.truth = None
        self.action_space = spaces.Discrete(self.vehicle.label_count)
        # A full fresh scan must fit inside each fixed physics step interval.
        period = (self.settings.step_sim_sec if self.ground_truth_map else max(
            self.settings.step_sim_sec, 1 / self.vehicle.lidar_scan_hz + self.vehicle.sim_time_step_sec))
        self.iterations = math.ceil(period / self.vehicle.sim_time_step_sec)
        self.step_sim_sec = self.iterations * self.vehicle.sim_time_step_sec
        self._closed = False
        self._needs_reset = True
        self.simulator = self.node = self.executor = self.context = self.thread = None
        self.current_step = 0
        self._physics_ns = 0  # owned server starts paused at time zero
        self.episode_return = 0.
        self.episode_distance = 0.
        self.episode_sim_time = 0.
        self.previous_action = None
        self.step_contact = (False, False)
        self.motion = MotionMonitor(self.settings, self.vehicle.forward_speed_mps)
        log_dir = Path(log_dir) if log_dir else config.RL_MODEL_DIR / "runtime"
        self.progress = LiveState(log_dir / "live_state.json", role=role, observation_mode=observation_mode,
                                  max_episode_sim_sec=self.settings.max_steps * self.step_sim_sec)
        self.progress.update(force=True, phase="starting", episode_number=0)
        try:
            self.context = Context()
            # DDS discovery settings must match the owned ROS launch processes.
            os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
            rclpy.init(context=self.context, domain_id=self.domain_id,
                       signal_handler_options=SignalHandlerOptions.NO)
            self.node = AicarGazeboNode(self.context, self.vehicle, self.settings, f"aicar_rl_{role}",
                                        enable_lidar=not bool(self.ground_truth_map))
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self.executor.spin, daemon=True)
            self.thread.start()
            self.simulator = SimulationProcess(
                domain_id=self.domain_id, port=self.port, world=world or config.WORLD,
                header=header or config.CONFIG_HEADER, spawn_pose=self.spawn_pose, seed=seed,
                log_path=log_dir / f"gazebo_{role}.log", timeout=self.settings.data_timeout_sec,
                gui=gui, real_time_factor=real_time_factor, enable_lidar=not bool(self.ground_truth_map))
            self.pause_client = self.node.create_client(Empty, "/pause_physics")
            self.reset_client = self.node.create_client(Empty, "/reset_world")
            self.models_client = self.node.create_client(GetModelList, "/get_model_list")
            self.pose_client = self.node.create_client(SetEntityState, "/set_entity_state")
            self._wait_for_model()
            self._call(self.pause_client, Empty.Request())
        except BaseException:
            self.close()
            raise

    def _await(self, predicate, message, timeout=None):
        deadline = time.monotonic() + (timeout or self.settings.data_timeout_sec)
        with self.node.changed:
            while True:
                if self.node.fault:
                    raise SimulationDataError(self.node.fault)
                if self.simulator is not None:
                    self.simulator.check_alive()
                if predicate():
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SimulationDataError(message)
                self.node.changed.wait(min(0.05, remaining))

    def _call(self, client, request, timeout=None):
        timeout = timeout or self.settings.data_timeout_sec
        if not client.wait_for_service(timeout_sec=timeout):
            raise SimulationDataError(f"服务不可用：{client.srv_name}")
        future = client.call_async(request)
        self._await(future.done, f"服务超时：{client.srv_name}", timeout)
        if future.exception() is not None:
            raise SimulationDataError(str(future.exception()))
        return future.result()

    def _wait_for_model(self):
        deadline = time.monotonic() + self.settings.startup_timeout_sec
        names = []
        while time.monotonic() < deadline:
            self.simulator.check_alive()
            if self.models_client.wait_for_service(timeout_sec=0.2):
                result = self._call(self.models_client, GetModelList.Request())
                names = list(result.model_names)
                if DEFAULT_ENTITY_NAME in result.model_names:
                    return
        raise SimulationDataError(f"车辆生成超时（发现模型 {names}）；查看 {self.simulator.log_path}")

    def _command(self, label):
        targets = wheel_targets(int(label), self.vehicle)
        expected = (targets.left_mps, targets.right_mps)
        with self.node.changed:
            before = self.node.wheel_sequence
        self.node.set_command(int(label))
        self._await(
            lambda: self.node.wheel_sequence > before and self.node.latest_wheels is not None
            and len(self.node.latest_wheels) == 2
            and all(math.isclose(a, b, abs_tol=1e-8) for a, b in zip(self.node.latest_wheels, expected)),
            "轮速控制器没有确认目标指令。")

    def _advance(self, iterations):
        start_ns = self._physics_ns
        end_ns = start_ns + round(iterations * self.vehicle.sim_time_step_sec * 1e9)
        actual_time, collided, current_contact, self.truth = self.simulator.advance(iterations)
        if abs(round(actual_time * 1e9) - end_ns) > 1000:
            raise SimulationDataError("物理步进时间与预期不符。")
        self.step_contact = (collided, current_contact)
        tolerance = round(self.vehicle.sim_time_step_sec * 1e9)
        self._await(lambda: self.node.sim_ns is not None and self.node.sim_ns >= end_ns - tolerance,
                    "仿真没有推进至预期时间。")
        self._physics_ns = end_ns
        return start_ns, end_ns

    def _snapshot(self, start_ns, end_ns):
        if self.ground_truth_map:
            if self.truth is None or self._physics_ns != end_ns:
                raise SimulationDataError("缺少本次物理步进的精确真值。")
            return None, dict(self.truth), self.step_contact[0], None, end_ns
        age_ns = round(self.settings.state_max_age_sec * 1e9)
        # Sensor timestamps must be within this action, not left over from reset.
        self._await(
            lambda: bool(self.node.scans and self.node.odometry)
            and self.node.scans[-1][0] > start_ns
            and self.node.odometry[-1][0] >= end_ns - age_ns,
            "本次步进缺少新雷达或里程计数据。")
        with self.node.changed:
            scans = [(t, d) for t, d in self.node.scans if start_ns < t <= end_ns + 1_000_000]
            odometry = [(t, d) for t, d in self.node.odometry if start_ns < t <= end_ns + 1_000_000]
            if not scans or not odometry:
                raise SimulationDataError("传感器帧时间戳超出本次动作范围。")
            lidar_stamp, distances = scans[-1]
            odom_stamp, odom = odometry[-1]
            return distances.copy(), dict(odom), self.step_contact[0], lidar_stamp, odom_stamp

    def _observation(self, lidar_mm):
        if self.ground_truth_map:
            return self.ground_truth_map.observation(self.truth,self.previous_action,self.vehicle)
        return (lidar_mm.astype(np.float32) / self.vehicle.lidar_max_distance_mm).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._closed:
            raise RuntimeError("环境已关闭。")
        self._needs_reset = True
        try:
            self.progress.update(force=True, phase="resetting")
            self._command(self.vehicle.stop_command)
            self._call(self.pause_client, Empty.Request())
            self._call(self.reset_client, Empty.Request())  # resets joints; keeps time monotonic
            request = SetEntityState.Request()
            request.state.name = DEFAULT_ENTITY_NAME
            request.state.reference_frame = "world"
            x, y, z, yaw = self.spawn_pose
            request.state.pose.position.x = float(x)
            request.state.pose.position.y = float(y)
            request.state.pose.position.z = float(z)
            request.state.pose.orientation.z = math.sin(yaw / 2)
            request.state.pose.orientation.w = math.cos(yaw / 2)
            # A new EntityState has zero linear/angular velocities.
            result = self._call(self.pose_client, request)
            if not result.success:
                raise SimulationDataError("车辆姿态复位失败。")
            self.node.clear_frames()
            iterations = math.ceil(max(self.settings.reset_settle_sec, self.step_sim_sec)
                                   / self.vehicle.sim_time_step_sec)
            start, end = self._advance(iterations)
            lidar, odom, collision, lidar_ns, odom_ns = self._snapshot(start, end)
            # Ignore contact caches/settling transients; validate the final
            # settled pose and current contacts rather than the old impact latch.
            collision = self.step_contact[1]
            if collision or math.dist((odom["x"], odom["y"]), (x, y)) > 0.03:
                raise SimulationDataError("出生点与障碍物相碰，请修改 SPAWN_POSE。")
            self.current_step = 0
            self.episode_return = self.episode_distance = self.episode_sim_time = 0.
            self.previous_action = None
            self.previous_position = (odom["x"], odom["y"])
            self.motion.reset(end / 1e9, self.previous_position, odom["yaw"])
            self._needs_reset = False
            self.progress.update(force=True, phase="ready",
                                 episode_number=self.progress.data["episode_number"] + 1,
                                 sim_duration_sec=0., distance_m=0., speed_mps=0.,
                                 action=None, reason="", step=0)
            return self._observation(lidar), {
                "observation_mode": self.observation_mode,
                "position": self.previous_position, "sim_time": end / 1e9,
                "yaw": odom["yaw"],
                "lidar_stamp_ns": lidar_ns, "odom_stamp_ns": odom_ns,
                "step": 0, "domain_id": self.domain_id}
        except BaseException as error:
            self.progress.update(force=True, phase="error", error_type=type(error).__name__)
            with suppress(Exception):
                self.close()
            raise

    def step(self, action):
        if self._closed or self._needs_reset:
            raise RuntimeError("开始新回合前必须调用 reset()。")
        if not self.action_space.contains(action):
            raise ValueError(f"非法动作：{action}")
        action = int(action)
        try:
            self._command(action)
            start, end = self._advance(self.iterations)
            lidar, odom, collided, lidar_ns, odom_ns = self._snapshot(start, end)
            position = (odom["x"], odom["y"])
            distance = math.dist(position, self.previous_position)
            dt = (end - start) / 1e9
            self.current_step += 1
            self.episode_distance += distance
            self.episode_sim_time += dt
            motion = self.motion.update(end / 1e9, position, odom["yaw"], odom["linear_x"], distance, dt)
            stuck, slow, looping, outside = (motion[k] for k in ("stuck", "slow", "looping", "out_of_bounds"))
            terminated = collided or stuck or slow or looping or outside
            truncated = not terminated and self.current_step >= self.settings.max_steps
            safety = {}
            if self.ground_truth_map:
                clearance, front_clearance = self.ground_truth_map.safety_distances(odom)
                safety = {"clearance_m": clearance, "front_clearance_m": front_clearance}
            reward, terms = compute_reward(
                lidar, linear_velocity=odom["linear_x"], distance_moved=distance, sim_dt=dt,
                action=action, previous_action=self.previous_action, vehicle=self.vehicle,
                collided=collided, stuck=stuck, looping=looping, slow=slow,
                out_of_bounds=outside, settings=self.reward_settings, **safety)
            self.episode_return += reward
            self.previous_position, self.previous_action = position, action
            reason = ("collision" if collided else "stuck" if stuck else "looping" if looping
                      else "slow" if slow else "out_of_bounds" if outside else "time_limit" if truncated else "")
            info = {
                "observation_mode": self.observation_mode, **safety,
                "step": self.current_step, "position": position, "linear_velocity": odom["linear_x"],
                "angular_velocity": odom["angular_z"], "collided": collided, "stuck": stuck,
                "yaw": odom["yaw"], "action": action, **motion,
                "episode_mean_speed_mps": self.motion.forward_distance / self.episode_sim_time,
                "timeout": truncated, "termination_reason": reason,
                "is_success": bool(truncated), "episode_return": self.episode_return,
                "episode_distance_m": self.episode_distance, "episode_sim_time_sec": self.episode_sim_time,
                "sim_dt": dt, "sim_time": end / 1e9, "lidar_stamp_ns": lidar_ns,
                "odom_stamp_ns": odom_ns, "lidar_ready": not bool(self.ground_truth_map), "reward_terms": terms,
            }
            if terminated or truncated:
                self._command(self.vehicle.stop_command)
                self._needs_reset = True
            self.progress.update(force=terminated or truncated,
                                 phase="episode_finished" if terminated or truncated else "driving",
                                 sim_duration_sec=self.episode_sim_time, distance_m=self.episode_distance,
                                 speed_mps=odom["linear_x"], action=action, step=self.current_step,
                                 reason=reason, speed_compliance=motion["speed_compliance"])
            return self._observation(lidar), reward, terminated, truncated, info
        except BaseException as error:
            self._needs_reset = True
            self.progress.update(force=True, phase="error", error_type=type(error).__name__)
            with suppress(Exception):
                self.close()
            # No fake zero/stale transition is passed to PPO on an I/O failure.
            raise

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        if self.node is not None:
            self.node.set_command(self.vehicle.stop_command)
        if self.simulator is not None:
            self.simulator.close()
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=3)
        if self.thread is not None:
            self.thread.join(timeout=3)
        if self.node is not None:
            self.node.destroy_node()
        if self.context is not None and self.context.ok():
            self.context.shutdown()
        if hasattr(self, "progress"):
            self.progress.update(force=True, phase="closed", ended_at=utc_now())
