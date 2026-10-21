"""Optional scan waits must never fabricate a frame or hide simulator faults."""

from collections import deque
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from rl_training.common.rl_env import AicarGazeboEnv, SimulationDataError


class LidarTimeoutTests(unittest.TestCase):
    def environment(self, *, optional=True):
        env = AicarGazeboEnv.__new__(AicarGazeboEnv)
        env.ground_truth_map = True
        env.capture_lidar = True
        env.lidar_wait_sec = 0.001 if optional else None
        env.settings = SimpleNamespace(data_timeout_sec=0.001)
        env.vehicle = SimpleNamespace(lidar_scan_hz=5.5, sim_time_step_sec=0.001)
        env.node = SimpleNamespace(changed=threading.Condition(), fault=None, scans=deque(maxlen=32))
        env.simulator = Mock()
        env.truth = {"x": 1.2, "y": 0.4}
        env._physics_ns = 200_000_000
        env.step_contact = (False, False)
        return env

    def test_optional_timeout_keeps_truth_without_fake_scan(self):
        env = self.environment()
        lidar, truth, collision, stamp, end = env._snapshot(0, 200_000_000)
        self.assertIsNone(lidar)
        self.assertIsNone(stamp)
        self.assertEqual(truth, env.truth)
        self.assertEqual(end, 200_000_000)
        self.assertFalse(collision)
        env.simulator.advance.assert_not_called()
        env.simulator.close.assert_not_called()

    def test_default_mode_still_requires_a_scan(self):
        env = self.environment(optional=False)
        with self.assertRaisesRegex(SimulationDataError, "缺少新鲜雷达帧"):
            env._snapshot(0, 200_000_000)

    def test_stale_and_future_frames_are_not_exported(self):
        env = self.environment()
        for stamp in (-1, 0, 17_000_000, 201_000_000):
            env.node.scans.append((stamp, np.ones(200, dtype=np.uint16)))
        self.assertIsNone(env._snapshot(0, 200_000_000)[0])

    def test_all_zero_scan_is_valid_and_latest_valid_frame_is_selected(self):
        env = self.environment()
        zero = np.zeros(200, dtype=np.uint16)
        env.node.scans.extend([(18_000_000, np.ones(200, dtype=np.uint16)),
                               (180_000_000, zero),
                               (201_000_000, np.ones(200, dtype=np.uint16))])
        frame = env._snapshot(0, 200_000_000)
        self.assertEqual(frame[3], 180_000_000)
        np.testing.assert_array_equal(frame[0], zero)
        self.assertIsNot(frame[0], zero)

    def test_delayed_delivery_is_accepted_without_advancing_physics(self):
        env = self.environment()
        env.lidar_wait_sec = 0.2

        def deliver():
            time.sleep(0.01)
            with env.node.changed:
                env.node.scans.append((180_000_000, np.ones(200, dtype=np.uint16)))
                env.node.changed.notify_all()

        thread = threading.Thread(target=deliver)
        thread.start()
        try:
            self.assertEqual(env._snapshot(0, 200_000_000)[3], 180_000_000)
            env.simulator.advance.assert_not_called()
        finally:
            thread.join()

    def test_sensor_fault_and_simulator_exit_are_not_treated_as_missing_frames(self):
        env = self.environment()
        env.node.fault = "原始雷达点数与共享配置不一致。"
        with self.assertRaisesRegex(SimulationDataError, "点数"):
            env._snapshot(0, 200_000_000)
        env.node.fault = None
        env.simulator.check_alive.side_effect = RuntimeError("Gazebo 进程已退出")
        with self.assertRaisesRegex(RuntimeError, "进程已退出"):
            env._snapshot(0, 200_000_000)

    def test_keyboard_interrupt_is_not_swallowed(self):
        env = self.environment()
        env.simulator.check_alive.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            env._snapshot(0, 200_000_000)


if __name__ == "__main__":
    unittest.main()
