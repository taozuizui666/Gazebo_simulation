"""Run with the ROS environment sourced: python -m unittest discover -s tests."""

import io
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from rl_training.common.simulator import SimulationProcess


class SimulatorShutdownTests(unittest.TestCase):
    def orphan(self, *, ignore_term=False, listen=True):
        # The launch parent exits first; its child keeps the original group.
        code = f"""
import os, signal, socket, time
if os.fork():
    os._exit(0)
signal.signal(signal.SIGINT, signal.SIG_IGN)
if {ignore_term!r}:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
server = socket.socket()
server.bind(('127.0.0.1', 0))
port = server.getsockname()[1]
if {listen!r}:
    server.listen()
else:
    server.close()
print(port, flush=True)
while True:
    time.sleep(1)
"""
        process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True,
                                   stdout=subprocess.PIPE, text=True)

        def cleanup():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            process.stdout.close()

        self.addCleanup(cleanup)
        port = int(process.stdout.readline())
        process.wait(timeout=5)
        sim = SimulationProcess.__new__(SimulationProcess)
        sim.process, sim.port = process, port
        sim.stepper, sim.log_file = None, io.StringIO()
        sim.log_path = Path("test-gazebo.log")
        return sim, process

    def test_parent_exit_does_not_leave_listening_child(self):
        sim, process = self.orphan()
        self.assertTrue(sim._port_in_use())
        sim.close()
        self.assertFalse(sim._group_running(process))
        self.assertFalse(sim._port_in_use())
        self.assertIsNone(sim.process)
        # Repeated cleanup must not signal an old process-group ID.
        with patch("os.killpg") as kill_group:
            sim.close()
            kill_group.assert_not_called()

    def test_child_ignoring_term_is_killed(self):
        sim, process = self.orphan(ignore_term=True)
        with patch("os.killpg", wraps=os.killpg) as kill_group:
            sim.close()
        self.assertIn((process.pid, signal.SIGKILL),
                      [call.args for call in kill_group.call_args_list])
        self.assertFalse(sim._port_in_use())

    def test_group_is_cleaned_even_after_port_is_free(self):
        sim, process = self.orphan(listen=False)
        self.assertFalse(sim._port_in_use())
        self.assertTrue(sim._group_running(process))
        sim.close()
        self.assertFalse(sim._group_running(process))

    def test_unrelated_listener_is_preserved_and_reported(self):
        sim, process = self.orphan(listen=False)
        with socket.socket() as foreign:
            foreign.bind(("127.0.0.1", sim.port))
            foreign.listen()
            log = sim.log_file
            with self.assertRaisesRegex(RuntimeError, "仍被占用"):
                sim.close()
            self.assertTrue(log.closed)
            self.assertFalse(sim._group_running(process))
            self.assertTrue(sim._port_in_use())

    def test_busy_port_is_rejected_before_launch(self):
        with socket.socket() as foreign, tempfile.TemporaryDirectory() as directory:
            foreign.bind(("127.0.0.1", 0))
            foreign.listen()
            world = Path(directory) / "world.sdf"
            world.write_text('<sdf version="1.6"><world name="test"/></sdf>')
            with patch.object(SimulationProcess, "_build_native") as build:
                with self.assertRaisesRegex(RuntimeError, "已被使用"):
                    SimulationProcess(domain_id=96, port=foreign.getsockname()[1],
                                      world=world, header=world, spawn_pose=(0, 0, 0, 0),
                                      seed=42, log_path=Path(directory) / "gazebo.log", timeout=1)
                build.assert_not_called()
            self.assertFalse((Path(directory) / "gazebo.log").exists())

    def test_stepper_error_still_cleans_process_group(self):
        sim, process = self.orphan()
        sim.stepper = Mock()
        sim.stepper.communicate.side_effect = OSError("stepper pipe failure")
        with self.assertRaisesRegex(OSError, "stepper pipe failure"):
            sim.close()
        self.assertFalse(sim._group_running(process))
        self.assertFalse(sim._port_in_use())

    def test_environment_releases_ros_resources_on_shutdown_error(self):
        from rl_training.common.rl_env import AicarGazeboEnv

        env = AicarGazeboEnv.__new__(AicarGazeboEnv)
        env._closed = False
        env.vehicle = Mock(stop_command=15)
        env.node, env.executor, env.thread, env.context = Mock(), Mock(), Mock(), Mock()
        env.simulator = Mock()
        env.simulator.close.side_effect = RuntimeError("port still busy")
        with self.assertRaisesRegex(RuntimeError, "port still busy"):
            env.close()
        env.executor.shutdown.assert_called_once()
        env.thread.join.assert_called_once()
        env.node.destroy_node.assert_called_once()
        env.context.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
