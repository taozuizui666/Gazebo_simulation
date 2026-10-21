"""Own one paused Gazebo process; master/ROS domain are isolated per environment."""

import os
import fcntl
import select
import shlex
from pathlib import Path
import signal
import socket
import subprocess
import time
import xml.etree.ElementTree as ET
import math

from ament_index_python.packages import get_package_share_directory

from my_gazebo_launch.config import DEFAULT_ENTITY_NAME


class SimulationProcess:
    def __init__(self, *, domain_id, port, world, header, spawn_pose, seed,
                 log_path, timeout, gui=False, real_time_factor=0.0, enable_lidar=True):
        self.domain_id, self.port = domain_id, port
        self.process = None
        self.stepper = None
        self.log_file = None
        self.timeout = timeout
        self.log_path = Path(log_path)
        world = Path(world).expanduser()
        if not world.is_absolute():
            world = Path(get_package_share_directory("my_gazebo_launch")) / world
        if not world.is_file():
            raise FileNotFoundError(world)
        self.world_name = ET.parse(world).getroot().find("world").get("name")
        self.environment = dict(os.environ)
        self.environment.update(
            ROS_DOMAIN_ID=str(domain_id), ROS_LOCALHOST_ONLY=os.environ.get("ROS_LOCALHOST_ONLY", "1"),
            GAZEBO_MASTER_URI=f"http://127.0.0.1:{port}",
            GAZEBO_MODEL_DATABASE_URI="",
            ROS_LOG_DIR=str(self.log_path.parent / f"ros_domain_{domain_id}"))
        if self._port_in_use():
            raise RuntimeError(f"Gazebo port {port} 已被使用；请修改 RL 配置，不能接管其他仿真。")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w", encoding="utf-8")
        x, y, z, yaw = spawn_pose
        try:
            self.stepper_binary, step_plugin = self._build_native()
            self.process = subprocess.Popen([
                "ros2", "launch", "my_gazebo_launch", "train_world.launch.py",
                f"gui:={str(gui).lower()}", "pause:=true", f"world:={world}",
                f"real_time_factor:={real_time_factor}", "lockstep:=true",
                f"training_step_plugin:={step_plugin}",
                f"enable_lidar:={str(enable_lidar).lower()}",
                f"config_header:={Path(header).resolve()}", f"seed:={seed}",
                f"x:={x}", f"y:={y}", f"z:={z}", f"yaw:={yaw}",
            ], env=self.environment, stdout=self.log_file, stderr=subprocess.STDOUT,
                start_new_session=True)
        except BaseException:
            self.close()
            raise

    def check_alive(self):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError(f"Gazebo 进程已退出；查看 {self.log_path}")

    def _port_in_use(self):
        with socket.socket() as probe:
            probe.settimeout(0.2)
            return probe.connect_ex(("127.0.0.1", self.port)) == 0

    def advance(self, iterations):
        self.check_alive()
        if self.stepper is None:
            self._start_stepper()
        self.stepper.stdin.write(f"{iterations}\n")
        self.stepper.stdin.flush()
        result = self._read_stepper(f"ACK {iterations}")
        fields = result.split()
        if len(fields) != 11:
            raise RuntimeError(f"物理步进返回数据异常：{result}")
        values = list(map(float, fields[2:]))
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError("物理真值包含非有限数据。")
        truth = dict(zip(("x", "y", "yaw", "linear_x", "angular_z", "z"), values[3:]))
        return values[0], bool(values[1]), bool(values[2]), truth

    def _read_stepper(self, expected):
        ready, _, _ = select.select([self.stepper.stdout], [], [], self.timeout)
        if not ready:
            raise RuntimeError("Gazebo 步进连接超时。")
        actual = self.stepper.stdout.readline().strip()
        if actual != expected and not actual.startswith(expected + " "):
            raise RuntimeError(f"Gazebo 步进连接异常：{actual!r}，期望 {expected!r}")
        return actual

    @staticmethod
    def _build_native():
        source_dir = Path(__file__).with_name("native")
        directory = Path(__file__).resolve().parents[2] / "build" / "rl_native"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "build.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for filename, extra in (("world_stepper", []), ("training_step_plugin", ["-shared", "-fPIC"])):
                source = source_dir / (filename + ".cpp")
                binary = directory / (filename + (".so" if extra else ""))
                if binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
                    continue
                flags = subprocess.check_output(["pkg-config", "--cflags", "--libs", "gazebo"], text=True)
                temporary = binary.with_name(f"{binary.name}.{os.getpid()}.tmp")
                subprocess.run(["g++", "-O2", "-std=c++17", *extra, str(source), "-o", str(temporary),
                                *shlex.split(flags)], check=True, timeout=120)
                temporary.replace(binary)
        return directory / "world_stepper", directory / "training_step_plugin.so"

    def _start_stepper(self):
        self.stepper = subprocess.Popen(
            [str(self.stepper_binary), self.world_name], env=self.environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log_file,
            text=True, bufsize=1)
        self._read_stepper("READY")

    @staticmethod
    def _group_running(process):
        if process.poll() is None:
            return True
        # launch can exit before its children. Ignore zombies, which own no
        # sockets and may await reaping by another process after launch exits.
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = path.read_text().rsplit(")", 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if int(fields[2]) == process.pid and fields[0] != "Z":
                return True
        return False

    @staticmethod
    def _wait_until(condition, timeout):
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)
        return True

    def _stop_launch(self):
        process = self.process
        if process is None:
            return
        for sig, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
            if not self._group_running(process):
                break
            try:
                # Let launch distribute the first interrupt. Later signals
                # target our original group even if launch has already exited.
                if sig == signal.SIGINT:
                    if process.poll() is not None:
                        continue
                    os.kill(process.pid, sig)
                else:
                    os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            if self._wait_until(lambda: not self._group_running(process), timeout):
                break
        if self._group_running(process):
            raise RuntimeError(f"Gazebo 进程组 {process.pid} 未能退出，已停止切换地图；查看 {self.log_path}")
        self.process = None
        if not self._wait_until(lambda: not self._port_in_use(), 5):
            raise RuntimeError(
                f"本实例已退出，但 Gazebo port {self.port} 仍被占用；"
                f"已停止切换地图，不会终止其他实例。查看 {self.log_path}")

    def close(self):
        try:
            try:
                if self.stepper is not None:
                    try:
                        self.stepper.communicate("0\n", timeout=3)
                    except (subprocess.TimeoutExpired, BrokenPipeError):
                        self.stepper.kill()
                        self.stepper.wait(timeout=3)
                    finally:
                        self.stepper = None
            finally:
                self._stop_launch()
        finally:
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None
