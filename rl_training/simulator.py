"""Own one paused Gazebo process; master/ROS domain are isolated per environment."""

import os
import fcntl
import select
import shlex
from pathlib import Path
import signal
import socket
import subprocess
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory

from my_gazebo_launch.config import DEFAULT_ENTITY_NAME


class SimulationProcess:
    def __init__(self, *, domain_id, port, world, header, spawn_pose, seed,
                 log_path, timeout, gui=False, real_time_factor=0.0):
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
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
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

    def advance(self, iterations):
        self.check_alive()
        if self.stepper is None:
            self._start_stepper()
        self.stepper.stdin.write(f"{iterations}\n")
        self.stepper.stdin.flush()
        result = self._read_stepper(f"ACK {iterations}")
        fields = result.split()
        if len(fields) != 5:
            raise RuntimeError(f"物理步进返回数据异常：{result}")
        return float(fields[2]), bool(int(fields[3])), bool(int(fields[4]))

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
        directory = Path(__file__).resolve().parents[1] / "build" / "rl_native"
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

    def close(self):
        if self.stepper is not None:
            try:
                self.stepper.communicate("0\n", timeout=3)
            except (subprocess.TimeoutExpired, BrokenPipeError):
                self.stepper.kill()
                self.stepper.wait(timeout=3)
            self.stepper = None
        process = self.process
        if process is not None and process.poll() is None:
            for sig, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
                if process.poll() is not None:
                    break
                try:
                    # Let ros2 launch deliver the first interrupt to its nodes;
                    # signalling the entire group twice interrupts their cleanup.
                    if sig == signal.SIGINT:
                        os.kill(process.pid, sig)
                    else:
                        os.killpg(process.pid, sig)
                    process.wait(timeout=timeout)
                except ProcessLookupError:
                    break
                except subprocess.TimeoutExpired:
                    continue
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
