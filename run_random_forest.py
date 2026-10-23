#!/usr/bin/env python3
"""打开 Gazebo，使用训练导出的随机森林根据实时雷达驾驶小车。"""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

# ===================== 可修改配置区 =====================
ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / "software/ML_model/simulation/randomForest.joblib"
WORLD = "assets/worlds/simple/ellipse_002.sdf"  # 本次训练保留的验证地图。
CONFIG_HEADER = ROOT / "my_gazebo_launch/vehicle_config.h"
GUI = True
RUN_SECONDS = 0.0             # 第一帧雷达之后的仿真秒数；0 表示一直运行，Ctrl+C 退出。
SEED = 42
ROS_DOMAIN = 93
GAZEBO_PORT = 11553
STARTUP_TIMEOUT_SEC = 60.0    # 等待首帧雷达的现实秒数。
STATUS_INTERVAL_SEC = 2.0     # 终端状态输出间隔，现实秒数。
LOG_PATH = ROOT / "results/random_forest/gazebo.log"
# ===================== 配置区结束 =====================


def load_model(path, vehicle):
    import joblib
    import numpy as np

    artifact = joblib.load(path)
    report = json.loads(path.with_name("training_report.json").read_text(encoding="utf-8"))
    config = report["configuration"]
    classifier = artifact["classifier"]
    indices = artifact["selected_indices"]
    if (artifact["data_source"] != "simulation"
            or artifact["lidar_point_count"] != vehicle.lidar_point_count
            or artifact["straight_command"] != vehicle.straight_label
            or artifact["allow_zero_input"] is not True
            or config["LIDAR_POINT_COUNT"] != vehicle.lidar_point_count
            or config["MAX_DISTANCE_MM"] != vehicle.lidar_max_distance_mm
            or config["USE_FRONT_SECTOR_ONLY"] != bool(vehicle.lidar_front_sector_only)):
        raise ValueError("模型雷达格式、零值约定或直行标签与车辆配置不一致。")
    if (not indices or any(type(i) is not int or not 0 <= i < vehicle.lidar_point_count for i in indices)
            or len(set(indices)) != len(indices)
            or len(indices) != classifier.n_features_in_
            or indices != report["selected_indices"]):
        raise ValueError("模型特征索引与训练报告或分类器输入不一致。")
    labels = classifier.classes_.tolist()
    if (labels != artifact["class_commands"] or labels != report["class_commands"]
            or any(label not in vehicle.labels for label in labels)):
        raise ValueError("模型输出必须是车辆支持的转向标签。")
    # Python predict 直接返回原始标签；不做 C++ 类别索引到指令的二次映射。
    prediction = int(classifier.predict(np.zeros((1, len(indices)), dtype=np.float32))[0])
    if prediction not in labels:
        raise ValueError("模型试推理返回了无效标签。")
    return classifier, indices


def stop_launch(process):
    # 先让 ROS launch 正常关闭子节点；超时才终止本脚本创建的进程组。
    try:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=20)
    finally:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                continue
            break


def drive(args, world, vehicle, classifier, indices):
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import Int32, UInt16MultiArray
    from my_gazebo_launch.config import LABEL_TOPIC, OBSERVATION_TOPIC

    class ForestDriver(Node):
        def __init__(self):
            super().__init__("random_forest_driver", parameter_overrides=[
                rclpy.parameter.Parameter("use_sim_time", value=True)])
            self.scans = 0
            self.first_scan_ns = None
            self.last_status = 0.0
            self.publisher = self.create_publisher(Int32, LABEL_TOPIC, 1)
            self.subscription = self.create_subscription(
                UInt16MultiArray, OBSERVATION_TOPIC, self.receive, 1)

        def publish(self, label):
            self.publisher.publish(Int32(data=label))

        def receive(self, message):
            distances = np.asarray(message.data, dtype=np.float32)
            if (len(distances) != vehicle.lidar_point_count
                    or np.any(distances >= vehicle.lidar_max_distance_mm)
                    or not np.any(distances)):
                self.publish(vehicle.stop_command)
                self.get_logger().warning("无效或全零雷达帧，停车。", throttle_duration_sec=2.0)
                return
            label = int(classifier.predict(distances[indices].reshape(1, -1))[0])
            self.publish(label)
            self.scans += 1
            if self.first_scan_ns is None:
                self.first_scan_ns = self.get_clock().now().nanoseconds
            if time.monotonic() - self.last_status >= STATUS_INTERVAL_SEC:
                print(f"雷达帧 {self.scans}；预测转向标签 {label}（{vehicle.straight_label} 为直行）", flush=True)
                self.last_status = time.monotonic()

    with socket.socket() as probe:
        probe.settimeout(0.2)
        if probe.connect_ex(("127.0.0.1", GAZEBO_PORT)) == 0:
            raise RuntimeError(f"Gazebo 端口 {GAZEBO_PORT} 已被占用，请修改脚本顶部 GAZEBO_PORT。")
    os.environ.update(ROS_DOMAIN_ID=str(ROS_DOMAIN), ROS_LOCALHOST_ONLY="1",
                      GAZEBO_MASTER_URI=f"http://127.0.0.1:{GAZEBO_PORT}",
                      GAZEBO_MODEL_DATABASE_URI="")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    os.environ["ROS_LOG_DIR"] = str(LOG_PATH.parent / "ros")
    process = node = None
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    try:
        node = ForestDriver()
        x, y, z, yaw = world.eval_spawn_poses[0]
        with LOG_PATH.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([
                "ros2", "launch", "my_gazebo_launch", "train_world.launch.py",
                f"world:={world.world}", f"config_header:={CONFIG_HEADER}",
                f"gui:={str(not args.headless and GUI).lower()}", f"seed:={SEED}",
                f"x:={x}", f"y:={y}", f"z:={z}", f"yaw:={yaw}",
            ], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"等待雷达；Gazebo 日志：{LOG_PATH}\n按 Ctrl+C 停车并关闭本次仿真。", flush=True)
        started = time.monotonic()
        while rclpy.ok():
            if process.poll() is not None:
                raise RuntimeError(f"Gazebo 已退出（状态 {process.returncode}），请查看 {LOG_PATH}")
            rclpy.spin_once(node, timeout_sec=0.2)
            if node.first_scan_ns is None:
                if time.monotonic() - started > STARTUP_TIMEOUT_SEC:
                    raise RuntimeError(f"等待有效雷达超时，请查看 {LOG_PATH}")
            elif args.seconds and (node.get_clock().now().nanoseconds - node.first_scan_ns) / 1e9 >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None and rclpy.ok():
                node.publish(vehicle.stop_command)
                time.sleep(0.1)  # 让停车消息在关闭仿真之前送达。
        finally:
            try:
                if process is not None:
                    stop_launch(process)
            finally:
                if node is not None:
                    print(f"运行结束；共完成 {node.scans} 帧随机森林推理。", flush=True)
                    node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL, help="训练导出的 joblib；同目录需有 training_report.json")
    parser.add_argument("--world", default=WORLD, help="地图 SDF；相对路径以 Gazebo_simulation 为基准")
    parser.add_argument("--check", action="store_true", help="检查模型、地图与车辆配置，不启动仿真")
    parser.add_argument("--headless", action="store_true", help="关闭 Gazebo 图形窗口")
    parser.add_argument("--seconds", type=float, default=RUN_SECONDS, help="运行仿真秒数；0 表示一直运行")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds 必须为非负有限数。")
    if os.environ.get("RF_RUN_ENV_READY") != "1":
        os.execvpe("bash", ["bash", "-c",
            'set -e; source /opt/ros/humble/setup.bash; '
            'source /usr/share/gazebo/setup.bash; source "$1/install/setup.bash"; '
            'export RF_RUN_ENV_READY=1; shift; exec "$@"',
            "run_random_forest", str(ROOT), sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            os.environ)
    from my_gazebo_launch.config import load_config
    from my_gazebo_launch.world_config import load_world_settings

    try:
        model = args.model.expanduser().resolve(strict=True)
        vehicle = load_config(CONFIG_HEADER)
        classifier, indices = load_model(model, vehicle)
        world = load_world_settings(args.world)
        print(f"模型：{model}\n输入特征数：{len(indices)}；输出标签：{classifier.classes_.tolist()}\n"
              f"地图：{world.world}\n出生点：{world.eval_spawn_poses[0]}\n"
              f"目标前进速度：{vehicle.forward_speed_mps:g} m/s；ROS domain：{ROS_DOMAIN}", flush=True)
        if args.check:
            print("配置与试推理检查通过；未启动仿真。")
        else:
            drive(args, world, vehicle, classifier, indices)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"运行失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
