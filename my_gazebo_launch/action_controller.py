"""ROS 2 label -> fixed-forward-speed Twist; no Raspberry Pi/PWM/UDP."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Int32

from .config import CMD_VEL_TOPIC, LABEL_TOPIC, WHEEL_TARGET_TOPIC, load_config
from .control import wheel_targets


class ActionController(Node):
    def __init__(self):
        super().__init__("action_controller")
        self.declare_parameter("config_header", "")
        self.config = load_config(self.get_parameter("config_header").value or None)
        self.command = self.config.stop_command
        self.last_command_ns = None
        self.previous_time_ns = None
        self.velocity_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 1)
        self.wheels_pub = self.create_publisher(Float64MultiArray, WHEEL_TARGET_TOPIC, 1)
        self.subscription = self.create_subscription(Int32, LABEL_TOPIC, self.receive, 1)
        self.timer = self.create_timer(1 / self.config.command_publish_hz, self.update)
        self.get_logger().info(
            f"labels=0..{self.config.label_count - 1}, straight={self.config.straight_label}, "
            f"forward={self.config.forward_speed_mps} m/s; waiting for a label")

    def receive(self, message):
        try:
            wheel_targets(message.data, self.config)
        except ValueError as error:
            self.get_logger().error(str(error))
            self.stop()
            return
        self.command = message.data
        self.last_command_ns = self.get_clock().now().nanoseconds
        self.publish(self.command)

    def publish(self, label):
        targets = wheel_targets(label, self.config)
        message = Twist()
        message.linear.x = targets.linear_mps
        message.angular.z = targets.angular_rad_s
        self.velocity_pub.publish(message)
        wheels = Float64MultiArray()
        wheels.data = [targets.left_mps, targets.right_mps]
        self.wheels_pub.publish(wheels)

    def stop(self):
        self.command = self.config.stop_command
        self.last_command_ns = None
        self.publish(self.command)

    def update(self):
        now = self.get_clock().now().nanoseconds
        if self.previous_time_ns is not None and now < self.previous_time_ns:
            self.stop()
        self.previous_time_ns = now
        if (self.last_command_ns is None or now < self.last_command_ns
                or now - self.last_command_ns >= self.config.command_timeout_ms * 1_000_000):
            self.stop()
        else:
            self.publish(self.command)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ActionController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
