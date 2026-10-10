"""Publish training-format millimetre observations from the Gazebo ray sensor."""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import MultiArrayDimension, UInt16MultiArray

from .config import OBSERVATION_TOPIC, RAW_SCAN_TOPIC, load_config
from .lidar_processing import scan_to_distances


class LidarAdapter(Node):
    def __init__(self):
        super().__init__("lidar_adapter")
        self.declare_parameter("config_header", "")
        self.config = load_config(self.get_parameter("config_header").value or None)
        self.publisher = self.create_publisher(UInt16MultiArray, OBSERVATION_TOPIC, 1)
        self.subscription = self.create_subscription(
            LaserScan, RAW_SCAN_TOPIC, self.receive, qos_profile_sensor_data)
        self.get_logger().info(
            f"{RAW_SCAN_TOPIC} -> {OBSERVATION_TOPIC}: "
            f"{self.config.lidar_point_count} clockwise bins, integer mm, "
            f"valid distance < {self.config.lidar_max_distance_mm}")

    def receive(self, scan):
        try:
            values = scan_to_distances(
                scan.ranges, scan.angle_min, scan.angle_increment,
                scan.range_min, scan.range_max, self.config)
        except ValueError as error:
            self.get_logger().error(str(error))
            return
        message = UInt16MultiArray()
        dimension = MultiArrayDimension()
        dimension.label = "lidar_bins_mm"
        dimension.size = dimension.stride = self.config.lidar_point_count
        message.layout.dim = [dimension]
        message.layout.data_offset = 0
        message.data = values
        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LidarAdapter()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
