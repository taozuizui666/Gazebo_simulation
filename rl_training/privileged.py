"""Teacher observations from static SDF collision geometry and exact physics pose.

No ray sensor, scan conversion, lidar topic or noisy measurement is used. The
local signed-distance field is a geometric encoding of the world map. Unsupported
world geometry is rejected instead of silently omitting an obstacle.
"""

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

# ===================== 特权教师可修改配置区 =====================
GRID_ROWS = 17                       # 纵向：从车后到车前
GRID_COLS = 17                       # 横向：从车右到车左
GRID_BACK_M = 0.4
GRID_FRONT_M = 1.4
GRID_HALF_WIDTH_M = 0.9
DISTANCE_CLIP_M = 0.5                # 距离场归一化范围；负值表示障碍内部
BODY_CLEARANCE_RADIUS_M = 0.085      # 奖励用保守车体半径；终止仍以真实接触为准
FORWARD_CHECK_M = 1.5
FORWARD_CHECK_SAMPLES = 61
OBSTACLE_Z_MIN_M = 0.02
OBSTACLE_Z_MAX_M = 0.22
# ===================== 配置区结束 =====================


@dataclass(frozen=True)
class PrivilegedSettings:
    grid_rows: int = GRID_ROWS
    grid_cols: int = GRID_COLS
    grid_back_m: float = GRID_BACK_M
    grid_front_m: float = GRID_FRONT_M
    grid_half_width_m: float = GRID_HALF_WIDTH_M
    distance_clip_m: float = DISTANCE_CLIP_M
    body_clearance_radius_m: float = BODY_CLEARANCE_RADIUS_M
    forward_check_m: float = FORWARD_CHECK_M
    forward_check_samples: int = FORWARD_CHECK_SAMPLES
    obstacle_z_min_m: float = OBSTACLE_Z_MIN_M
    obstacle_z_max_m: float = OBSTACLE_Z_MAX_M

    def validate(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("特权观测参数必须是正的有限数。")
        if any(not isinstance(v, int) or v < 3 for v in
               (self.grid_rows, self.grid_cols, self.forward_check_samples)):
            raise ValueError("距离场行列数与前方检查采样数必须为 >=3 的整数。")
        if self.obstacle_z_max_m <= self.obstacle_z_min_m:
            raise ValueError("障碍物高度范围无效。")
        return self

    @property
    def dimension(self):
        return self.grid_rows * self.grid_cols + 3

    def specification(self):
        return {"kind": "ground_truth_local_distance_field_v1", "dimension": self.dimension,
                "settings": asdict(self), "state_features": ["forward_speed", "yaw_rate", "previous_action"],
                "lidar_required": False, "pose_source": "gazebo_physics_step_ack"}


def resolve_world(world):
    path = Path(world).expanduser()
    if not path.is_absolute():
        from ament_index_python.packages import get_package_share_directory
        path = Path(get_package_share_directory("my_gazebo_launch")) / path
    return path.resolve(strict=True)


def observation_specification(mode, vehicle):
    if mode == "privileged":
        return PrivilegedSettings().validate().specification()
    if mode == "lidar":
        return {"kind": "canonical_lidar_v1", "dimension": vehicle.lidar_point_count,
                "lidar_required": True, "preprocessing": "canonical_mm / max_mm; policy sqrt"}
    raise ValueError(f"未知观测模式：{mode}")


def _pose(element):
    pose = element.find("pose")
    if pose is not None and pose.attrib:
        raise ValueError("特权地图暂不支持带 relative_to/自定义格式的 pose。")
    values = list(map(float, element.findtext("pose", "0 0 0 0 0 0").split()))
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise ValueError("SDF pose 必须是六个有限数值。")
    x,y,z,roll,pitch,yaw = values
    if abs(roll) > 1e-6 or abs(pitch) > 1e-6:
        raise ValueError("当前二维特权地图只支持无 roll/pitch 倾斜的障碍。")
    c,s = math.cos(yaw),math.sin(yaw)
    return np.array([[c,-s,x],[s,c,y],[0,0,1.]]), z


class GroundTruthMap:
    def __init__(self, world, settings=None):
        self.settings = (settings or PrivilegedSettings()).validate()
        self.path = resolve_world(world)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        root = ET.parse(self.path).getroot().find("world")
        if root is None:
            raise ValueError("特权地图必须是包含 world 的 SDF 文件。")
        if root.findall("frame") or root.findall("state"):
            raise ValueError("当前特权地图不支持 frame 或 state 覆盖，请先展平世界几何。")
        for include in root.findall("include"):
            if include.findtext("uri", "").rstrip('/') not in ("model://sun", "model://ground_plane"):
                raise ValueError("特权地图暂不支持外部模型 include，请展平为静态 box 碰撞体。")
        centers, half_sizes, angles = [], [], []
        for model in root.findall("model"):
            if model.get("name") in ("ground_plane", "sun"):
                continue
            if model.findtext("static", "false").strip().lower() not in ("true", "1"):
                raise ValueError("当前教师读取静态世界几何，不支持动态障碍。")
            if model.findall("model") or model.findall("include") or model.findall("plugin"):
                raise ValueError("特权地图不支持嵌套模型或可移动模型插件。")
            model_matrix, model_z = _pose(model)
            for link in model.findall("link"):
                link_matrix, link_z = _pose(link)
                for collision in link.findall("collision"):
                    box = collision.find("geometry/box/size")
                    if box is None:
                        raise ValueError("当前教师地图仅支持静态 box 碰撞体，不能忽略其他形状。")
                    size = np.asarray(list(map(float, box.text.split())))
                    if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
                        raise ValueError("无效的 box 尺寸。")
                    collision_matrix, collision_z = _pose(collision)
                    z = model_z + link_z + collision_z
                    if z + size[2]/2 <= self.settings.obstacle_z_min_m or z - size[2]/2 >= self.settings.obstacle_z_max_m:
                        continue
                    matrix = model_matrix @ link_matrix @ collision_matrix
                    centers.append(matrix[:2,2]);half_sizes.append(size[:2]/2)
                    angles.append(math.atan2(matrix[1,0],matrix[0,0]))
        if not centers:
            raise ValueError("特权地图没有可用于避障的碰撞几何。")
        self.centers, self.half_sizes = np.asarray(centers),np.asarray(half_sizes)
        self.cos, self.sin = np.cos(angles),np.sin(angles)
        x,y = np.meshgrid(np.linspace(-self.settings.grid_back_m,self.settings.grid_front_m,self.settings.grid_rows),
                         np.linspace(-self.settings.grid_half_width_m,self.settings.grid_half_width_m,self.settings.grid_cols), indexing="ij")
        self.local_grid = np.column_stack((x.ravel(),y.ravel()))
        self.forward_x = np.linspace(0,self.settings.forward_check_m,self.settings.forward_check_samples)
        self.local_forward = np.column_stack((self.forward_x,np.zeros_like(self.forward_x)))

    def signed_distance(self, points):
        points = np.asarray(points,dtype=np.float64).reshape(-1,2)
        delta = points[:,None,:] - self.centers[None,:,:]
        x = delta[:,:,0]*self.cos + delta[:,:,1]*self.sin
        y = -delta[:,:,0]*self.sin + delta[:,:,1]*self.cos
        q = np.abs(np.stack((x,y),axis=-1)) - self.half_sizes
        distances = np.linalg.norm(np.maximum(q,0),axis=-1) + np.minimum(q.max(axis=-1),0)
        return distances.min(axis=1)

    @staticmethod
    def world_points(local, truth):
        c,s = math.cos(truth["yaw"]),math.sin(truth["yaw"])
        return local @ np.array([[c,s],[-s,c]]) + (truth["x"],truth["y"])

    def observation(self, truth, previous_action, vehicle):
        distances = self.signed_distance(self.world_points(self.local_grid,truth))
        field = np.clip(distances/self.settings.distance_clip_m,-1,1)
        max_yaw = max(2*vehicle.forward_speed_mps*vehicle.max_wheel_speed_ratio/vehicle.wheel_separation_m,1e-6)
        extra = [np.clip(truth["linear_x"]/vehicle.forward_speed_mps,-2,2)/2,
                 np.clip(truth["angular_z"]/max_yaw,-2,2)/2,
                 0 if previous_action is None else (vehicle.straight_label-previous_action)/vehicle.straight_label]
        return np.concatenate((field,extra)).astype(np.float32)

    def safety_distances(self, truth):
        # Clearance of a conservative swept disk along straight forward motion.
        # This is queried from map geometry, not a lidar beam/range measurement.
        gap = self.signed_distance(self.world_points(self.local_forward,truth)) - self.settings.body_clearance_radius_m
        occupied = np.flatnonzero(gap <= 0)
        front = self.forward_x[occupied[0]] if occupied.size else self.settings.forward_check_m
        return max(0.,float(gap[0])),float(front)
