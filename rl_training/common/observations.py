"""Describe the selected input contract without mixing saved model families."""

from rl_training.privileged.observation import PrivilegedSettings


def observation_specification(mode, vehicle):
    if mode == "privileged":
        return PrivilegedSettings().validate().specification()
    if mode == "lidar":
        return {"kind": "canonical_lidar_v1", "dimension": vehicle.lidar_point_count,
                "lidar_required": True, "preprocessing": "canonical_mm / max_mm; policy sqrt"}
    raise ValueError(f"未知观测模式：{mode}")
