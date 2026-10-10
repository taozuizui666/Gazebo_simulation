"""Inspect wheel targets, input shape and a rendered URDF without ROS."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from .config import DEFAULT_URDF, load_config
from .control import wheel_targets
from .lidar_processing import scan_to_distances, training_row
from .robot_description import render_robot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-header")
    parser.add_argument("--label", type=int)
    parser.add_argument("--urdf-output", type=Path)
    parser.add_argument("--csv-row", action="store_true", help="Print a 1 m demo row, not a dataset")
    args = parser.parse_args(argv)
    config = load_config(args.config_header)
    label = config.straight_label if args.label is None else args.label
    targets = wheel_targets(label, config)
    n = config.lidar_raw_samples
    distances = scan_to_distances([1.0] * n, -math.pi, 2 * math.pi / n,
                                  config.lidar_min_range_m,
                                  config.lidar_physical_max_range_m, config)
    if args.urdf_output:
        source_root = Path(__file__).resolve().parents[1]
        template = source_root / DEFAULT_URDF
        if not template.exists():
            from ament_index_python.packages import get_package_share_directory
            template = Path(get_package_share_directory("my_gazebo_launch")) / DEFAULT_URDF
        args.urdf_output.write_text(render_robot(template, config), encoding="utf-8")
    if args.csv_row:
        print(training_row(distances, label, config), end="")
    else:
        print(json.dumps({
            "labels": config.labels, "straight_label": config.straight_label,
            "command": label, "targets": asdict(targets),
            "observation_columns": len(distances), "csv_columns": len(distances) + 1,
            "active_indices": [i for i in range(len(distances)) if config.is_active_index(i)],
            "sample_distances_mm": distances,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
