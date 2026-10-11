"""地图路径及其同名 JSON 中的出生点、工作边界。"""

from dataclasses import dataclass
import json
import math
from pathlib import Path


def resolve_world(world):
    path = Path(world).expanduser()
    if not path.is_absolute():
        source = Path(__file__).resolve().parents[1] / path
        if source.is_file():
            path = source
        else:
            from ament_index_python.packages import get_package_share_directory
            path = Path(get_package_share_directory("my_gazebo_launch")) / path
    return path.resolve(strict=True)


@dataclass(frozen=True)
class WorldSettings:
    world: Path
    train_spawn_poses: tuple
    eval_spawn_poses: tuple
    workspace_bounds: tuple | None


def load_world_settings(world, *, train_spawn_poses=None, eval_spawn_poses=None,
                        workspace_bounds=None, disable_bounds=False):
    path = resolve_world(world)
    metadata = path.with_suffix(".json")
    data = json.loads(metadata.read_text(encoding="utf-8")) if metadata.is_file() else {}
    train = train_spawn_poses if train_spawn_poses is not None else data.get("train_spawn_poses")
    evaluation = eval_spawn_poses if eval_spawn_poses is not None else data.get("eval_spawn_poses")
    train = train if train is not None else evaluation
    evaluation = evaluation if evaluation is not None else train
    if not train or not evaluation:
        raise ValueError(f"{path.name} 缺少出生点；请提供同名 JSON，或使用 "
                         "--train-spawn / --eval-spawn X Y Z YAW（弧度）。")

    def poses(values):
        result = tuple(tuple(float(v) for v in pose) for pose in values)
        if any(len(pose) != 4 or not all(math.isfinite(v) for v in pose) for pose in result):
            raise ValueError("出生点应为四个有限数：x y z yaw（米、弧度）。")
        return result

    train, evaluation = poses(train), poses(evaluation)
    bounds = workspace_bounds if workspace_bounds is not None else data.get("workspace_bounds")
    bounds = None if disable_bounds or bounds is None else tuple(float(v) for v in bounds)
    if bounds is not None:
        if (len(bounds) != 4 or not all(math.isfinite(v) for v in bounds)
                or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]):
            raise ValueError("工作边界应为 xmin xmax ymin ymax，且最小值小于最大值。")
        if any(not (bounds[0] < x < bounds[1] and bounds[2] < y < bounds[3])
               for x, y, _, _ in train + evaluation):
            raise ValueError("出生点必须位于工作边界内。")
    return WorldSettings(path, train, evaluation, bounds)
