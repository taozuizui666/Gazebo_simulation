#!/usr/bin/env python3
"""Freeze a model only after matching long, multi-start evaluations pass."""

import argparse
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl_training import config

# 发布到本地已验证目录的门槛；不会上传或生成随机森林数据。
MIN_EPISODES_PER_REPORT = 8
MIN_SCENARIOS_PER_REPORT = 8
MIN_EPISODE_SIM_SECONDS = 300.0
MIN_SPEED_COMPLIANCE = 0.95
LATEST_VALIDATED_NAME = "latest_validated"


def qualify(model_path, report_paths, output_dir):
    latest = output_dir / LATEST_VALIDATED_NAME
    if latest.exists() and not latest.is_symlink():
        raise ValueError(f"固定模型入口已被普通文件/目录占用：{latest}")
    model_bytes = model_path.read_bytes()
    digest = hashlib.sha256(model_bytes).hexdigest()
    reports = []
    for path in report_paths:
        report = json.loads(path.read_text())
        episodes = report["episodes"]
        if report["model_sha256"] != digest:
            raise ValueError(f"{path}: 报告与待保存的模型不是同一份权重。")
        if (not report["passed"] or len(episodes) < MIN_EPISODES_PER_REPORT
                or len({e["scenario"] for e in episodes}) < MIN_SCENARIOS_PER_REPORT):
            raise ValueError(f"{path}: 多起点验证未达标。")
        for entry in episodes:
            if (entry["reason"] != "time_limit" or not entry["success"]
                    or entry["sim_duration_sec"] < MIN_EPISODE_SIM_SECONDS - 1e-6
                    or entry["speed_compliance"] < MIN_SPEED_COMPLIANCE
                    or any(entry[key] for key in ("collided", "stuck", "looping", "slow", "out_of_bounds"))):
                raise ValueError(f"{path}: 巡航时长、速度或失败次数不符合门槛。")
        reports.append(report)
    if not reports:
        raise ValueError("至少提供一份独立验证报告。")
    with ZipFile(BytesIO(model_bytes)) as archive:
        steps = int(json.loads(archive.read("data"))["num_timesteps"])
    source = model_path.parent
    metadata = json.loads((source / "run_config.json").read_text())
    # Reward normalization is not used for inference, but retain the matching
    # state for an actual training continuation.
    normalizer = source / f"rl_policy_vecnormalize_{steps}_steps.pkl"
    if not normalizer.exists():
        normalizer = source / ("best_vecnormalize.pkl" if model_path.name == "best_model.zip" else "vecnormalize.pkl")
    if not normalizer.exists():
        raise FileNotFoundError(f"缺少配套续训状态：{normalizer}")
    name = f"validated_{steps}_{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%d_%H%M%S}"
    directory = output_dir / name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "model.zip").write_bytes(model_bytes)
    shutil.copy2(normalizer, directory / "vecnormalize.pkl")
    for filename in ("vehicle_config.h", "training_config.py", "training_curve.png", "training_curve_data.json"):
        if (source / filename).exists():
            shutil.copy2(source / filename, directory / filename)
    if (source / "source_snapshot").exists():
        shutil.copytree(source / "source_snapshot", directory / "source_snapshot")
    metadata.update(status="validated", training_timesteps=steps,
                    saved_model="model.zip", source_model=str(model_path.resolve()),
                    model_sha256=digest)
    (directory / "run_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    for index, path in enumerate(report_paths, 1):
        target = directory / f"validation_{index}"
        target.mkdir()
        for name in ("evaluation.json", "trajectories.png", "trajectories.npz"):
            if (path.parent / name).exists():
                shutil.copy2(path.parent / name, target / name)
    episodes = [entry for report in reports for entry in report["episodes"]]
    summary = {
        "model_sha256": digest, "training_timesteps": steps, "episodes": len(episodes),
        "simulation_seconds": sum(e["sim_duration_sec"] for e in episodes),
        "distance_m": sum(e["distance_m"] for e in episodes),
        "min_episode_speed_compliance": min(e["speed_compliance"] for e in episodes),
        "min_episode_mean_speed_mps": min(e["mean_speed_mps"] for e in episodes),
        "max_episode_mean_speed_mps": max(e["mean_speed_mps"] for e in episodes),
        "collisions": 0, "stalls": 0, "small_loops": 0, "slow_failures": 0, "out_of_bounds": 0,
        "scope": "Only the recorded worlds, vehicle configuration, starts, noise and durations were tested.",
        "source_reports": [str(p.resolve()) for p in report_paths],
    }
    (directory / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    # Keep a stable local inference path, while preserving every frozen model.
    temporary = output_dir / f".{LATEST_VALIDATED_NAME}_{uuid4().hex}.tmp"
    temporary.symlink_to(directory.name, target_is_directory=True)
    temporary.replace(latest)
    print(f"已冻结通过验证的模型：{directory}")
    print(f"固定入口：{latest / 'model.zip'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=config.POLICY_DIR)
    args = parser.parse_args()
    qualify(args.model, args.report, args.output_dir)


if __name__ == "__main__":
    main()
