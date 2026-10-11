"""Read training artifacts into a small, public, path-free dashboard snapshot."""

from collections import Counter, deque
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import time

from . import progress_config as cfg
from rl_training.common.paths import training_results_dir

FAILURES = ("collision", "stuck", "looping", "slow", "out_of_bounds")
TAGS = {
    "reward": "rollout/ep_rew_mean", "completion": "cruise/completion_rate",
    "collision": "cruise/collision_rate", "stuck": "cruise/stuck_rate",
    "looping": "cruise/loop_rate", "speed": "cruise/mean_speed_mps",
    "duration": "cruise/sim_duration_sec", "value_loss": "train/value_loss",
    "kl": "train/approx_kl", "entropy": "train/entropy_loss",
    "explained_variance": "train/explained_variance",
}


def number(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def epoch(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (AttributeError, TypeError, ValueError):
        return 0.


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value else None


def episode_stats(episodes):
    """Completed time limits are right-censored, never invented failure times."""
    counts = Counter(e["reason"] for e in episodes)
    by_reason = {}
    for reason in FAILURES:
        events = [e for e in episodes if e["reason"] == reason]
        durations = [e["sim_duration_sec"] for e in events]
        by_reason[reason] = {
            "count": len(events), "mean_sec": statistics.mean(durations) if durations else None,
            "min_sec": min(durations) if durations else None,
            "last": events[-1] if events else None,
        }
    censored = [e["sim_duration_sec"] for e in episodes if e["reason"] == "time_limit"]
    return {
        "count": len(episodes), "counts": {k: counts[k] for k in (*FAILURES, "time_limit")},
        "failure_count": sum(counts[k] for k in FAILURES), "failures": by_reason,
        "completion_rate": counts["time_limit"] / len(episodes) if episodes else None,
        "longest_observed_sec": max((e["sim_duration_sec"] for e in episodes), default=None),
        "longest_completed_sec": max(censored, default=None),
        "total_sim_sec": sum(e["sim_duration_sec"] for e in episodes),
        "total_distance_m": sum(e.get("distance_m", 0.) for e in episodes),
        "mean_speed_mps": statistics.mean(e["mean_speed_mps"] for e in episodes
                                           if e.get("mean_speed_mps") is not None)
                          if any(e.get("mean_speed_mps") is not None for e in episodes) else None,
    }


def public_episode(entry, **extra):
    duration = number(entry.get("sim_duration_sec", entry.get("episode_sim_time_sec")))
    reason = entry.get("reason", entry.get("termination_reason"))
    if duration is None or duration < 0 or reason not in (*FAILURES, "time_limit"):
        return None
    return {
        "reason": reason, "sim_duration_sec": duration,
        "distance_m": number(entry.get("distance_m", entry.get("episode_distance_m")), 0.),
        "mean_speed_mps": number(entry.get("mean_speed_mps", entry.get("episode_mean_speed_mps"))),
        "speed_compliance": number(entry.get("speed_compliance")),
        "reward": number(entry.get("reward", entry.get("r"))),
        "scenario": entry.get("scenario"),
        "world": Path(str(entry.get("world", ""))).name,
        "spawn_index": number(entry.get("spawn_index")), **extra,
    }


class SnapshotBuilder:
    def __init__(self, models_dir=cfg.MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.cache = {}
        self.media = {}

    def cached(self, path, reader, default):
        path = Path(path)
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            key = (str(path), reader.__name__)
            if key not in self.cache or self.cache[key][0] != signature:
                self.cache[key] = (signature, reader(path))
            return self.cache[key][1]
        except (OSError, ValueError, KeyError, TypeError, csv.Error):
            return default

    def read_json(self, path):
        def read_json_file(p):
            result = json.loads(p.read_text())
            return result if isinstance(result, dict) else {}
        return self.cached(path, read_json_file, {})

    def monitor(self, path):
        def read_monitor(p):
            text = p.read_text()
            # Ignore a partial last row while Monitor is appending it.
            text = text[:text.rfind("\n") + 1]
            lines = text.splitlines()
            if len(lines) < 2 or not lines[0].startswith("#"):
                return [], 0
            started = json.loads(lines[0][1:]).get("t_start", 0)
            rows = deque(maxlen=max(1000, cfg.RECENT_EPISODES))
            count = 0
            for index, row in enumerate(csv.DictReader(io.StringIO("\n".join(lines[1:]))), 1):
                entry = public_episode(row, number=index, worker=p.parent.name,
                                       ended_at=iso(started + number(row.get("t"), 0.)))
                if entry:
                    rows.append(entry)
                    count += 1
            return list(rows), count
        return self.cached(path, read_monitor, ([], 0))

    def curves(self, run_id, mode):
        paths = sorted((self.models_dir / mode / "logs").glob(run_id + "_*/events.out.tfevents.*"))
        if not paths:
            return {}
        key = ("curves", mode, run_id)
        signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
        if self.cache.get(key, (None,))[0] == signature:
            return self.cache[key][1]
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            merged = {name: {} for name in TAGS}
            for directory in sorted({p.parent for p in paths}):
                ea = EventAccumulator(str(directory), size_guidance={"scalars": 0})
                ea.Reload()
                available = set(ea.Tags().get("scalars", []))
                for name, tag in TAGS.items():
                    if tag in available:
                        for point in ea.Scalars(tag):
                            if math.isfinite(point.value):
                                merged[name][point.step] = point.value
            result = {}
            for name, points in merged.items():
                rows = sorted(points.items())
                if len(rows) > cfg.MAX_CHART_POINTS:
                    rows = [rows[round(i * (len(rows)-1)/(cfg.MAX_CHART_POINTS-1))]
                            for i in range(cfg.MAX_CHART_POINTS)]
                result[name] = rows
            self.cache[key] = (signature, result)
            return result
        except (ImportError, OSError, ValueError):
            return {}

    def live(self, path, worker):
        data = self.read_json(path)
        if not data:
            return None
        allowed = ("updated_at", "phase", "episode_number", "max_episode_sim_sec", "sim_duration_sec",
                   "distance_m", "speed_mps", "action", "step", "reason", "speed_compliance", "error_type",
                   "world", "spawn_index", "spawn_pose")
        return {"worker": worker, **{k: data[k] for k in allowed if k in data}}

    def training_run(self, path):
        meta = self.read_json(path / "run_config.json")
        status = self.read_json(path / "training_status.json")
        if not meta:
            return None
        rows, total = [], 0
        for monitor in sorted(path.glob("worker_*/episodes.monitor.csv")):
            entries, count = self.monitor(monitor)
            rows.extend(entries)
            total += count
        rows.sort(key=lambda e: (e.get("ended_at") or "", e["worker"], e["number"]))
        rows = rows[-cfg.RECENT_EPISODES:]
        workers = [s for p in sorted(path.glob("worker_*/live_state.json"))
                   if (s := self.live(p, p.parent.name))]
        periodic = self.live(path / "evaluation_server/live_state.json", "定期评估")
        campaign = meta.get("multi_map") or {}
        suite = self.read_json(path / "evaluation/multi_map.json")
        suite_episodes = [e for row in suite.get("episodes", []) if (e := public_episode(row))]
        map_evaluation = [{"world": Path(w).name,
                           "training_steps": suite.get("training_steps"),
                           "updated_at": suite.get("updated_at"),
                           "stats": episode_stats([e for e in suite_episodes if e["world"] == Path(w).name])}
                          for w in campaign.get("eval_worlds", [])]
        terminal = meta.get("status") not in ("starting", "training")
        updated = meta.get("ended_at") if terminal else max(
            [meta.get("started_at", ""), status.get("updated_at", "")]
            + [w.get("updated_at", "") for w in workers]
            + ([periodic.get("updated_at", "")] if periodic else []), key=epoch)
        running = not terminal and time.time() - epoch(updated) < cfg.STALE_AFTER_SEC
        args = meta.get("arguments", {})
        vehicle = meta.get("vehicle", {})
        environment = meta.get("environment", {})
        steps = meta.get("training_timesteps", status.get("training_timesteps", 0)) if terminal else status.get("training_timesteps", 0)
        return {
            "id": path.name, "started_at": meta.get("started_at"), "ended_at": meta.get("ended_at"),
            "observation_mode": meta.get("observation_mode", "lidar"),
            "observation_spec": meta.get("observation_spec"),
            "status": meta.get("status") if terminal or running else "stale", "is_running": running,
            "updated_at": updated, "phase": status.get("phase", "finished" if terminal else "collecting"),
            "steps": steps, "start_steps": meta.get("start_timesteps", 0),
            "target_steps": meta.get("target_total_timesteps", meta.get("requested_timesteps")),
            "elapsed_wall_sec": max(0, epoch(meta.get("ended_at"))-epoch(meta.get("started_at")))
                                if terminal else status.get("elapsed_wall_sec"),
            "time_budget_sec": status.get("time_budget_sec", number(args.get("hours"), 0.) * 3600),
            "remaining_wall_sec": status.get("remaining_wall_sec") if not terminal else None,
            "training_deadline": status.get("training_deadline"),
            "samples_per_sec": status.get("steps_per_second"), "total_episodes": total,
            "episode_limit_sec": environment.get("max_steps", 0) * math.ceil(
                (environment.get("step_sim_sec", .2) if meta.get("observation_mode") == "privileged" else
                 max(environment.get("step_sim_sec", .2), 1 / vehicle.get("lidar_scan_hz", 5.5)
                     + vehicle.get("sim_time_step_sec", .001))) / vehicle.get("sim_time_step_sec", .001))
                * vehicle.get("sim_time_step_sec", .001),
            "num_envs": args.get("num_envs"), "world": Path(str(args.get("world", ""))).name,
            "train_worlds": [Path(w).name for w in campaign.get("train_worlds", [])],
            "eval_worlds": [Path(w).name for w in campaign.get("eval_worlds", [])],
            "map_evaluation": map_evaluation,
            "vehicle": vehicle, "environment": environment, "ppo": meta.get("ppo", {}),
            "seed": args.get("seed"), "device": args.get("device", "cpu"), "rollout_steps": args.get("rollout_steps"),
            "error_type": "TrainingError" if meta.get("error") else None,
            "recent": episode_stats(rows), "episodes": list(reversed(rows)), "workers": workers,
            "periodic_evaluation": periodic, "curves": self.curves(path.name, meta.get("observation_mode", "lidar")),
        }

    def validation(self, mode):
        path = self.models_dir / mode / "policies" / "latest_validated"
        summary = self.read_json(path / "validation_summary.json")
        meta = self.read_json(path / "run_config.json")
        if not summary or not meta:
            return None
        def model_hash(p):
            return hashlib.sha256(p.read_bytes()).hexdigest()
        digest = self.cached(path / "model.zip", model_hash, None)
        episodes = []
        reports = []
        for report_path in sorted(path.glob("validation_*/evaluation.json")):
            report = self.read_json(report_path)
            if report.get("model_sha256") != digest:
                continue
            reports.append({"world": Path(str(report.get("world", ""))).name,
                            **{k: report.get(k) for k in ("seed", "pose_jitter_m", "pose_jitter_deg", "passed")}})
            for entry in report.get("episodes", []):
                item = public_episode(entry, number=len(episodes)+1, worker="独立验收")
                if item:
                    episodes.append(item)
        matches = bool(digest and digest == summary.get("model_sha256") and episodes
                       and all(r["passed"] for r in reports)
                       and len(episodes) == summary.get("episodes"))
        result_path = training_results_dir(path)
        for name, source in (("validation.png", result_path / "validation_1/trajectories.png"),
                             ("learning.png", result_path / "training_curve.png")):
            if source.is_file():
                self.media[f"media/{mode}_{name}"] = source
        stats = episode_stats(episodes)
        return {
            "id": path.resolve().name, "verified": matches, "sha256": digest,
            "observation_mode": meta.get("observation_mode", "lidar"),
            "observation_spec": meta.get("observation_spec"),
            "training_run": meta.get("run_id"), "steps": summary.get("training_timesteps"),
            "updated_at": iso((path / "validation_summary.json").stat().st_mtime),
            "stats": stats, "episodes": episodes, "reports": reports,
            "min_completed_sec": min((e["sim_duration_sec"] for e in episodes), default=None),
            "min_speed_compliance": summary.get("min_episode_speed_compliance"),
            "min_mean_speed_mps": summary.get("min_episode_mean_speed_mps"),
            "max_mean_speed_mps": summary.get("max_episode_mean_speed_mps"),
            "vehicle": meta.get("vehicle", {}), "ppo": meta.get("ppo", {}),
            "trajectory_metrics": {k:v for k,v in self.read_json(path / "trajectory_metrics.json").items()
                                   if k != "episodes"},
        }

    def evaluations(self):
        evaluations = []
        paths = [p for mode in cfg.MODEL_MODES for p in sorted(
            (self.models_dir / mode / "policies").glob("evaluation_*/evaluation_status.json"),
            reverse=True)[:cfg.MAX_RUNS]]
        for path in sorted(paths, key=lambda p: p.parent.name, reverse=True):
            data = self.read_json(path)
            live = self.live(path.parent / "live_state.json", "独立评估")
            updated = max([data.get("updated_at", ""), (live or {}).get("updated_at", "")], key=epoch)
            running = data.get("status") == "running" and time.time()-epoch(updated) < cfg.STALE_AFTER_SEC
            entries = [e for row in data.get("episodes", []) if (e := public_episode(row))]
            evaluations.append({
                "id": path.parent.name, "status": data.get("status") if data.get("status") != "running" or running else "stale",
                "observation_mode": data.get("observation_mode", "lidar"),
                "is_running": running, "updated_at": updated, "model_name": data.get("model_name"),
                "model_sha256": data.get("model_sha256"), "model_run": data.get("model_run"),
                "completed_episodes": len(entries), "total_episodes": data.get("total_episodes"),
                "stats": episode_stats(entries), "live": live, "passed": data.get("passed"),
                "error_type": data.get("error_type"),
            })
        return evaluations

    def build(self):
        self.media = {}
        candidates = [p for mode in cfg.MODEL_MODES for p in sorted(
            (self.models_dir / mode / "policies").glob("PPO_*/run_config.json"),
            key=lambda p: p.stat().st_mtime, reverse=True)[:cfg.MAX_RUNS]]
        runs = [r for p in candidates if (r := self.training_run(p.parent))]
        runs.sort(key=lambda r: (r["is_running"], epoch(r.get("started_at"))), reverse=True)
        validations = {mode: self.validation(mode) for mode in cfg.MODEL_MODES}
        validation = validations.get("privileged") or validations.get("lidar")
        # Retain schema-1 defaults during browser/CDN cache transitions.
        if validation:
            mode = validation["observation_mode"]
            for name in ("validation.png", "learning.png"):
                source = self.media.get(f"media/{mode}_{name}")
                if source:
                    self.media[f"media/{name}"] = source
        return {"schema_version": 1, "task": "持续避障巡航", "runs": runs,
                "default_run_id": runs[0]["id"] if runs else None,
                "validated_model": validation, "validated_models": validations,
                "evaluations": self.evaluations(),
                "recent_window": cfg.RECENT_EPISODES, "stale_after_sec": cfg.STALE_AFTER_SEC,
                "media": {k: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for k,p in self.media.items()}}
