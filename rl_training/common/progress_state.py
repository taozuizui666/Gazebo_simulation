"""Small atomic telemetry files; no changes to observations, actions or rewards."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import warnings

from rl_training.monitoring.progress_config import LOCAL_UPDATE_SEC


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


class LiveState:
    def __init__(self, path, **initial):
        self.path = Path(path)
        self.data = {"started_at": utc_now(), "pid": os.getpid(), **initial}
        self.last_write = 0.0
        self.warned = False

    def update(self, force=False, **fields):
        self.data.update(fields)
        now = time.monotonic()
        if not force and now - self.last_write < LOCAL_UPDATE_SEC:
            return
        self.data["updated_at"] = utc_now()
        try:
            atomic_json(self.path, self.data)
            self.last_write = now
        except OSError as error:
            # Progress reporting must not break vehicle control/training.
            if not self.warned:
                warnings.warn(f"进度文件暂时无法写入：{type(error).__name__}")
                self.warned = True
