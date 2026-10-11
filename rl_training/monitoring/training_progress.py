"""Own an optional progress publisher without stopping an existing service."""

from contextlib import contextmanager
import fcntl
from pathlib import Path
import subprocess
import sys

from . import progress_config as cfg


@contextmanager
def training_progress(enabled, publish):
    if not enabled:
        yield
        return
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    process = None
    try:
        with (cfg.OUTPUT_DIR / ".publisher.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("复用已有进度发布器；保持它原有的本地/GitHub 设置。", flush=True)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)
                command = [sys.executable, "-u", str(Path(__file__).with_name("publish_progress.py")),
                           "--serve", "--output-dir", str(cfg.OUTPUT_DIR),
                           "--models-dir", str(cfg.MODELS_DIR), "--host", cfg.LOCAL_HOST,
                           "--port", str(cfg.LOCAL_PORT)]
                if publish:
                    command.append("--publish")
                try:
                    process = subprocess.Popen(command, start_new_session=True)
                    try:
                        code = process.wait(timeout=1)
                        print(f"进度发布器已退出（{code}），请查看上方日志；训练继续。", flush=True)
                    except subprocess.TimeoutExpired:
                        pass
                except OSError as error:
                    print(f"进度发布器启动失败：{error}；训练继续。", file=sys.stderr, flush=True)
        yield
    finally:
        if process is not None and process.poll() is None:
            print("训练结束，等待进度发布器保存并同步最终状态。", flush=True)
            process.terminate()
            try:
                process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                print("最终同步超时，已停止本次发布器；请检查网络后手动同步。",
                      file=sys.stderr, flush=True)
        elif process is None:
            # The existing service remains independent; ask for its next refresh now.
            try:
                (cfg.OUTPUT_DIR / ".refresh").touch()
            except OSError as error:
                print(f"无法请求进度刷新：{error}", file=sys.stderr, flush=True)
