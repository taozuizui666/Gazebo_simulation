#!/usr/bin/env python3
"""手机进度页：本地 5 秒快照、GitHub 独立数据分支、只读局域网服务。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rl_training.monitoring import progress_config as cfg
from rl_training.monitoring.progress_data import SnapshotBuilder
from rl_training.common.progress_state import atomic_json, utc_now

ASSETS = ("index.html", "app.js", "style.css")
MEDIA_FILES = {f"media/{mode}_{name}.png" for mode in cfg.MODEL_MODES
               for name in ("validation", "learning")} | {"media/validation.png", "media/learning.png"}
SERVED_FILES = {"/", "/index.html", "/app.js", "/style.css", "/config.json", "/data.json",
                *('/' + name for name in MEDIA_FILES)}


def copy_changed(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = source.read_bytes()
    if destination.exists() and destination.read_bytes() == content:
        return
    temporary = destination.with_name("." + destination.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)


def browser_config(remote=False):
    return {"mode": "github" if remote else "local",
            "data_url": cfg.RAW_DATA_BASE + "data.json" if remote else "data.json",
            "asset_base": cfg.RAW_DATA_BASE if remote else "",
            "fallback_url": "data.json" if remote else None,
            "poll_seconds": cfg.REMOTE_POLL_SEC if remote else cfg.LOCAL_UPDATE_SEC,
            "publish_seconds": cfg.REMOTE_PUBLISH_SEC if remote else cfg.LOCAL_UPDATE_SEC,
            "pages_url": cfg.PAGES_URL}


def asset_fingerprint(output):
    return hashlib.sha256(b"".join((output/name).read_bytes() for name in ASSETS)
                          + json.dumps(browser_config(remote=True), sort_keys=True).encode()).hexdigest()


def build_local(builder, output):
    output.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        copy_changed(cfg.ASSETS_DIR / name, output / name)
    config = browser_config()
    if not (output / "config.json").exists() or json.loads((output / "config.json").read_text()) != config:
        atomic_json(output / "config.json", config)
    data = builder.build()
    fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    previous = builder.read_json(output / "data.json")
    if previous.get("snapshot_id") != fingerprint:
        data.update(snapshot_id=fingerprint, generated_at=utc_now())
        atomic_json(output / "data.json", data)
    else:
        data = previous
    for relative, source in builder.media.items():
        copy_changed(source, output / relative)
    return data


class GitPublisher:
    """Only managed static files; detached worktrees; never force-push or switch source branches."""

    def __init__(self, repo=cfg.REPOSITORY, worktrees=cfg.WORKTREE_DIR, remote=cfg.GIT_REMOTE,
                 pages_branch=cfg.PAGES_BRANCH, data_branch=cfg.DATA_BRANCH):
        self.repo, self.worktrees, self.remote = Path(repo), Path(worktrees), remote
        self.pages_branch, self.data_branch = pages_branch, data_branch
        self.last_data = None
        self.last_assets = None
        # Preserve repository authentication/transport settings (e.g. SSH over
        # port 443); a default environment override would hide core.sshCommand.
        self.configured_ssh = subprocess.run(
            ["git", "config", "--get", "core.sshCommand"], cwd=self.repo,
            capture_output=True, text=True, timeout=cfg.GIT_TIMEOUT_SEC).stdout.strip()
        current = self.git(["branch", "--show-current"]).stdout.strip()
        if pages_branch == data_branch or current in (pages_branch, data_branch):
            raise ValueError("页面/数据分支必须互相独立，并与当前源码分支不同。")
        for branch in (pages_branch, data_branch):
            self.git(["check-ref-format", "--branch", branch])

    def git(self, args, cwd=None, check=True, input=None):
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        if not self.configured_ssh:
            env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=10")
        result = subprocess.run(["git", *args], cwd=cwd or self.repo, env=env, input=input,
                                capture_output=True, text=True, timeout=cfg.GIT_TIMEOUT_SEC)
        if check and result.returncode:
            raise RuntimeError(f"Git {' '.join(args[:2])} 失败：{result.stderr.strip()[:700]}")
        return result

    def worktree(self, branch, folder):
        path = self.worktrees / folder
        found = self.git(["ls-remote", "--heads", self.remote, f"refs/heads/{branch}"]).stdout.strip()
        reference = f"refs/remotes/{self.remote}/{branch}"
        if found:
            self.git(["fetch", "--no-tags", self.remote, f"refs/heads/{branch}:{reference}"])
        if path.exists():
            if not (path / ".git").is_file():
                raise RuntimeError(f"工作目录已被其他文件占用：{path}")
            common = self.git(["rev-parse", "--git-common-dir"], cwd=path).stdout.strip()
            expected = self.git(["rev-parse", "--git-common-dir"]).stdout.strip()
            if (path / common).resolve() != (self.repo / expected).resolve():
                raise RuntimeError("发布工作区属于另一个仓库。")
            if self.git(["status", "--porcelain"], cwd=path).stdout.strip():
                raise RuntimeError(f"发布工作区有未提交修改，已保留，请先检查：{path}")
            if found:
                self.git(["merge", "--ff-only", reference], cwd=path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if found:
                start = reference
            else:
                tree = self.git(["hash-object", "-t", "tree", "-w", "--stdin"], input="").stdout.strip()
                start = self.git(["commit-tree", tree, "-m", f"Initialize {branch} for RL progress"]).stdout.strip()
            self.git(["worktree", "add", "--detach", str(path), start])
        return path

    def commit_push(self, path, branch, names, message):
        self.git(["add", "--", *names], cwd=path)
        if self.git(["diff", "--cached", "--quiet"], cwd=path, check=False).returncode:
            self.git(["commit", "-m", message], cwd=path)
        self.git(["push", self.remote, f"HEAD:refs/heads/{branch}"], cwd=path)

    def publish(self, output, data, deploy=False):
        media = [name for name in data.get("media", {}) if name in MEDIA_FILES]
        if data["snapshot_id"] != self.last_data:
            worktree = self.worktree(self.data_branch, "data")
            for name in ["data.json", *media]:
                copy_changed(output / name, worktree / name)
            self.commit_push(worktree, self.data_branch, ["data.json", *media],
                             f"RL progress {data['snapshot_id'][:12]} [skip ci]")
            self.last_data = data["snapshot_id"]
        remote_config = browser_config(remote=True)
        assets_hash = asset_fingerprint(output)
        if self.last_assets != assets_hash or deploy:
            worktree = self.worktree(self.pages_branch, "pages")
            marker = worktree / "dashboard-version.txt"
            needs_deploy = deploy or not marker.exists() or marker.read_text().strip() != assets_hash
            if needs_deploy:
                for name in [*ASSETS, "data.json", *media]:
                    copy_changed(output / name, worktree / name)
                atomic_json(worktree / "config.json", remote_config)
                (worktree / ".nojekyll").write_text("")
                marker.write_text(assets_hash + "\n")
                self.commit_push(worktree, self.pages_branch,
                                 [*ASSETS, "data.json", *media, "config.json", ".nojekyll", marker.name],
                                 "Update mobile RL progress dashboard")
            else:
                self.git(["push", self.remote, f"HEAD:refs/heads/{self.pages_branch}"], cwd=worktree)
            self.last_assets = assets_hash


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if urlsplit(self.path).path not in SERVED_FILES:
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):
        if urlsplit(self.path).path not in SERVED_FILES:
            self.send_error(404)
            return
        super().do_HEAD()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, *args):
        pass


def stage_snapshot(output, data):
    """Capture one consistent generation before asynchronous Git operations."""
    stage = tempfile.TemporaryDirectory(prefix=".upload-", dir=output)
    try:
        path = Path(stage.name)
        atomic_json(path / "data.json", data)
        for name in [*ASSETS, *data.get("media", {})]:
            copy_changed(output / name, path / name)
        return stage
    except BaseException:
        stage.cleanup()
        raise


def publish_staged(publisher, stage, data, deploy):
    try:
        publisher.publish(Path(stage.name), data, deploy=deploy)
    finally:
        stage.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="持续更新并提供只读网页，手机可通过本机局域网 IP 访问")
    parser.add_argument("--watch", action="store_true", help="持续生成快照，不启动 HTTP 服务")
    parser.add_argument("--publish", action="store_true", help="同步 GitHub 进度数据，并在网页代码变化时部署 Pages")
    parser.add_argument("--deploy", action="store_true", help="同时更新 Pages 的备用快照，需要 --publish")
    parser.add_argument("--host", default=cfg.LOCAL_HOST)
    parser.add_argument("--port", type=int, default=cfg.LOCAL_PORT)
    parser.add_argument("--models-dir", type=Path, default=cfg.MODELS_DIR,
                        help="包含 lidar/ 和 privileged/ 的模型总目录")
    parser.add_argument("--output-dir", type=Path, default=cfg.OUTPUT_DIR)
    args = parser.parse_args()
    if args.deploy and not args.publish:
        parser.error("--deploy 需要 --publish")
    if not 1 <= args.port <= 65535:
        parser.error("port 必须在 1～65535 之间")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / ".publisher.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.exit(1, "该输出目录已有进度服务在运行。\n")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    builder = SnapshotBuilder(args.models_dir)
    publisher = GitPublisher() if args.publish else None
    server = None
    next_publish = 0.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="github-progress") if publisher else None
    pending = None
    publishing_data = None
    try:
        data = build_local(builder, args.output_dir)
        if args.serve:
            server = ThreadingHTTPServer((args.host, args.port), partial(DashboardHandler, directory=str(args.output_dir)))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print(f"本地页面：http://127.0.0.1:{args.port}；同 Wi-Fi 手机使用本机局域网 IP。", flush=True)
        print(f"本地快照：{args.output_dir / 'index.html'}", flush=True)
        if publisher:
            print(f"GitHub 页面：{cfg.PAGES_URL}；数据每 {cfg.REMOTE_PUBLISH_SEC:g} 秒检查同步。", flush=True)
        while not stop.is_set():
            refresh = args.output_dir / ".refresh"
            if refresh.exists():
                refresh.unlink(missing_ok=True)
                next_publish = 0.
            data = build_local(builder, args.output_dir)
            if pending is not None and pending.done():
                try:
                    pending.result()
                    args.deploy = False
                    print(f"GitHub 已同步快照 {publishing_data['snapshot_id'][:12]}，快照时间 {publishing_data['generated_at']}", flush=True)
                except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                    print(f"同步暂未成功，本地继续更新，稍后重试：{error}", file=sys.stderr, flush=True)
                pending = None
            if publisher and pending is None and time.monotonic() >= next_publish:
                if not (args.watch or args.serve):
                    publisher.publish(args.output_dir, data, deploy=args.deploy)
                    print(f"GitHub 已同步快照 {data['snapshot_id'][:12]}", flush=True)
                elif (publisher.last_data != data['snapshot_id'] or args.deploy
                      or publisher.last_assets != asset_fingerprint(args.output_dir)):
                    publishing_data = data
                    pending = executor.submit(publish_staged, publisher, stage_snapshot(args.output_dir, data), data, args.deploy)
                next_publish = time.monotonic() + cfg.REMOTE_PUBLISH_SEC
            if not (args.watch or args.serve):
                break
            stop.wait(cfg.LOCAL_UPDATE_SEC)
    finally:
        if executor:
            executor.shutdown(wait=True)
        if stop.is_set():
            try:
                data = build_local(builder, args.output_dir)
                if publisher:
                    publisher.publish(args.output_dir, data, deploy=args.deploy)
                print("最终进度已同步。" if publisher else "最终本地进度已保存。", flush=True)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                print(f"最终进度同步失败：{error}；可稍后手动运行发布器重试。",
                      file=sys.stderr, flush=True)
        if server:
            server.shutdown()
            server.server_close()
        lock.close()


if __name__ == "__main__":
    main()
