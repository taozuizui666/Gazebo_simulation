"""Bounded cleanup for an interrupted or failed parallel Gazebo rollout."""

from contextlib import suppress
import time
from stable_baselines3.common.vec_env import SubprocVecEnv

WORKER_SHUTDOWN_TIMEOUT_SEC = 45.0


class CruiseSubprocVecEnv(SubprocVecEnv):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            if hasattr(self, "processes"):
                self.close()
            raise

    def close(self):
        if getattr(self, "closed", False):
            return
        # A KeyboardInterrupt can occur after some step replies were consumed.
        # Never blindly recv() again from all workers during shutdown.
        for remote in self.remotes:
            with suppress(EOFError, BrokenPipeError, OSError):
                while remote.poll(0):
                    remote.recv()
                remote.send(("close", None))
        deadline = time.monotonic() + WORKER_SHUTDOWN_TIMEOUT_SEC
        for process in self.processes:
            process.join(timeout=max(0., deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for remote in self.remotes:
            remote.close()
        self.closed = True
