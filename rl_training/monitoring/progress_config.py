"""本地与手机进度页面的集中配置；无需 ROS 环境即可运行发布器。"""

from pathlib import Path

# ===================== 可修改配置区 =====================
SIMULATION_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = SIMULATION_ROOT / "rl_models"  # 下分 lidar / privileged，各自含 policies 和 logs
MODEL_MODES = ("lidar", "privileged")
OUTPUT_DIR = SIMULATION_ROOT / "rl_models" / "progress"
ASSETS_DIR = Path(__file__).with_name("dashboard")
REPOSITORY = SIMULATION_ROOT.parent
WORKTREE_DIR = SIMULATION_ROOT / "rl_models" / "runtime" / "publisher"
LOCAL_HOST = "0.0.0.0"  # 同一 Wi-Fi 的手机可访问；仅本机查看时改为 127.0.0.1
LOCAL_PORT = 8765
LOCAL_UPDATE_SEC = 5.0
REMOTE_PUBLISH_SEC = 60.0
REMOTE_POLL_SEC = 15.0
STALE_AFTER_SEC = 180.0
RECENT_EPISODES = 100
MAX_RUNS = 12
MAX_CHART_POINTS = 400
GIT_REMOTE = "origin"
PAGES_BRANCH = "gh-pages"
DATA_BRANCH = "rl-progress-data"
PAGES_URL = "https://taozuizui666.github.io/dissertation/"
RAW_DATA_BASE = "https://raw.githubusercontent.com/taozuizui666/dissertation/rl-progress-data/"
GIT_TIMEOUT_SEC = 45
# ===================== 配置区结束 =====================
