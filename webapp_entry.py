"""
Web UI entry point - starts the local Flask server and opens the browser.
Packaged as BuddyADWeb.exe alongside BuddyAD.exe.
"""
import os
import sys
import time
import socket
import logging
from logging.handlers import RotatingFileHandler
import threading
import webbrowser
from datetime import datetime

FROZEN = bool(getattr(sys, "frozen", False)) or "__compiled__" in globals()
if FROZEN:
    BASE_DIR = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
    DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.dirname(sys.executable)), "BuddyAd")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)

# Logs rotate at 5MB per file (3 backups: .1/.2/.3); anything older than 30
# days is deleted on startup so the log directory cannot grow without bound.
def _cleanup_old_logs(log_dir, max_age_days=30):
    try:
        cutoff = datetime.now().timestamp() - max_age_days * 86400
        for name in os.listdir(log_dir):
            if ".log" not in name.lower():
                continue
            path = os.path.join(log_dir, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except OSError:
                pass
    except OSError:
        pass


_cleanup_old_logs(DATA_DIR)

# RotatingFileHandler: 5MB per file, 3 backups (.1/.2/.3), then 30-day age
# cleanup above retires stale files.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            os.path.join(DATA_DIR, 'shot_by_shot_web.log'),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding='utf-8',
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _excepthook(exc_type, exc_value, exc_tb):
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _excepthook


def _pick_port(preferred=5000):
    """Use the preferred port when free, otherwise let the OS assign one."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", preferred))
        return preferred
    except OSError:
        s.close()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        try:
            s.close()
        except OSError:
            pass


def _open_browser_when_ready(url):
    if os.environ.get("SBS_NO_BROWSER") == "1":
        logger.info(f"Server ready at {url} (browser opening suppressed by SBS_NO_BROWSER)")
        return
    import urllib.request
    for _ in range(240):
        try:
            urllib.request.urlopen(url, timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    logger.info(f"Opening browser: {url}")
    webbrowser.open(url)


def main():
    port = _pick_port(int(os.environ.get("SBS_PORT", "5000")))
    url = f"http://127.0.0.1:{port}"
    logger.info(f"Starting Shot-by-Shot web server at {url}")

    threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    from app import app
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Fatal: {e}")
        print(f"Error starting web server: {e}")
        print("See the log at", os.path.join(DATA_DIR, "shot_by_shot_web.log"))
