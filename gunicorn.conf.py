import sys

bind = "0.0.0.0:9000"
workers = 4
threads = 2
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 1800
keepalive = 1800

accesslog = "./access.log"
errorlog = "./error.log"
loglevel = "info"

daemon = False
pidfile = "detect_gunicorn.pid"

preload_app = True

sys.stdout = sys.stderr