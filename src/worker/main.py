from time import sleep

from app.config import settings
from app.db.session import init_db
from worker.tasks import run_pending_jobs


def main() -> None:
    init_db()
    interval = max(int(getattr(settings, "worker_poll_interval_sec", 2)), 1)
    while True:
        run_pending_jobs()
        sleep(interval)


if __name__ == "__main__":
    main()
