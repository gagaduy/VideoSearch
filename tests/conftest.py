import os
from pathlib import Path
import sys

TEST_DB_PATH = Path(__file__).resolve().parents[1] / "test.db"
os.environ.setdefault("DATABASE_URL", f"sqlite:///{TEST_DB_PATH}")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if TEST_DB_PATH.exists():
    TEST_DB_PATH.unlink(missing_ok=True)

from app.db.session import init_db  # noqa: E402

init_db()
