from __future__ import annotations

import json
from pathlib import Path

from app.db.session import get_session_factory, init_db
from worker.keyframe_importer import import_keyframe_dataset


def run(dataset_root: str | Path) -> dict[str, object]:
    init_db()
    session = get_session_factory()()
    try:
        payload = import_keyframe_dataset(session, dataset_root)
        session.commit()
        return payload
    finally:
        session.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import a pre-extracted keyframe dataset into the retrieval index.")
    parser.add_argument("dataset_root", help="Path to the dataset root containing keyframe/ and media-info*/")
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root), indent=2, sort_keys=True))
