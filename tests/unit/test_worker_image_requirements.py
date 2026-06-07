from pathlib import Path


def test_worker_dockerfile_installs_git() -> None:
    dockerfile = Path("docker/worker.Dockerfile").read_text()

    assert " git" in dockerfile or "git " in dockerfile


def test_worker_requirements_include_ultralytics_clip_dependency() -> None:
    requirements = Path("requirements/worker.txt").read_text()

    assert "git+https://github.com/ultralytics/CLIP.git" in requirements
