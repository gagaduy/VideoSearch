import hashlib
from pathlib import Path


def _byte_signature(frame_path: Path) -> str:
    return hashlib.sha1(frame_path.read_bytes()).hexdigest()


def _phash_signature(frame_path: Path) -> str:
    try:
        import imagehash
        from PIL import Image

        with Image.open(frame_path) as image:
            return str(imagehash.phash(image))
    except Exception:
        return _byte_signature(frame_path)


def _signature_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        return 0 if left == right else max(len(left), len(right))
    return sum(1 for left_char, right_char in zip(left, right, strict=False) if left_char != right_char)


def keep_distinct_frames(frames: list[Path], distance_threshold: int) -> list[Path]:
    if not frames:
        return []
    kept = [frames[0]]
    previous_signature = _phash_signature(frames[0])
    for frame_path in frames[1:]:
        current_signature = _phash_signature(frame_path)
        if _signature_distance(previous_signature, current_signature) > distance_threshold:
            kept.append(frame_path)
            previous_signature = current_signature
    return kept
