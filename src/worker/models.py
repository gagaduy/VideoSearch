from dataclasses import dataclass


@dataclass
class IndexedArtifacts:
    embedding: list[float]
    caption: str
    ocr: str
    objects: list[dict[str, object]]

