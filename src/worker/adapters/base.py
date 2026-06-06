from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    model_name: str
    values: list[float]

