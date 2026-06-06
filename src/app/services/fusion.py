def fuse_scores(
    embedding_score: float,
    caption_score: float,
    ocr_score: float,
    object_score: float,
    temporal_score: float,
) -> float:
    return embedding_score + caption_score + ocr_score + object_score + temporal_score

