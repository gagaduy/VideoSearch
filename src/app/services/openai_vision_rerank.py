from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

from openai import OpenAI

from app.config import settings


def select_rerank_candidates(results: list[dict[str, object]], top_k: int) -> list[dict[str, object]]:
    return list(results[:top_k])


def blend_rerank_score(*, local_score: float, vision_score: float, local_weight: float, vision_weight: float) -> float:
    return round((local_score * local_weight) + (vision_score * vision_weight), 3)


def should_run_openai_vision_rerank(*, enabled: bool, api_key: str, candidates: list[dict[str, object]]) -> bool:
    return bool(enabled and api_key and candidates)


def parse_vision_rerank_scores(payload: dict[str, object]) -> dict[int, float]:
    items = payload.get("items", [])
    scores: dict[int, float] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        frame_id = item.get("frame_id")
        vision_score = item.get("vision_score")
        if isinstance(frame_id, int) and isinstance(vision_score, (int, float)):
            scores[int(frame_id)] = float(vision_score)
    return scores


def _image_content(image_path: Path) -> list[dict[str, str]]:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return [{"type": "input_image", "image_url": f"data:{mime_type};base64,{image_b64}"}]


def _request_openai_vision_scores(
    query: str,
    candidates: list[dict[str, object]],
    query_image_paths: list[Path] | None = None,
    instruction: str | None = None,
) -> dict[str, object]:
    client = OpenAI(api_key=settings.openai_api_key)
    prompt = instruction or (
        "You are reranking video retrieval frame candidates. "
        "Given the user query and the candidate frame images, score each frame from 0.0 to 1.0. "
        "Prefer exact object presence, count, relationships, and action matches. "
        "Return JSON only with the shape "
        '{"items":[{"frame_id":123,"vision_score":0.91}]}. '
        f"User query: {query}"
    )
    content: list[dict[str, str]] = [
        {
            "type": "input_text",
            "text": prompt,
        }
    ]
    for index, image_path in enumerate(query_image_paths or [], start=1):
        if not image_path.exists():
            continue
        content.append({"type": "input_text", "text": f"Query frame {index}"})
        content.extend(_image_content(image_path))
    for candidate in candidates:
        frame_id = int(candidate.get("frame_id") or -1)
        image_path = Path(str(candidate.get("_image_path", "")).strip())
        if frame_id <= 0 or not image_path.exists():
            continue
        content.append({"type": "input_text", "text": f"Candidate frame_id={frame_id}"})
        content.extend(_image_content(image_path))

    response = client.with_options(timeout=settings.openai_vision_rerank_timeout_sec).responses.create(
        model=settings.openai_vision_rerank_model,
        input=[{"role": "user", "content": content}],
    )
    text = response.output_text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {"items": []}
    return json.loads(text[start : end + 1])


def run_openai_vision_rerank(
    query: str,
    candidates: list[dict[str, object]],
    query_image_paths: list[Path] | None = None,
    instruction: str | None = None,
) -> dict[int, float]:
    try:
        payload = _request_openai_vision_scores(
            query,
            candidates,
            query_image_paths=query_image_paths,
            instruction=instruction,
        )
    except Exception:
        return {}
    return parse_vision_rerank_scores(payload)
