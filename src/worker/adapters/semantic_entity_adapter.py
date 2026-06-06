from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from worker.adapters.internvl_adapter import InternvlAdapter


class SemanticEntityAdapter:
    def __init__(self, model_name: str = "gpt-4.1-mini") -> None:
        self.model_name = model_name
        self._local_adapter = InternvlAdapter()

    def _extract_json_payload(self, text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("no json object found")
        return json.loads(text[start : end + 1])

    def extract(self, image_path: str, caption_text: str, ocr_text: str) -> dict[str, object]:
        try:
            local = self._local_adapter.describe_image(image_path)
            local_entities = list(local.get("entities", []))
            counts: dict[str, int] = {}
            normalized_entities: list[dict[str, object]] = []
            for item in local_entities:
                label = str(item.get("label", "")).strip().lower()
                if not label:
                    continue
                aliases = [str(alias).strip().lower() for alias in item.get("aliases", []) if str(alias).strip()]
                attributes = [str(attr).strip().lower() for attr in item.get("attributes", []) if str(attr).strip()]
                normalized_entities.append(
                    {
                        "label": label,
                        "count": 1,
                        "aliases": aliases,
                        "regions": [],
                        "attributes": attributes,
                    }
                )
                counts[label] = max(counts.get(label, 0), 1)
                for alias in aliases:
                    counts[alias] = max(counts.get(alias, 0), 1)
            if normalized_entities:
                return {"entities": normalized_entities, "counts": counts}
        except Exception:
            pass

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return {"entities": [], "counts": {}}

        try:
            from openai import OpenAI

            image_file = Path(image_path)
            mime_type = mimetypes.guess_type(image_file.name)[0] or "image/png"
            image_b64 = base64.b64encode(image_file.read_bytes()).decode("utf-8")
            prompt = (
                "Analyze this video keyframe for retrieval indexing. Return JSON only with keys "
                "`entities` and `counts`. `entities` is an array of objects with keys: "
                "`label` (specific noun), `count` (integer >= 1), `aliases` (include broader retrieval terms like fish, shark, eel when valid), "
                "`regions` (any of left, center, right, top, middle, bottom), and `attributes` (short descriptors). "
                "`counts` is a flattened count map that includes both specific labels and broad aliases when clearly implied. "
                "Prefer semantically correct marine entities over generic detector classes. "
                "Use the caption/OCR as hints but correct them from the image when needed.\n"
                f"Caption hint: {caption_text}\n"
                f"OCR hint: {ocr_text}"
            )
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=self.model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"data:{mime_type};base64,{image_b64}"},
                        ],
                    }
                ],
            )
            payload = self._extract_json_payload(response.output_text.strip())
            entities: list[dict[str, object]] = []
            counts: dict[str, int] = {}
            for item in payload.get("entities", []):
                label = str(item.get("label", "")).strip().lower()
                count = int(item.get("count", 0) or 0)
                if not label or count <= 0:
                    continue
                aliases = []
                for alias in item.get("aliases", []):
                    normalized = str(alias).strip().lower()
                    if normalized and normalized not in aliases:
                        aliases.append(normalized)
                regions = []
                for region in item.get("regions", []):
                    normalized = str(region).strip().lower()
                    if normalized and normalized not in regions:
                        regions.append(normalized)
                attributes = []
                for attribute in item.get("attributes", []):
                    normalized = str(attribute).strip().lower()
                    if normalized and normalized not in attributes:
                        attributes.append(normalized)
                entities.append(
                    {
                        "label": label,
                        "count": count,
                        "aliases": aliases,
                        "regions": regions,
                        "attributes": attributes,
                    }
                )
            for key, value in dict(payload.get("counts", {})).items():
                normalized = str(key).strip().lower()
                if not normalized:
                    continue
                try:
                    count = int(value)
                except Exception:
                    continue
                if count > 0:
                    counts[normalized] = count
            for item in entities:
                count = int(item["count"])
                counts[str(item["label"])] = max(counts.get(str(item["label"]), 0), count)
                for alias in item["aliases"]:
                    counts[str(alias)] = max(counts.get(str(alias), 0), count)
            return {"entities": entities, "counts": counts}
        except Exception:
            return {"entities": [], "counts": {}}
