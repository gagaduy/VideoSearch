from __future__ import annotations

import json
from typing import Any

from app.config import settings

_NOISE_TAGS = {
    "the",
    "image",
    "shows",
    "showing",
    "this",
    "that",
    "with",
    "and",
    "background",
}


class InternvlAdapter:
    def __init__(
        self,
        model_name: str | None = None,
        *,
        load_in_8bit: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.internvl_model
        self.load_in_8bit = settings.internvl_use_8bit if load_in_8bit is None else load_in_8bit
        self.max_new_tokens = max_new_tokens or settings.internvl_max_new_tokens
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        import torch
        from transformers import AutoModel, AutoTokenizer

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "use_flash_attn": False,
            "trust_remote_code": True,
        }
        quantized_kwargs: dict[str, Any] = dict(model_kwargs)
        if self.load_in_8bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                quantized_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except Exception:
                quantized_kwargs = dict(model_kwargs)

        try:
            if self.load_in_8bit and torch.cuda.is_available():
                self._model = AutoModel.from_pretrained(self.model_name, **quantized_kwargs).eval()
            else:
                self._model = AutoModel.from_pretrained(self.model_name, **model_kwargs).eval()
        except Exception:
            self._model = AutoModel.from_pretrained(self.model_name, **model_kwargs).eval()
        if torch.cuda.is_available() and not hasattr(self._model, "hf_device_map"):
            self._model = self._model.cuda()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False)
        return self._model, self._tokenizer

    def _run_prompt(self, image_path: str, prompt: str) -> str:
        import torch
        import torchvision.transforms as T
        from PIL import Image
        from torchvision.transforms.functional import InterpolationMode

        model, tokenizer = self._lazy_load()

        image_size = int(getattr(model.config.force_image_size, "to_tuple", lambda: None)() or 448) if hasattr(model.config, "force_image_size") else 448
        if isinstance(getattr(model.config, "force_image_size", None), int):
            image_size = int(model.config.force_image_size)

        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)

        def build_transform(input_size: int):
            return T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=imagenet_mean, std=imagenet_std),
                ]
            )

        def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, tile_size: int):
            best_ratio_diff = float("inf")
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff and area > 0.5 * tile_size * tile_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
            return best_ratio

        def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12, tile_size: int = 448):
            width, height = image.size
            aspect_ratio = width / height
            target_ratios = sorted(
                {
                    (i, j)
                    for n in range(min_num, max_num + 1)
                    for i in range(1, n + 1)
                    for j in range(1, n + 1)
                    if min_num <= i * j <= max_num
                },
                key=lambda item: item[0] * item[1],
            )
            target_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, tile_size)
            target_width = tile_size * target_ratio[0]
            target_height = tile_size * target_ratio[1]
            blocks = target_ratio[0] * target_ratio[1]
            resized = image.resize((target_width, target_height))
            processed = []
            for index in range(blocks):
                box = (
                    (index % (target_width // tile_size)) * tile_size,
                    (index // (target_width // tile_size)) * tile_size,
                    ((index % (target_width // tile_size)) + 1) * tile_size,
                    ((index // (target_width // tile_size)) + 1) * tile_size,
                )
                processed.append(resized.crop(box))
            if len(processed) != 1:
                processed.append(image.resize((tile_size, tile_size)))
            return processed

        with Image.open(image_path).convert("RGB") as image:
            transform = build_transform(image_size)
            images = dynamic_preprocess(image, tile_size=image_size)
            pixel_values = torch.stack([transform(item) for item in images])

        model_dtype = next(model.parameters()).dtype
        model_device = getattr(model, "device", next(model.parameters()).device)
        if torch.cuda.is_available():
            pixel_values = pixel_values.to(dtype=model_dtype, device=model_device)

        generation_config = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        question = f"<image>\n{prompt}"
        return str(model.chat(tokenizer, pixel_values, question, generation_config)).strip()

    def _extract_json_payload(self, text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("no json object found")
        return json.loads(text[start : end + 1])

    def _normalize_tags(self, values: list[Any]) -> list[str]:
        tags: list[str] = []
        for value in values:
            normalized = str(value).strip().lower()
            if normalized and normalized not in _NOISE_TAGS and normalized not in tags:
                tags.append(normalized)
        return tags

    def _normalize_entities(self, values: list[dict[str, Any]]) -> list[dict[str, object]]:
        entities: list[dict[str, object]] = []
        for item in values:
            label = str(item.get("label", "")).strip().lower()
            if not label:
                continue
            aliases = self._normalize_tags(list(item.get("aliases", [])))
            attributes = self._normalize_tags(list(item.get("attributes", [])))
            entities.append(
                {
                    "label": label,
                    "aliases": aliases,
                    "attributes": attributes,
                }
            )
        return entities

    def describe_image(self, image_path: str) -> dict[str, object]:
        prompt = (
            "Describe this video keyframe for retrieval indexing. "
            "Return JSON only with keys: caption, tags, entities. "
            "caption must be one concise sentence. "
            "tags must be a list of short retrieval terms. "
            "entities must be a list of objects with keys label, aliases, attributes."
        )
        try:
            text = self._run_prompt(image_path, prompt)
            payload = self._extract_json_payload(text)
            return {
                "caption": str(payload.get("caption", "")).strip(),
                "tags": self._normalize_tags(list(payload.get("tags", []))),
                "entities": self._normalize_entities(list(payload.get("entities", []))),
                "model_name": self.model_name,
                "image_path": image_path,
                "raw_text": text,
            }
        except Exception:
            try:
                fallback_text = self._run_prompt(image_path, "Please describe this image in one concise sentence.")
            except Exception:
                fallback_text = ""
            tags = self._normalize_tags(
                token
                for token in fallback_text.lower().replace(",", " ").split()
                if token.isalpha() and len(token) > 2
            )
            entities = [{"label": tag, "aliases": [], "attributes": []} for tag in tags[:4]]
            return {
                "caption": fallback_text.strip(),
                "tags": tags[:8],
                "entities": entities,
                "model_name": self.model_name,
                "image_path": image_path,
                "raw_text": fallback_text,
            }

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
