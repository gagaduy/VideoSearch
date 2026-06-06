from hashlib import sha1
from pathlib import Path

from worker.adapters.base import EmbeddingResult


class OpenClipAdapter:
    def __init__(self, model_name: str = "ViT-H-14", pretrained: str = "laion2b_s32b_b79k") -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = "cpu"

    def _lazy_load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import open_clip
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self._device,
            )
            tokenizer = open_clip.get_tokenizer(self.model_name)
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            return True
        except Exception:
            self._model = None
            self._preprocess = None
            self._tokenizer = None
            return False

    def _fallback_values(self, seed: bytes) -> list[float]:
        digest = sha1(seed).digest()
        return [round(int.from_bytes(digest[index : index + 4], "big") / 2**32, 6) for index in range(0, 32, 4)]

    def embed_text(self, text: str) -> EmbeddingResult:
        if self._lazy_load():
            import torch

            tokens = self._tokenizer([text]).to(self._device)
            with torch.inference_mode():
                features = self._model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            return EmbeddingResult(model_name=self.model_name, values=features[0].detach().cpu().tolist())
        seed = text.strip().encode("utf-8") or b" "
        return EmbeddingResult(model_name=self.model_name, values=self._fallback_values(seed))

    def embed_image(self, image_path: str) -> EmbeddingResult:
        if self._lazy_load():
            import torch
            from PIL import Image

            with Image.open(image_path) as image:
                batch = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)
            with torch.inference_mode():
                features = self._model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
            return EmbeddingResult(model_name=self.model_name, values=features[0].detach().cpu().tolist())
        return EmbeddingResult(model_name=self.model_name, values=self._fallback_values(Path(image_path).read_bytes()))
