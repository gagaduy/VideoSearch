import base64
import mimetypes
import os
from pathlib import Path

from app.config import settings
from worker.adapters.internvl_adapter import InternvlAdapter


class CaptionAdapter:
    def __init__(self, model_name: str | None = None, *, local_adapter: InternvlAdapter | None = None) -> None:
        self._local_adapter = local_adapter or InternvlAdapter()
        self.model_name = model_name or settings.openai_model

    def caption(self, image_path: str) -> dict[str, object]:
        try:
            local = self._local_adapter.describe_image(image_path)
            if local.get("caption"):
                return {"caption": local["caption"], "model_name": str(local["model_name"]), "confidence": 0.9}
        except Exception:
            pass

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=api_key)
                image_file = Path(image_path)
                mime_type = mimetypes.guess_type(image_file.name)[0] or "image/png"
                image_b64 = base64.b64encode(image_file.read_bytes()).decode("utf-8")
                response = client.responses.create(
                    model=self.model_name,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Describe this video frame in one concise sentence."},
                                {"type": "input_image", "image_url": f"data:{mime_type};base64,{image_b64}"},
                            ],
                        }
                    ],
                )
                text = response.output_text.strip()
                if text:
                    return {"caption": text, "model_name": self.model_name, "confidence": 1.0}
            except Exception:
                pass
        return {"caption": "", "model_name": "disabled", "confidence": 0.0}

    def close(self) -> None:
        close = getattr(self._local_adapter, "close", None)
        if callable(close):
            close()
