import json

try:
    import torch
except ImportError:
    print(json.dumps({"cuda_available": False, "device_count": 0, "reason": "torch_not_installed"}))
    raise SystemExit(0)

payload = {
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}

print(json.dumps(payload))

