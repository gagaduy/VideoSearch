Co-DETR model assets for offline object indexing belong in this directory.

Expected default filenames:

- `co_dino_5scale_r50_1x_coco.py`
- `co_dino_5scale_r50_1x_coco.pth`

Notes:

- This project keeps Co-DETR strictly in the offline worker enrich/indexing path.
- Search should continue to read indexed object metadata only; no online detector pass is introduced.
- The runtime currently depends on a full `mmcv` install with native ops, not `mmcv-lite`.
- The active env must provide a CUDA/PyTorch/MMCV toolchain that is mutually compatible before the adapter can run for real.
