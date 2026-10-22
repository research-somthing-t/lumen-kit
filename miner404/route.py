from __future__ import annotations

import io
from types import ModuleType

import numpy as np
from PIL import Image

from . import prompts, prompts_front

DIFF, IOU, BORDER = 30, 0.90, 8


def mirror_iou(image: bytes) -> float:
    a = np.asarray(Image.open(io.BytesIO(image)).convert("RGB"), dtype=np.int16)
    border = np.concatenate([a[:BORDER].reshape(-1, 3), a[-BORDER:].reshape(-1, 3),
                             a[:, :BORDER].reshape(-1, 3), a[:, -BORDER:].reshape(-1, 3)])
    mask = np.abs(a - np.median(border, axis=0)).max(axis=2) > DIFF
    xs = np.nonzero(mask)[1]
    if xs.size == 0:
        return 0.0
    box = mask[:, xs.min():xs.max() + 1]
    flip = box[:, ::-1]
    return float(np.logical_and(box, flip).sum() / np.logical_or(box, flip).sum())


def prompt_set(image: bytes) -> ModuleType:
    try:
        return prompts_front if mirror_iou(image) >= IOU else prompts
    except Exception:
        return prompts
