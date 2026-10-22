from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

BACKGROUND_DISTANCE = 20
MIN_SHARE = 200


def _tile_detail(rgb) -> float | None:
    h, w = rgb.shape[:2]
    mask = np.abs(rgb - 255).sum(axis=2) > BACKGROUND_DISTANCE
    if int(mask.sum()) * MIN_SHARE < h * w:
        return None
    gray3 = rgb.sum(axis=2)
    edges3 = np.abs(np.diff(gray3, axis=0))[:, :-1] + np.abs(np.diff(gray3, axis=1))[:-1, :]
    inner = mask[:-1, :-1]
    return int(edges3[inner].sum()) / (3 * max(1, int(inner.sum())))


def _mean(values: list[float | None]) -> float | None:
    return None if any(v is None for v in values) else math.fsum(values) / len(values)


def detail(sheet: bytes, px: int) -> float | None:
    rgb_all = np.asarray(Image.open(io.BytesIO(sheet)).convert("RGB"), dtype=np.int64)
    return _mean([_tile_detail(rgb_all[(i // 2) * px:(i // 2 + 1) * px, (i % 2) * px:(i % 2 + 1) * px])
                  for i in range(4)])


def views_detail(pngs: list[bytes]) -> float | None:
    return _mean([_tile_detail(np.asarray(Image.open(io.BytesIO(p)).convert("RGB"), dtype=np.int64)) for p in pngs])


def standardized(values: list[float]) -> list[float]:
    mean = math.fsum(values) / len(values)
    spread = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values))
    return [0.0] * len(values) if spread == 0 else [(v - mean) / spread for v in values]


def choose(penalties: list[float | None], details: list[float | None], panel_weight: float,
           detail_weight: float) -> int:
    n = len(penalties)
    score = [0.0] * n
    rated = [i for i in range(n) if penalties[i] is not None]
    if rated:
        for i, z in zip(rated, standardized([-penalties[i] for i in rated])):
            score[i] += panel_weight * z
        for i in range(n):
            if penalties[i] is None:
                score[i] = -math.inf
    measured = [i for i in range(n) if details[i] is not None and score[i] != -math.inf]
    if measured:
        for i, z in zip(measured, standardized([details[i] for i in measured])):
            score[i] += detail_weight * z
    best = max(score)
    return min(i for i in range(n) if score[i] == best)
