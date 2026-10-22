from __future__ import annotations

import io
import math
import re
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image


VIEWS = ("30 degrees left", "30 degrees right", "from below", "from above")
TILE = 512
SIDE = 64
OVERLAY_SIDE = 256
WEIGHTS = {"ncc": 2.322, "pix_err": -0.0567, "aspect_err": -3.1964, "hist_sim": 2.3108, "contrast": 0.084, "fill": 9.7765}
LUMA = np.array([299, 587, 114], dtype=np.int64)
BINS = 6
@dataclass(frozen=True)
class Appearance:
    crop: np.ndarray
    mask: np.ndarray
    aspect: float
    fill: float
    contrast: float
    hist: np.ndarray


def _appearance(rgb: np.ndarray, mask: np.ndarray) -> Appearance | None:
    rows, columns = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    pixels = rgb[mask].astype(np.int64)
    if rows.size == 0 or len(pixels) * 1000 < mask.size:
        return None
    crop = rgb[rows[0]: rows[-1] + 1, columns[0]: columns[-1] + 1].copy()
    inside = mask[rows[0]: rows[-1] + 1, columns[0]: columns[-1] + 1]
    crop[~inside] = 255
    height, width = crop.shape[:2]
    side = max(height, width)
    top, left = (side - height) // 2, (side - width) // 2
    square = np.full((side, side, 3), 255, dtype=np.uint8)
    square[top: top + height, left: left + width] = crop
    shape = np.zeros((side, side), dtype=np.uint8)
    shape[top: top + height, left: left + width] = inside.astype(np.uint8) * 255
    luma = pixels @ LUMA
    count, total = len(luma), int(luma.sum())
    contrast = math.sqrt(count * int((luma * luma).sum()) - total * total) / count / 1000
    sampled = pixels[:: max(1, count // 20000)] * BINS // 256
    hist = np.bincount((sampled[:, 0] * BINS + sampled[:, 1]) * BINS + sampled[:, 2], minlength=BINS ** 3)
    return Appearance(
        np.asarray(Image.fromarray(square).resize((SIDE, SIDE), Image.Resampling.BILINEAR)),
        np.asarray(Image.fromarray(shape).resize((SIDE, SIDE), Image.Resampling.BILINEAR)),
        width / height, len(pixels) / mask.size, contrast, hist.astype(np.int64))


def reference_stats(image: bytes) -> Appearance | None:
    photo = Image.open(io.BytesIO(image)).convert("RGB").resize((TILE, TILE), Image.Resampling.BILINEAR)
    rgb = np.asarray(photo)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    mask = ((rgb.astype(np.float64) - np.median(border, axis=0)) ** 2).sum(axis=2) > 1600.0
    if not 0.01 < mask.mean() < 0.95:
        return None
    return _appearance(rgb, mask)


def render_stats(grid: bytes) -> list[Appearance | None]:
    image = Image.open(io.BytesIO(grid)).convert("RGB")
    width, height = image.width // 2, image.height // 2
    views = []
    for row in range(2):
        for column in range(2):
            tile = image.crop((column * width, row * height, (column + 1) * width, (row + 1) * height))
            rgb = np.asarray(tile.resize((TILE, TILE), Image.Resampling.BILINEAR))
            views.append(_appearance(rgb, (rgb < 250).any(axis=2)))
    return views


def _metrics(view: Appearance, photo: Appearance) -> dict[str, float]:
    a, b = (view.crop.astype(np.int64) @ LUMA).ravel(), (photo.crop.astype(np.int64) @ LUMA).ravel()
    count, sum_a, sum_b = a.size, int(a.sum()), int(b.sum())
    spread = (count * int((a * a).sum()) - sum_a * sum_a) * (count * int((b * b).sum()) - sum_b * sum_b)
    ncc = (count * int((a * b).sum()) - sum_a * sum_b) / math.sqrt(spread) if spread > 0 else 0.0
    errors = np.abs(view.crop.astype(np.int64) - photo.crop.astype(np.int64))
    view_total, photo_total = int(view.hist.sum()), int(photo.hist.sum())
    shared = int(np.minimum(view.hist * photo_total, photo.hist * view_total).sum())
    return {"ncc": ncc, "pix_err": int(errors.sum()) / errors.size, "aspect_err": abs(math.log(view.aspect / photo.aspect)),
            "hist_sim": shared / (view_total * photo_total), "contrast": view.contrast, "fill": view.fill}


def view_score(view: Appearance | None, photo: Appearance) -> float:
    if view is None:
        return 0.0
    return sum(WEIGHTS[name] * value for name, value in _metrics(view, photo).items())


def grid_score(grid: bytes, photo: Appearance) -> float:
    return round(sum(view_score(view, photo) for view in render_stats(grid)) / 4, 6)


def feedback(grid: bytes, photo: Appearance) -> tuple[str, bytes]:
    views = render_stats(grid)
    pairs = [(name, view) for name, view in zip(VIEWS, views, strict=True) if view is not None]
    if not pairs:
        return "", b""

    def overlap(view: Appearance) -> float:
        union = int(np.maximum(view.mask, photo.mask).astype(np.int64).sum())
        return int(np.minimum(view.mask, photo.mask).astype(np.int64).sum()) / union if union else 0.0

    closest_name, closest = max(pairs, key=lambda pair: round(view_score(pair[1], photo), 6))
    lines = [
        f"- width / height of the object: photo {photo.aspect:.2f}; "
        + ", ".join(f"{name} {view.aspect:.2f}" for name, view in pairs),
        "- outline overlap with the photo (1.00 = identical): "
        + ", ".join(f"{name} {overlap(view):.2f}" for name, view in pairs),
        f"- contrast of the object: photo {photo.contrast:.0f}; "
        + ", ".join(f"{name} {view.contrast:.0f}" for name, view in pairs),
        f"- closest view to the photo: {closest_name}; the overlay image compares it with the photo",
    ]
    canvas = np.full((SIDE, SIDE, 3), 255, dtype=np.uint8)
    canvas[photo.mask >= 128] = (224, 64, 64)
    canvas[closest.mask >= 128] = (64, 96, 224)
    canvas[(closest.mask >= 128) & (photo.mask >= 128)] = (128, 128, 128)
    overlay = Image.fromarray(canvas).resize((OVERLAY_SIDE, OVERLAY_SIDE), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    overlay.save(buffer, "PNG")
    return "\n".join(lines), buffer.getvalue()



SIL_TILE = 256
_RE_WIREFRAME = re.compile(r"\bwireframe\s*[:=]\s*true\b")
_RE_TRANSPARENT_TRUE = re.compile(r"\btransparent\s*[:=]\s*true\b")
_RE_LOW_OPACITY = re.compile(r"\bopacity\s*[:=]\s*(?:0(?:\.[0-3]\d*)?|\.[0-3]\d*)\b")
_RE_NEAR_BLACK = re.compile(
    r"(?:0x[01][0-9a-fA-F][01][0-9a-fA-F][01][0-9a-fA-F]"
    r"|#[01][0-9a-fA-F][01][0-9a-fA-F][01][0-9a-fA-F]"
    r"|#[01]{3})(?![0-9a-fA-F])"
)


@dataclass(frozen=True)
class SilAppearance:
    mask64: np.ndarray
    aspect: float
    fg_luma: float
    hist: np.ndarray


def _sil_appearance(rgb: np.ndarray, mask: np.ndarray) -> SilAppearance | None:
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    pixels = rgb[mask].astype(np.int64)
    if rows.size == 0 or len(pixels) * 1000 < mask.size:
        return None
    inside = mask[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]
    height, width = inside.shape[:2]
    side = max(height, width)
    top, left = (side - height) // 2, (side - width) // 2
    shape = np.zeros((side, side), dtype=np.uint8)
    shape[top: top + height, left: left + width] = inside.astype(np.uint8) * 255
    luma_sum = int((pixels @ LUMA).sum())
    fg_luma = luma_sum / (len(pixels) * 1000)
    sampled = pixels[:: max(1, len(pixels) // 20000)] * BINS // 256
    hist = np.bincount((sampled[:, 0] * BINS + sampled[:, 1]) * BINS + sampled[:, 2], minlength=BINS ** 3)
    mask64 = np.asarray(Image.fromarray(shape).resize((SIDE, SIDE), Image.Resampling.BILINEAR))
    return SilAppearance(mask64, width / height, fg_luma, hist.astype(np.int64))


def flood_bg_mask(rgb: np.ndarray, tol_sq_x4: int = 6400) -> np.ndarray:
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    sorted_b = np.sort(border.astype(np.int64), axis=0)
    mid = sorted_b.shape[0] // 2
    med2 = sorted_b[mid - 1] + sorted_b[mid]
    diff2 = rgb.astype(np.int64) * 2 - med2
    close = (diff2 * diff2).sum(axis=2) <= tol_sq_x4
    h, w = close.shape
    ext = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for y in range(h):
        for x in (0, w - 1):
            if close[y, x] and not ext[y, x]:
                ext[y, x] = True
                q.append((y, x))
    for x in range(w):
        for y in (0, h - 1):
            if close[y, x] and not ext[y, x]:
                ext[y, x] = True
                q.append((y, x))
    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and close[ny, nx] and not ext[ny, nx]:
                ext[ny, nx] = True
                q.append((ny, nx))
    return ~ext


def sillint_photo(image: bytes) -> SilAppearance | None:
    rgb = np.asarray(Image.open(io.BytesIO(image)).convert("RGB").resize((SIL_TILE, SIL_TILE), Image.Resampling.BILINEAR))
    mask = flood_bg_mask(rgb)
    fg = int(mask.sum())
    if fg * 100 <= mask.size or fg * 100 >= mask.size * 95:
        return None
    return _sil_appearance(rgb, mask)


def sillint_views(grid: bytes) -> list[SilAppearance | None]:
    image = Image.open(io.BytesIO(grid)).convert("RGB")
    width, height = image.width // 2, image.height // 2
    out: list[SilAppearance | None] = []
    for row in range(2):
        for col in range(2):
            tile = image.crop((col * width, row * height, (col + 1) * width, (row + 1) * height))
            rgb = np.asarray(tile.resize((SIL_TILE, SIL_TILE), Image.Resampling.BILINEAR))
            out.append(_sil_appearance(rgb, (rgb < 250).any(axis=2)))
    return out


def sillint_js(raw: str) -> list[str]:
    if not raw:
        return []
    alerts: list[str] = []
    if _RE_WIREFRAME.search(raw):
        alerts.append("wireframe:true found (remove unless object is actual wire mesh)")
    low_ops = _RE_LOW_OPACITY.findall(raw)
    if low_ops and _RE_TRANSPARENT_TRUE.search(raw):
        alerts.append(f"{len(low_ops)} low-opacity (<0.4) material(s): verify only genuine glass/clear parts are translucent")
    near_blk = _RE_NEAR_BLACK.findall(raw)
    if near_blk:
        alerts.append(f"{len(near_blk)} near-black hex literal(s) (<#202020): lift dark parts to >= #2a2a2a")
    return alerts


def sillint_report(grid: bytes, photo: SilAppearance | None, raw: str) -> str:
    short_views = ("left", "right", "below", "above")
    lines: list[str] = []
    if photo is not None:
        views = sillint_views(grid)
        side_pairs = [(name, v) for name, v in zip(short_views[:2], views[:2], strict=True) if v is not None]
        all_pairs = [(name, v) for name, v in zip(short_views, views, strict=True) if v is not None]
        if all_pairs:
            lines.append(
                f"- aspect (w/h): photo {photo.aspect:.2f}; "
                + ", ".join(f"{n} {v.aspect:.2f}" for n, v in all_pairs)
            )
        if side_pairs:
            ious = []
            hists = []
            lumas = []
            pt = int(photo.hist.sum())
            for name, v in side_pairs:
                union = int(np.maximum(v.mask64, photo.mask64).astype(np.int64).sum())
                iou = int(np.minimum(v.mask64, photo.mask64).astype(np.int64).sum()) / union if union else 0.0
                ious.append(f"{name} {iou:.2f}")
                vt = int(v.hist.sum())
                shared = int(np.minimum(v.hist * pt, photo.hist * vt).sum()) if vt and pt else 0
                hsim = shared / (vt * pt) if vt and pt else 0.0
                hists.append(f"{name} {hsim:.2f}")
                lumas.append(f"{name} {v.fg_luma - photo.fg_luma:+.0f}")
            lines.append("- side silhouette overlap (flood IoU): " + ", ".join(ious))
            lines.append(f"- fg brightness vs photo (photo {photo.fg_luma:.0f}, neg=darker): " + ", ".join(lumas))
            lines.append("- side RGB hist similarity: " + ", ".join(hists))
    alerts = sillint_js(raw)
    if alerts:
        lines.append("- JS lint: " + "; ".join(alerts))
    return "\n".join(lines)[:650]
