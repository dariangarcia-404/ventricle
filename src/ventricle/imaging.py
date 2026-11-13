"""Small image-processing helpers (NumPy only; no SciPy/OpenCV needed)."""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image


def normalize(frame: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Percentile-window a frame into [0, 1]."""
    a = np.asarray(frame, dtype=np.float32)
    lo, hi = np.percentile(a, [low, high])
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    return np.clip((a - lo) / max(float(hi - lo), 1e-6), 0, 1)


def to_image(frame: np.ndarray) -> Image.Image:
    return Image.fromarray((normalize(frame) * 255).astype(np.uint8)).convert("RGB")


def circle_mask(shape, x: float, y: float, radius: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return (xx - float(x)) ** 2 + (yy - float(y)) ** 2 <= float(radius) ** 2


def flood(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    """4-connected region of ``mask`` containing (x, y)."""
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
        return out
    queue = deque([(y, x)])
    out[y, x] = True
    while queue:
        cy, cx = queue.popleft()
        for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                queue.append((ny, nx))
    return out


def components(mask: np.ndarray, min_area: int = 15) -> list[dict]:
    """Connected components of a boolean mask, largest first."""
    remaining = mask.copy()
    found = []
    for y, x in zip(*np.nonzero(mask)):
        if not remaining[y, x]:
            continue
        region = flood(remaining, int(x), int(y))
        remaining &= ~region
        area = int(region.sum())
        if area >= min_area:
            ys, xs = np.nonzero(region)
            found.append({"area_px": area, "cx": float(xs.mean()), "cy": float(ys.mean())})
    return sorted(found, key=lambda c: c["area_px"], reverse=True)


def roi_stats(image: np.ndarray, roi: dict) -> dict:
    """Summary statistics inside a circular ROI ``{"x", "y", "radius"}``."""
    values = image[circle_mask(image.shape, roi["x"], roi["y"], roi["radius"])].astype(float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError(f"ROI {roi} contains no pixels")
    return {
        "roi": {k: float(roi[k]) for k in ("x", "y", "radius")},
        "n_pixels": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }
