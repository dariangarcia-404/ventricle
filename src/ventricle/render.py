"""Contact sheets and ROI overlays, so every number can be audited visually."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .imaging import to_image
from .series import Stack


def contact_sheet(stack: Stack, path: Path, by: str = "slice", max_tiles: int = 30, cols: int = 6, thumb: int = 200) -> Path:
    """Grid of thumbnails across slices (at the first phase) or phases (at the middle slice)."""
    if by == "phase":
        frames = [(stack.values[p, stack.slices // 2], f"phase {p + 1}  t={stack.trigger_times_ms[p]:.0f}ms") for p in range(stack.phases)]
    else:
        frames = [(stack.values[0, s], f"slice {s + 1}  {stack.slice_positions_mm[s]:.1f}mm") for s in range(stack.slices)]
    take = np.linspace(0, len(frames) - 1, min(max_tiles, len(frames))).round().astype(int)
    frames = [frames[i] for i in sorted(set(take.tolist()))]
    cols = min(cols, len(frames))
    cell = thumb + 8
    sheet = Image.new("RGB", (cols * cell, math.ceil(len(frames) / cols) * (cell + 16)), "black")
    draw = ImageDraw.Draw(sheet)
    for i, (frame, caption) in enumerate(frames):
        im = to_image(frame)
        im.thumbnail((thumb, thumb))
        x, y = (i % cols) * cell + 4, (i // cols) * (cell + 16) + 4
        sheet.paste(im, (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
        draw.text((x, y + thumb + 2), caption, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def roi_overlay(frame: np.ndarray, rois: dict, path: Path, size: int = 512) -> Path:
    """Draw labelled circular ROIs (and point seeds with radius 0) on a frame."""
    im = to_image(frame)
    scale = size / max(im.size)
    im = im.resize((round(im.width * scale), round(im.height * scale)))
    draw = ImageDraw.Draw(im)
    palette = ["#ff4d4d", "#4dd2ff", "#ffd24d", "#7dff7d", "#ff8cff"]
    for i, (name, roi) in enumerate(rois.items()):
        color = palette[i % len(palette)]
        x, y = roi["x"] * scale, roi["y"] * scale
        r = max(float(roi.get("radius", 0)) * scale, 4)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
        draw.text((x + r + 3, y - 6), name, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return path


def seed_overlay(stack: Stack, seeds: list[dict], path: Path, cols: int = 4, thumb: int = 220) -> Path:
    """Montage of every slice with its LV/RV seeds, for checking guided volumes."""
    cell = thumb + 8
    sheet = Image.new("RGB", (cols * cell, math.ceil(stack.slices / cols) * (cell + 16)), "black")
    draw = ImageDraw.Draw(sheet)
    colors = {"LV": "#ff4d4d", "RV": "#4dd2ff"}
    for s in range(stack.slices):
        im = to_image(stack.values[0, s])
        ow = im.width
        im.thumbnail((thumb, thumb))
        k = im.width / ow
        x0, y0 = (s % cols) * cell + 4, (s // cols) * (cell + 16) + 4
        sheet.paste(im, (x0, y0))
        for seed in (d for d in seeds if int(d["slice"]) == s):
            x, y = x0 + seed["x"] * k, y0 + seed["y"] * k
            c = colors.get(str(seed["label"]).upper(), "yellow")
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=c, width=2)
            draw.text((x + 7, y - 7), str(seed["label"]).upper(), fill=c)
        draw.text((x0, y0 + thumb + 2), f"slice {s + 1}/{stack.slices}", fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path
