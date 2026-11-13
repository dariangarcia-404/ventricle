"""Find DICOM series in a folder (or a mounted CD) and load them as image stacks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import dicom
from .dicom import first


@dataclass
class Series:
    number: int
    description: str
    image_type: str
    files: list[Path]
    meta: dict = field(repr=False)

    @property
    def label(self) -> str:
        return f"#{self.number} {self.description}".strip()


@dataclass
class Stack:
    """Images of one series arranged as ``values[phase, slice, row, col]``.

    ``values`` are rescaled, real-world numbers (signal, cm/s-encoded phase,
    T1 ms, ...). Cine series have many phases; maps and LGE usually have one.
    """

    series: Series
    values: np.ndarray
    trigger_times_ms: np.ndarray
    slice_positions_mm: np.ndarray
    pixel_spacing_mm: tuple[float, float]
    slice_spacing_mm: float
    stored_range: tuple[float, float]

    @property
    def phases(self) -> int:
        return self.values.shape[0]

    @property
    def slices(self) -> int:
        return self.values.shape[1]

    @property
    def pixel_area_mm2(self) -> float:
        return self.pixel_spacing_mm[0] * self.pixel_spacing_mm[1]


def _iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") or name.upper() == "DICOMDIR":
                continue
            yield Path(dirpath) / name


def discover(root) -> list[Series]:
    """Group every readable DICOM image below ``root`` into series."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    groups: dict = {}
    for path in [root] if root.is_file() else _iter_files(root):
        if not dicom.is_dicom(path):
            continue
        try:
            meta = dicom.read(path)
        except dicom.DicomError:
            continue
        if "Rows" not in meta:
            continue  # not an image (e.g. a structured report)
        key = meta.get("SeriesInstanceUID") or (first(meta.get("SeriesNumber"), -1), meta.get("SeriesDescription", ""))
        groups.setdefault(key, []).append((path, meta))
    series = []
    for items in groups.values():
        meta = items[0][1]
        itype = meta.get("ImageType", "")
        series.append(Series(
            number=int(first(meta.get("SeriesNumber"), -1)),
            description=str(meta.get("SeriesDescription", "")).strip(),
            image_type="\\".join(itype) if isinstance(itype, list) else str(itype),
            files=[p for p, _ in items],
            meta=meta,
        ))
    return sorted(series, key=lambda s: (s.number, s.description))


def by_number(series: list[Series], number: int) -> Series:
    for s in series:
        if s.number == number:
            return s
    raise LookupError(f"no series #{number}; run `ventricle inventory` to list them")


def _slice_position(meta: dict) -> float:
    """Position of an image along its slice normal (falls back to SliceLocation)."""
    pos = meta.get("ImagePositionPatient")
    ori = meta.get("ImageOrientationPatient")
    if isinstance(pos, list) and isinstance(ori, list) and len(pos) == 3 and len(ori) == 6:
        normal = np.cross(np.asarray(ori[:3], float), np.asarray(ori[3:], float))
        return float(np.dot(normal, np.asarray(pos, float)))
    loc = first(meta.get("SliceLocation"))
    return float(loc) if isinstance(loc, (int, float)) else 0.0


def load(series: Series) -> Stack:
    """Load all pixels of a series into a :class:`Stack`."""
    by_slice: dict[float, list] = {}
    sample = None
    lo, hi = np.inf, -np.inf
    for path in series.files:
        meta = dicom.read(path, pixels=True)
        raw = meta.pop("array")
        frames = raw if raw.ndim == 3 else raw[None]
        values = dicom.rescale(meta, frames)
        bits = int(first(meta.get("BitsStored"), 16))
        signed = int(first(meta.get("PixelRepresentation"), 0)) == 1
        raw_lo, raw_hi = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if signed else (0, 2**bits - 1)
        ends = dicom.rescale(meta, np.array([raw_lo, raw_hi]))
        lo, hi = min(lo, float(ends.min())), max(hi, float(ends.max()))
        key = round(_slice_position(meta), 2)
        trigger = float(first(meta.get("TriggerTime"), 0.0) or 0.0)
        instance = int(first(meta.get("InstanceNumber"), 0) or 0)
        for i, frame in enumerate(values):
            by_slice.setdefault(key, []).append((trigger, instance, i, frame))
        sample = sample or meta
    if not by_slice:
        raise ValueError(f"series {series.label} contains no images")

    keys = sorted(by_slice)
    phases = min(len(v) for v in by_slice.values())
    ordered = [sorted(by_slice[k], key=lambda r: r[:3])[:phases] for k in keys]
    shapes = {r[3].shape for group in ordered for r in group}
    if len(shapes) != 1:
        raise ValueError(f"series {series.label} mixes image sizes {sorted(shapes)}")
    values = np.stack([np.stack([r[3] for r in group]) for group in ordered], axis=1)

    spacing = sample.get("PixelSpacing", [1.0, 1.0])
    if not isinstance(spacing, list) or len(spacing) < 2:
        spacing = [1.0, 1.0]
    positions = np.asarray(keys, float)
    if len(positions) > 1:
        slice_spacing = float(np.median(np.abs(np.diff(positions))))
    else:
        slice_spacing = float(first(sample.get("SpacingBetweenSlices"), first(sample.get("SliceThickness"), 1.0)) or 1.0)
    return Stack(
        series=series,
        values=values.astype(np.float32),
        trigger_times_ms=np.asarray([r[0] for r in ordered[0]], float),
        slice_positions_mm=positions,
        pixel_spacing_mm=(float(spacing[0]), float(spacing[1])),
        slice_spacing_mm=slice_spacing,
        stored_range=(lo, hi),
    )
