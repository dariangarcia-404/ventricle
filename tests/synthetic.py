"""Write tiny synthetic DICOM files so tests never need real patient data."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ventricle.dicom import EXPLICIT_VR_LE, IMPLICIT_VR_LE


def _el(group, elem, vr, value: bytes, explicit: bool) -> bytes:
    if len(value) % 2:
        value += b"\x00" if vr in {"UI", "OB"} else b" "
    head = struct.pack("<HH", group, elem)
    if not explicit:
        return head + struct.pack("<I", len(value)) + value
    if vr in {"OB", "OW", "SQ", "UN", "UT"}:
        return head + vr.encode() + b"\x00\x00" + struct.pack("<I", len(value)) + value
    return head + vr.encode() + struct.pack("<H", len(value)) + value


def _s(v) -> bytes:
    if isinstance(v, (list, tuple)):
        return "\\".join(_fmt(x) for x in v).encode()
    return _fmt(v).encode()


def _fmt(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def write(path: Path, pixels: np.ndarray, *, explicit=True, series_uid="1.2.3.4", series_number=1,
          description="test", instance=1, trigger=0.0, position=(0.0, 0.0, 0.0), spacing=(1.5, 1.5),
          slope=1.0, intercept=0.0, image_type=("ORIGINAL", "PRIMARY"), with_sequence=True,
          syntax=None) -> Path:
    syntax = syntax or (EXPLICIT_VR_LE if explicit else IMPLICIT_VR_LE)
    rows, cols = pixels.shape
    meta = _el(0x0002, 0x0010, "UI", syntax.encode(), True)
    meta = _el(0x0002, 0x0000, "UL", struct.pack("<I", len(meta)), True) + meta
    body = b""
    body += _el(0x0008, 0x0008, "CS", _s(list(image_type)), explicit)
    body += _el(0x0008, 0x0060, "CS", b"MR", explicit)
    if with_sequence:
        # an undefined-length sequence containing a nested undefined-length item,
        # with a decoy tag inside that must NOT be read
        inner = _el(0x0008, 0x103E, "LO", b"DECOY", explicit)
        item = struct.pack("<HHI", 0xFFFE, 0xE000, 0xFFFFFFFF) + inner + struct.pack("<HHI", 0xFFFE, 0xE00D, 0)
        if explicit:
            seq = struct.pack("<HH", 0x0008, 0x1140) + b"SQ\x00\x00" + struct.pack("<I", 0xFFFFFFFF)
        else:
            seq = struct.pack("<HHI", 0x0008, 0x1140, 0xFFFFFFFF)
        body += seq + item + struct.pack("<HHI", 0xFFFE, 0xE0DD, 0)
    body += _el(0x0008, 0x103E, "LO", description.encode(), explicit)
    body += _el(0x0010, 0x0010, "PN", b"DOE^JANE", explicit)  # must never be surfaced
    body += _el(0x0018, 0x1060, "DS", _s(float(trigger)), explicit)
    body += _el(0x0020, 0x000E, "UI", series_uid.encode(), explicit)
    body += _el(0x0020, 0x0011, "IS", _s(series_number), explicit)
    body += _el(0x0020, 0x0013, "IS", _s(instance), explicit)
    body += _el(0x0020, 0x0032, "DS", _s([float(p) for p in position]), explicit)
    body += _el(0x0020, 0x0037, "DS", _s([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), explicit)
    body += _el(0x0028, 0x0002, "US", struct.pack("<H", 1), explicit)
    body += _el(0x0028, 0x0004, "CS", b"MONOCHROME2", explicit)
    body += _el(0x0028, 0x0010, "US", struct.pack("<H", rows), explicit)
    body += _el(0x0028, 0x0011, "US", struct.pack("<H", cols), explicit)
    body += _el(0x0028, 0x0030, "DS", _s([float(s) for s in spacing]), explicit)
    body += _el(0x0028, 0x0100, "US", struct.pack("<H", 16), explicit)
    body += _el(0x0028, 0x0101, "US", struct.pack("<H", 12), explicit)
    body += _el(0x0028, 0x0103, "US", struct.pack("<H", 0), explicit)
    body += _el(0x0028, 0x1052, "DS", _s(float(intercept)), explicit)
    body += _el(0x0028, 0x1053, "DS", _s(float(slope)), explicit)
    body += _el(0x7FE0, 0x0010, "OW", pixels.astype("<u2").tobytes(), explicit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 128 + b"DICM" + meta + body)
    return path


def disk(shape, cx, cy, r, value=1000, background=100) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    img = np.full(shape, background, dtype=np.uint16)
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r**2] = value
    return img
