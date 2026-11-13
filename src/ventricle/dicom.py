"""A small, dependency-free DICOM Part 10 reader.

Only the handful of tags needed for cardiac-MR measurement are decoded, and
patient-identifying tags (name, ID, birth date, ...) are deliberately never
read, so they cannot leak into any output this package writes.

Supported transfer syntaxes: Implicit VR Little Endian and Explicit VR Little
Endian (the uncompressed formats used by most scanner CD exports).
Compressed (encapsulated) pixel data is rejected with a clear error; convert
such studies first, e.g. with ``gdcmconv --raw`` or ``dcmdjpeg``.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

IMPLICIT_VR_LE = "1.2.840.10008.1.2"
EXPLICIT_VR_LE = "1.2.840.10008.1.2.1"
SUPPORTED_SYNTAXES = {IMPLICIT_VR_LE, EXPLICIT_VR_LE}

# VRs whose explicit-VR header uses a 2-byte reserved field and a 4-byte length.
LONG_VR = {"OB", "OD", "OF", "OL", "OV", "OW", "SQ", "SV", "UC", "UN", "UR", "UT", "UV"}
UNDEFINED = 0xFFFFFFFF
ITEM = (0xFFFE, 0xE000)
ITEM_DELIM = (0xFFFE, 0xE00D)
SEQ_DELIM = (0xFFFE, 0xE0DD)
PIXEL_DATA = (0x7FE0, 0x0010)

# tag -> (keyword, VR). The VR is needed to decode implicit-VR files.
TAGS = {
    (0x0002, 0x0010): ("TransferSyntaxUID", "UI"),
    (0x0008, 0x0008): ("ImageType", "CS"),
    (0x0008, 0x0016): ("SOPClassUID", "UI"),
    (0x0008, 0x0060): ("Modality", "CS"),
    (0x0008, 0x103E): ("SeriesDescription", "LO"),
    (0x0018, 0x0050): ("SliceThickness", "DS"),
    (0x0018, 0x0088): ("SpacingBetweenSlices", "DS"),
    (0x0018, 0x1060): ("TriggerTime", "DS"),
    (0x0020, 0x000E): ("SeriesInstanceUID", "UI"),
    (0x0020, 0x0011): ("SeriesNumber", "IS"),
    (0x0020, 0x0013): ("InstanceNumber", "IS"),
    (0x0020, 0x0032): ("ImagePositionPatient", "DS"),
    (0x0020, 0x0037): ("ImageOrientationPatient", "DS"),
    (0x0020, 0x1041): ("SliceLocation", "DS"),
    (0x0028, 0x0002): ("SamplesPerPixel", "US"),
    (0x0028, 0x0004): ("PhotometricInterpretation", "CS"),
    (0x0028, 0x0008): ("NumberOfFrames", "IS"),
    (0x0028, 0x0010): ("Rows", "US"),
    (0x0028, 0x0011): ("Columns", "US"),
    (0x0028, 0x0030): ("PixelSpacing", "DS"),
    (0x0028, 0x0100): ("BitsAllocated", "US"),
    (0x0028, 0x0101): ("BitsStored", "US"),
    (0x0028, 0x0103): ("PixelRepresentation", "US"),
    (0x0028, 0x1050): ("WindowCenter", "DS"),
    (0x0028, 0x1051): ("WindowWidth", "DS"),
    (0x0028, 0x1052): ("RescaleIntercept", "DS"),
    (0x0028, 0x1053): ("RescaleSlope", "DS"),
}

_BINARY = {"US": "H", "SS": "h", "UL": "I", "SL": "i", "FL": "f", "FD": "d"}


class DicomError(ValueError):
    """Raised for files this reader cannot (or will not) interpret."""


def first(value, default=None):
    """Return the first element of a multi-valued attribute, or the value itself."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


def _decode(vr: str, raw: bytes):
    if vr in _BINARY:
        code = _BINARY[vr]
        n = len(raw) // struct.calcsize(code)
        values = list(struct.unpack("<" + code * n, raw[: n * struct.calcsize(code)]))
        return values[0] if len(values) == 1 else values
    text = raw.decode("latin-1", errors="replace").rstrip(" \x00")
    parts = [p.strip() for p in text.split("\\")]
    if vr in {"DS", "IS"}:
        out = []
        for p in parts:
            if not p:
                continue
            try:
                out.append(int(p) if vr == "IS" else float(p))
            except ValueError:
                out.append(p)
        return out[0] if len(out) == 1 else out
    return parts if len(parts) > 1 else text


def _header(data: bytes, pos: int, explicit: bool):
    """Return (tag, vr, length, value_pos) for the element starting at ``pos``."""
    if pos + 8 > len(data):
        raise EOFError
    group, elem = struct.unpack_from("<HH", data, pos)
    tag = (group, elem)
    if group == 0xFFFE:  # item / delimiters never carry a VR
        return tag, None, struct.unpack_from("<I", data, pos + 4)[0], pos + 8
    if explicit:
        vr = data[pos + 4 : pos + 6].decode("ascii", errors="replace")
        if vr in LONG_VR:
            if pos + 12 > len(data):
                raise EOFError
            return tag, vr, struct.unpack_from("<I", data, pos + 8)[0], pos + 12
        return tag, vr, struct.unpack_from("<H", data, pos + 6)[0], pos + 8
    vr = TAGS.get(tag, (None, "UN"))[1]
    return tag, vr, struct.unpack_from("<I", data, pos + 4)[0], pos + 8


def _skip_undefined(data: bytes, pos: int, explicit: bool, terminator) -> int:
    """Skip nested content of undefined length; return position after ``terminator``."""
    while True:
        tag, vr, length, value_pos = _header(data, pos, explicit)
        if tag == terminator:
            return value_pos
        if length == UNDEFINED:
            # A sequence ends at SEQ_DELIM; an item ends at ITEM_DELIM.
            end = ITEM_DELIM if tag == ITEM else SEQ_DELIM
            pos = _skip_undefined(data, value_pos, explicit, end)
        else:
            pos = value_pos + length


def read(path, pixels: bool = False) -> dict:
    """Read the supported tags (and optionally the pixel array) of one file.

    Returns a dict keyed by DICOM keyword. With ``pixels=True`` the dict also
    holds ``array``: a NumPy array shaped (rows, cols) or (frames, rows, cols)
    containing *stored* values (apply :func:`rescale` for real-world units).
    """
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 132 or data[128:132] != b"DICM":
        raise DicomError(f"not a DICOM Part 10 file: {path}")

    meta: dict = {}
    pos = 132
    explicit = True  # group 0002 is always explicit VR little endian
    switched = False
    pixel_bytes = None
    while True:
        try:
            if not switched and struct.unpack_from("<H", data, pos)[0] != 0x0002:
                syntax = meta.get("TransferSyntaxUID", EXPLICIT_VR_LE)
                if syntax not in SUPPORTED_SYNTAXES:
                    raise DicomError(
                        f"unsupported transfer syntax {syntax} in {path.name}; "
                        "decompress the study first (e.g. gdcmconv --raw)"
                    )
                explicit = syntax == EXPLICIT_VR_LE
                switched = True
            tag, vr, length, value_pos = _header(data, pos, explicit)
        except (EOFError, struct.error):
            break
        if tag == PIXEL_DATA:
            if length == UNDEFINED:
                raise DicomError(f"encapsulated (compressed) pixel data in {path.name}")
            pixel_bytes = data[value_pos : value_pos + length]
            break
        if length == UNDEFINED:
            pos = _skip_undefined(data, value_pos, explicit, SEQ_DELIM)
            continue
        if tag in TAGS:
            keyword, default_vr = TAGS[tag]
            meta[keyword] = _decode(vr if vr and vr != "UN" else default_vr, data[value_pos : value_pos + length])
        pos = value_pos + length

    if pixels:
        meta["array"] = _pixel_array(meta, pixel_bytes, path)
    return meta


def _pixel_array(meta: dict, raw, path: Path) -> np.ndarray:
    if raw is None:
        raise DicomError(f"no pixel data in {path.name}")
    if int(first(meta.get("SamplesPerPixel"), 1)) != 1:
        raise DicomError(f"only single-channel (grayscale) images are supported: {path.name}")
    rows = int(first(meta.get("Rows"), 0))
    cols = int(first(meta.get("Columns"), 0))
    frames = int(first(meta.get("NumberOfFrames"), 1) or 1)
    bits = int(first(meta.get("BitsAllocated"), 16))
    signed = int(first(meta.get("PixelRepresentation"), 0)) == 1
    dtype = {8: "i1" if signed else "u1", 16: "<i2" if signed else "<u2", 32: "<i4" if signed else "<u4"}.get(bits)
    if dtype is None:
        raise DicomError(f"unsupported BitsAllocated={bits} in {path.name}")
    expected = rows * cols * frames
    arr = np.frombuffer(raw, dtype=np.dtype(dtype))
    if arr.size < expected:
        raise DicomError(f"truncated pixel data in {path.name}")
    arr = arr[:expected].reshape((frames, rows, cols))
    return arr[0] if frames == 1 else arr


def rescale(meta: dict, array: np.ndarray) -> np.ndarray:
    """Apply RescaleSlope/RescaleIntercept, returning float32 real-world values."""
    slope = float(first(meta.get("RescaleSlope"), 1.0) or 1.0)
    intercept = float(first(meta.get("RescaleIntercept"), 0.0) or 0.0)
    return array.astype(np.float32) * slope + intercept


def is_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False
