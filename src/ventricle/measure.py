"""Measurement calculations. Pure functions over :class:`~ventricle.series.Stack` data.

Every result carries ``qa_status = "REVIEW_REQUIRED"``: these are exploratory
estimates from simple seeds/circles, not expert contours.
"""

from __future__ import annotations

import re

import numpy as np

from .imaging import circle_mask, flood, normalize, roi_stats
from .series import Stack

DISCLAIMER = (
    "Research/educational estimate only. Not a medical device; not for diagnosis or "
    "treatment. Compare with a clinical CMR report read by a qualified physician."
)


def _envelope(method: str, **extra) -> dict:
    return {"method": method, "qa_status": "REVIEW_REQUIRED", "disclaimer": DISCLAIMER, **extra}


# --------------------------------------------------------------------------- volumes

def seeded_area(frame: np.ndarray, x: float, y: float) -> float:
    """Area (pixels) of the bright blood pool grown from one seed point.

    Several relative thresholds are tried inside a disc around the seed, and the
    most compact plausible region is kept, so the region cannot leak far into
    the atria, great vessels, or chest wall.
    """
    img = normalize(frame)
    h, w = img.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return 0.0
    local = circle_mask(img.shape, xi, yi, max(26, int(0.16 * min(h, w))))
    seed_value = float(img[yi, xi])
    best = None
    for factor in (0.40, 0.50, 0.60, 0.70):
        threshold = max(0.18, min(0.85, seed_value * factor))
        area = float(flood((img >= threshold) & local, xi, yi).sum())
        if 30 <= area <= 4000:
            score = abs(area - 500) / 500
            if best is None or score < best[0]:
                best = (score, area)
    return best[1] if best else 0.0


def ventricular_volumes(stack: Stack, seeds: list[dict], bsa_m2: float | None = None) -> dict:
    """LV/RV volumes and ejection fractions from one LV and one RV seed per slice.

    ``seeds`` is a list of ``{"slice": i, "label": "LV"|"RV", "x": px, "y": px}``.
    Volume per phase = sum over slices of (area x slice spacing) (Simpson's method).
    """
    by_slice: dict[int, dict] = {}
    for seed in seeds:
        by_slice.setdefault(int(seed["slice"]), {})[str(seed["label"]).upper()] = seed
    missing = [s for s in range(stack.slices) if {"LV", "RV"} - set(by_slice.get(s, {}))]
    if missing:
        raise ValueError(f"need one LV and one RV seed on every slice; missing on slices {missing}")

    areas = {k: np.zeros((stack.phases, stack.slices)) for k in ("LV", "RV")}
    for p in range(stack.phases):
        for s in range(stack.slices):
            for label in ("LV", "RV"):
                seed = by_slice[s][label]
                areas[label][p, s] = seeded_area(stack.values[p, s], seed["x"], seed["y"])

    ml_per_px = stack.pixel_area_mm2 * stack.slice_spacing_mm / 1000.0
    result = _envelope(
        "seeded bright-blood region growth, Simpson's summation",
        series=stack.series.label,
        phases=stack.phases,
        slices=stack.slices,
        pixel_spacing_mm=list(stack.pixel_spacing_mm),
        slice_spacing_mm=stack.slice_spacing_mm,
        failed_seed_fraction=float(np.mean([(a == 0).mean() for a in areas.values()])),
    )
    for label, a in areas.items():
        k = label.lower()
        curve = a.sum(axis=1) * ml_per_px
        edv, esv = float(curve.max()), float(curve.min())
        result[f"{k}_volume_ml_by_phase"] = curve.round(2).tolist()
        result[f"{k}_edv_ml"] = edv
        result[f"{k}_esv_ml"] = esv
        result[f"{k}_stroke_volume_ml"] = edv - esv
        result[f"{k}_ef_percent"] = 100.0 * (edv - esv) / edv if edv > 0 else None
        if bsa_m2:
            result[f"{k}_edvi_ml_m2"] = edv / bsa_m2
            result[f"{k}_esvi_ml_m2"] = esv / bsa_m2
    result["rv_lv_edv_ratio"] = result["rv_edv_ml"] / result["lv_edv_ml"] if result["lv_edv_ml"] else None
    if bsa_m2:
        result["bsa_m2"] = bsa_m2
    result["limitations"] = (
        "Seeds are placed on one phase and reused for all phases; basal-slice inclusion, "
        "trabeculation/papillary conventions, and through-plane motion are not handled."
    )
    return result


# --------------------------------------------------------------------------- flow

def venc_from_description(description: str) -> float | None:
    """Parse a velocity-encoding value like ``V150`` / ``venc 150`` from a series name."""
    match = re.search(r"(?:^|[^A-Za-z])(?:VENC[\s_=]*|V)(\d+(?:\.\d+)?)(?![\d.])", description, re.IGNORECASE)
    return float(match.group(1)) if match else None


def flow_curve(phase: Stack, roi: dict, venc_cm_s: float, pixel_area_mm2: float, background: dict | None = None) -> dict:
    """Integrate through-plane flow in a circular ROI over the cardiac cycle.

    Phase values are mapped linearly onto [-VENC, +VENC] using the series'
    full stored range, then flow(t) = sum(velocity x pixel area).
    """
    lo, hi = phase.stored_range
    full_scale = max(abs(lo), abs(hi), 1e-6)
    velocity = phase.values[:, 0] / full_scale * venc_cm_s  # (phases, rows, cols), cm/s
    mask = circle_mask(velocity.shape[1:], roi["x"], roi["y"], roi["radius"])
    if mask.sum() < 5:
        raise ValueError("flow ROI is too small")
    offset = np.zeros(len(velocity))
    if background:
        bg = circle_mask(velocity.shape[1:], background["x"], background["y"], background["radius"])
        offset = np.median(velocity[:, bg], axis=1)
        velocity = velocity - offset[:, None, None]
    rate_ml_s = velocity[:, mask].sum(axis=1) * pixel_area_mm2 / 100.0  # cm/s * cm^2

    times = phase.trigger_times_ms
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1000.0
    widths_s = np.diff(np.r_[times, times[-1] + dt]) / 1000.0  # the cine loop is periodic
    mid = (rate_ml_s + np.roll(rate_ml_s, -1)) / 2.0
    positive = float(np.sum(np.clip(mid, 0, None) * widths_s))
    negative = float(-np.sum(np.clip(mid, None, 0) * widths_s))
    forward, reverse = max(positive, negative), min(positive, negative)
    return {
        "roi": {k: float(roi[k]) for k in ("x", "y", "radius")},
        "roi_pixels": int(mask.sum()),
        "venc_cm_s": venc_cm_s,
        "flow_rate_ml_s_by_phase": rate_ml_s.round(3).tolist(),
        "background_offset_cm_s_by_phase": offset.round(3).tolist(),
        "net_volume_ml_signed": positive - negative,
        "forward_volume_ml": forward,
        "reverse_volume_ml": reverse,
        "regurgitant_fraction_percent": 100.0 * reverse / forward if forward > 0 else None,
        "polarity_note": "Forward is taken as the dominant direction; the DICOM does not say which sign is antegrade.",
    }


def flow(pulmonary: dict, aortic: dict | None = None) -> dict:
    """Combine per-vessel flow curves into regurgitant fraction and Qp:Qs."""
    result = _envelope("circular-ROI phase-contrast flow integration", pulmonary=pulmonary)
    if aortic:
        result["aorta"] = aortic
        result["qp_qs"] = pulmonary["forward_volume_ml"] / aortic["forward_volume_ml"] if aortic["forward_volume_ml"] else None
    return result


# --------------------------------------------------------------------------- tissue

def ecv(hematocrit: float, myo_native: float, blood_native: float, myo_post: float, blood_post: float) -> float:
    """Extracellular volume fraction (0-1) from native/post-contrast T1 (ms)."""
    if not 0 < hematocrit < 1:
        raise ValueError("hematocrit must be a fraction, e.g. 0.42")
    d_myo = 1.0 / myo_post - 1.0 / myo_native
    d_blood = 1.0 / blood_post - 1.0 / blood_native
    if d_blood == 0:
        raise ValueError("blood T1 did not change between native and post-contrast maps")
    return (1.0 - hematocrit) * d_myo / d_blood


def t1(native: np.ndarray, post: np.ndarray, rois: dict, hematocrit: float | None = None) -> dict:
    """Median T1 in myocardium/blood ROIs on native and post maps, plus ECV if possible."""
    stats = {
        "native_myocardium": roi_stats(native, rois["native_myocardium"]),
        "native_blood": roi_stats(native, rois["native_blood"]),
        "post_myocardium": roi_stats(post, rois["post_myocardium"]),
        "post_blood": roi_stats(post, rois["post_blood"]),
    }
    medians = {k: v["median"] for k, v in stats.items()}
    result = _envelope(
        "circular-ROI medians on scanner-derived T1 maps",
        units="ms (assumed; verify with your scanner protocol)",
        **{f"{k}_t1_ms": v for k, v in medians.items()},
        roi_statistics=stats,
        note="T1 reference ranges are specific to scanner, field strength, and sequence.",
    )
    if hematocrit is not None:
        result["hematocrit"] = hematocrit
        result["ecv_percent"] = 100.0 * ecv(
            hematocrit, medians["native_myocardium"], medians["native_blood"],
            medians["post_myocardium"], medians["post_blood"],
        )
    return result


def lge(image: np.ndarray, target: dict, remote: dict, n_sd: float = 5.0) -> dict:
    """Compare a target myocardial ROI with remote myocardium on an LGE/PSIR image."""
    t, r = roi_stats(image, target), roi_stats(image, remote)
    threshold = r["mean"] + n_sd * r["std"]
    values = image[circle_mask(image.shape, target["x"], target["y"], target["radius"])]
    return _envelope(
        f"circular-ROI signal comparison, remote mean + {n_sd:g} SD threshold",
        target=t,
        remote=r,
        threshold=float(threshold),
        target_fraction_above_threshold_percent=float(100.0 * np.mean(values > threshold)),
        target_to_remote_mean_ratio=t["mean"] / r["mean"] if r["mean"] else None,
        target_z_score=(t["mean"] - r["mean"]) / r["std"] if r["std"] else None,
        limitations=(
            "Circles can include blood pool or fat; thin walls, motion, partial volume, and "
            "PSIR artifacts mean this cannot rule enhancement in or out."
        ),
    )
