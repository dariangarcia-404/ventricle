import math

import numpy as np
import pytest

from ventricle import measure
from ventricle.series import discover, load
from synthetic import disk, write

SHAPE = (128, 128)


def make_cine(root, radii_lv, radii_rv, slices=3, spacing=(1.5, 1.5), slice_gap=8.0):
    for s in range(slices):
        for p, (rl, rr) in enumerate(zip(radii_lv, radii_rv)):
            img = disk(SHAPE, 70, 64, rl)
            img[disk(SHAPE, 35, 64, rr, value=1, background=0).astype(bool)] = 1000
            write(root / f"s{s}p{p}.dcm", img, series_number=10, description="SAX cine",
                  series_uid="1.10", instance=s * 100 + p, trigger=p * 40.0,
                  position=(0.0, 0.0, s * slice_gap), spacing=spacing)


def test_discover_and_load_geometry(tmp_path):
    make_cine(tmp_path / "study", [12, 9], [10, 8])
    write(tmp_path / "study" / "other.dcm", disk(SHAPE, 60, 60, 5), series_number=2, series_uid="1.2", description="scout")
    series = discover(tmp_path / "study")
    assert [s.number for s in series] == [2, 10]
    stack = load(series[1])
    assert stack.values.shape == (2, 3, *SHAPE)
    assert stack.slice_spacing_mm == pytest.approx(8.0)
    assert stack.trigger_times_ms.tolist() == [0.0, 40.0]


def test_ventricular_volumes_track_disk_area(tmp_path):
    make_cine(tmp_path, [14, 10], [11, 7])
    stack = load(discover(tmp_path)[0])
    seeds = [{"slice": s, "label": l, "x": x, "y": 64} for s in range(3) for l, x in (("LV", 70), ("RV", 35))]
    r = measure.ventricular_volumes(stack, seeds, bsa_m2=2.0)
    ml = lambda radius: 3 * math.pi * radius**2 * 1.5 * 1.5 * 8.0 / 1000
    assert r["lv_edv_ml"] == pytest.approx(ml(14), rel=0.05)
    assert r["lv_esv_ml"] == pytest.approx(ml(10), rel=0.05)
    assert r["rv_edv_ml"] == pytest.approx(ml(11), rel=0.05)
    assert r["lv_ef_percent"] == pytest.approx(100 * (1 - 100 / 196), abs=3)
    assert r["lv_edvi_ml_m2"] == pytest.approx(r["lv_edv_ml"] / 2)
    assert r["qa_status"] == "REVIEW_REQUIRED"


def test_volumes_require_every_seed(tmp_path):
    make_cine(tmp_path, [12], [9])
    stack = load(discover(tmp_path)[0])
    with pytest.raises(ValueError, match="missing on slices"):
        measure.ventricular_volumes(stack, [{"slice": 0, "label": "LV", "x": 70, "y": 64}])


def test_flow_integrates_known_velocity(tmp_path):
    # 12-bit phase image, rescale -> [-4096, 4094]; VENC 150 cm/s.
    # Systole frames at +75 cm/s, one diastolic frame at -15 cm/s, inside a 10-px disk.
    venc, frames = 150.0, [75, 75, 0, -15, 0]
    for p, v in enumerate(frames):
        stored = np.full(SHAPE, 2048, np.uint16)
        stored[disk(SHAPE, 64, 64, 10, 1, 0).astype(bool)] = round(2048 + v / venc * 2048)
        write(tmp_path / f"p{p}.dcm", stored, series_uid="1.39", series_number=39,
              description="Flow_Pulmonary_V150_P", trigger=p * 100.0, slope=2.0, intercept=-4096.0)
    phase = load(discover(tmp_path)[0])
    assert measure.venc_from_description(phase.series.description) == 150
    out = measure.flow_curve(phase, {"x": 64, "y": 64, "radius": 10}, venc, phase.pixel_area_mm2)
    area_cm2 = out["roi_pixels"] * 1.5 * 1.5 / 100
    rates = np.array(frames) * area_cm2
    mid = (rates + np.roll(rates, -1)) / 2 * 0.1
    assert out["forward_volume_ml"] == pytest.approx(mid[mid > 0].sum(), rel=0.01)
    assert out["reverse_volume_ml"] == pytest.approx(-mid[mid < 0].sum(), rel=0.01)
    combined = measure.flow(out, out)
    assert combined["qp_qs"] == pytest.approx(1.0)


def test_venc_parsing():
    assert measure.venc_from_description("Flow_Aorta_V200_P") == 200
    assert measure.venc_from_description("PC venc=250 through-plane") == 250
    assert measure.venc_from_description("SAX cine") is None


def test_ecv_formula():
    # textbook example: native myo 1000, blood 1600; post myo 450, blood 300; Hct 0.40
    expected = 0.6 * (1 / 450 - 1 / 1000) / (1 / 300 - 1 / 1600)
    assert measure.ecv(0.40, 1000, 1600, 450, 300) == pytest.approx(expected)
    with pytest.raises(ValueError):
        measure.ecv(42, 1000, 1600, 450, 300)


def test_t1_and_lge_roi_stats():
    native = np.full(SHAPE, 1000.0); native[:, :40] = 1600
    post = np.full(SHAPE, 450.0); post[:, :40] = 300
    rois = {"native_myocardium": {"x": 90, "y": 64, "radius": 5}, "native_blood": {"x": 15, "y": 64, "radius": 5},
            "post_myocardium": {"x": 90, "y": 64, "radius": 5}, "post_blood": {"x": 15, "y": 64, "radius": 5}}
    r = measure.t1(native, post, rois, hematocrit=0.4)
    assert r["ecv_percent"] == pytest.approx(100 * measure.ecv(0.4, 1000, 1600, 450, 300))

    rng = np.random.default_rng(0)
    image = rng.normal(100, 5, SHAPE); image[60:70, 20:30] = 400
    out = measure.lge(image, {"x": 25, "y": 65, "radius": 3}, {"x": 90, "y": 90, "radius": 8})
    assert out["target_fraction_above_threshold_percent"] == 100
    assert out["target_z_score"] > 20
