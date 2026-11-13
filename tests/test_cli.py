import json

from ventricle.cli import main
from synthetic import disk, write
from test_measure import SHAPE, make_cine


def test_inventory_and_preview(tmp_path, capsys):
    make_cine(tmp_path / "study", [12, 9], [10, 8])
    assert main(["inventory", str(tmp_path / "study")]) == 0
    assert "SAX cine" in capsys.readouterr().out
    assert main(["preview", str(tmp_path / "study"), "--out", str(tmp_path / "png")]) == 0
    assert (tmp_path / "png" / "series_010_slice.png").exists()


def test_volume_from_seed_file_is_reproducible(tmp_path):
    make_cine(tmp_path / "study", [12, 9], [10, 8])
    seeds = [{"slice": s, "label": l, "x": x, "y": 64} for s in range(3) for l, x in (("LV", 70), ("RV", 35))]
    (tmp_path / "seeds.json").write_text(json.dumps({"seeds": seeds}))
    out = tmp_path / "out"
    assert main(["volume", str(tmp_path / "study"), "--series", "10", "--rois", str(tmp_path / "seeds.json"), "--out", str(out)]) == 0
    result = json.loads((out / "results.json").read_text())
    assert result["lv_edv_ml"] > result["lv_esv_ml"] > 0
    assert (out / "seeds.png").exists()
    assert str(tmp_path) not in (out / "results.json").read_text()  # no local paths in outputs


def test_lge_and_errors(tmp_path, capsys):
    write(tmp_path / "lge" / "a.dcm", disk(SHAPE, 30, 30, 4, value=900), series_number=5, series_uid="1.5", description="PSIR")
    rois = tmp_path / "rois.json"
    rois.write_text(json.dumps({"target": {"x": 30, "y": 30, "radius": 3}, "remote": {"x": 90, "y": 90, "radius": 6}}))
    assert main(["lge", str(tmp_path / "lge"), "--series", "5", "--rois", str(rois), "--out", str(tmp_path / "o")]) == 0
    assert main(["lge", str(tmp_path / "lge"), "--series", "99", "--rois", str(rois)]) == 1
    assert "no series #99" in capsys.readouterr().err
