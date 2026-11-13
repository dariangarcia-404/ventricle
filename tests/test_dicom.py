import numpy as np
import pytest

from ventricle import dicom
from synthetic import write


@pytest.mark.parametrize("explicit", [True, False])
def test_reads_tags_and_pixels(tmp_path, explicit):
    pixels = np.arange(12, dtype=np.uint16).reshape(3, 4)
    path = write(tmp_path / "a.dcm", pixels, explicit=explicit, description="SAX cine",
                 series_number=7, trigger=42.5, slope=2.0, intercept=-10.0)
    meta = dicom.read(path, pixels=True)
    assert meta["SeriesDescription"] == "SAX cine"  # not the decoy inside the sequence
    assert meta["SeriesNumber"] == 7
    assert meta["TriggerTime"] == 42.5
    assert meta["PixelSpacing"] == [1.5, 1.5]
    np.testing.assert_array_equal(meta["array"], pixels)
    np.testing.assert_allclose(dicom.rescale(meta, meta["array"]), pixels * 2.0 - 10.0)


def test_never_reads_patient_identifiers(tmp_path):
    meta = dicom.read(write(tmp_path / "a.dcm", np.zeros((2, 2), np.uint16)), pixels=True)
    assert "DOE" not in repr({k: v for k, v in meta.items() if k != "array"})


def test_rejects_compressed_and_non_dicom(tmp_path):
    jpeg = write(tmp_path / "j.dcm", np.zeros((2, 2), np.uint16), syntax="1.2.840.10008.1.2.4.50")
    with pytest.raises(dicom.DicomError, match="transfer syntax"):
        dicom.read(jpeg)
    junk = tmp_path / "junk.dcm"
    junk.write_bytes(b"hello")
    with pytest.raises(dicom.DicomError):
        dicom.read(junk)
    assert not dicom.is_dicom(junk)
