# ventricle

**Small, auditable cardiac MRI (CMR) measurements from the DICOM files on your scan CD.**

> [!WARNING]
> **Research and educational use only. This is not a medical device.** It has not been
> validated, cleared, or approved by the FDA or any other regulator. Do not use it to
> diagnose, treat, or make decisions about anyone's care. Every result is marked
> `REVIEW_REQUIRED`. Compare anything it produces with a clinical CMR report read by a
> qualified physician, and bring questions to your care team.

Patients often leave a cardiac MRI with a CD of DICOM images and a long wait for the
report. `ventricle` lets a curious, technical reader see *how* the standard CMR numbers
are calculated from those images. Every number comes with the clicks that produced it
and an overlay image, so you can check it yourself.

| Command | What it estimates | Input series |
| --- | --- | --- |
| `inventory` | Lists every series on the disc | any |
| `preview` | Contact-sheet PNGs across slices or cine phases | any |
| `volume` | LV/RV end-diastolic and end-systolic volume, stroke volume, ejection fraction, RV/LV ratio, indexed volumes | short-axis cine stack |
| `flow` | Forward and reverse flow volume, regurgitant fraction, Qp:Qs | phase-contrast magnitude and phase |
| `t1` | Native and post-contrast myocardial and blood T1, extracellular volume (ECV) | scanner T1 maps |
| `lge` | Late-gadolinium-enhancement target vs. remote myocardium (ratio, z-score, % above mean + *n* SD) | LGE / PSIR |

## Install

Requires Python 3.10+ with NumPy and Pillow. The interactive pickers use Tk, which ships
with python.org and most Linux Python builds. With Homebrew Python, also run
`brew install python-tk`.

```bash
pip install git+https://github.com/dariangarcia-404/ventricle
```

## Use

Mount the CD (or copy it to a folder) and start by listing what's on it:

```bash
ventricle inventory /Volumes/MY_CD
```

```
   #  description                       images  type
  10  cine_SAX                             300  ORIGINAL\PRIMARY\M\...
  32  Flow_Aorta_V200_MAG                   30  ...
  33  Flow_Aorta_V200_P                     30  ...
  38  Flow_Pulmonary_V150_MAG               30  ...
  39  Flow_Pulmonary_V150_P                 30  ...
  ...
```

Series names vary by scanner vendor and site, so you pick the series numbers yourself:

```bash
ventricle preview /Volumes/MY_CD --series 10 --by phase      # look before measuring
ventricle volume  /Volumes/MY_CD --series 10 --bsa 1.9        # click LV then RV on every slice
ventricle flow    /Volumes/MY_CD --pulmonary 38 39 --aorta 32 33
ventricle t1      /Volumes/MY_CD --native 57 --post 64 --hematocrit 0.42
ventricle lge     /Volumes/MY_CD --series 86
```

Each command opens a simple picker window, writes to `ventricle-output/<command>/`, and saves:

- `results.json` with the numbers, method, limitations, and `qa_status`
- `rois.json` with exactly what you clicked
- one or more `*.png` overlays showing where the ROIs or seeds landed

Re-run any measurement exactly, without the GUI, by passing the saved clicks back:

```bash
ventricle flow /Volumes/MY_CD --pulmonary 38 39 --aorta 32 33 --rois ventricle-output/flow/rois.json
```

See [`examples/`](examples/) for the ROI file formats.

## How the numbers are calculated

- **Volumes (Simpson's method).** On each short-axis slice, one click per ventricle seeds a
  bright-blood region grown inside a bounded disc. Volume = Σ area × slice spacing. EDV and ESV
  are the largest and smallest volumes over the cine. EF = (EDV − ESV) / EDV. Indexed values
  divide by BSA.
- **Flow.** Phase pixels are mapped linearly onto ±VENC using the series' stored range.
  The VENC is read from names like `V150` or passed with `--venc-*`. Then
  flow(t) = Σ velocity × pixel area, integrated over the periodic cine loop.
  Regurgitant fraction = reverse / forward. Qp:Qs = pulmonary forward / aortic forward.
  An optional background ROI subtracts phase offset.
- **ECV.** ECV = (1 − Hct) × (1/T1<sub>myo,post</sub> − 1/T1<sub>myo,native</sub>) /
  (1/T1<sub>blood,post</sub> − 1/T1<sub>blood,native</sub>), using ROI medians.
- **LGE.** Target and remote circular ROIs are compared by mean ratio, z-score, and the fraction
  of target pixels above remote mean + *n*·SD (default 5).

## Limitations (please read)

- A click and a circle are not expert contours. Basal-slice selection, trabeculation and papillary
  conventions, through-plane motion, partial volume, and artifacts can all change results
  materially.
- Seeds for `volume` are placed on one phase and reused for every phase.
- `flow` cannot know which sign is antegrade, so it takes the dominant direction as forward.
- T1 and ECV reference ranges depend on scanner, field strength, and sequence. There is no
  universal "normal."
- `lge` cannot rule enhancement in or out. Pattern and distribution, which need a trained reader,
  matter more than intensity.
- Only uncompressed DICOM (implicit or explicit VR little endian) is read. Convert compressed
  studies first, for example with `gdcmconv --raw`.
- It reads MR images only. Findings from other tests (such as an ECG) are out of scope.

## Privacy

Your scans are yours. `ventricle` runs entirely on your machine and makes no network calls.
Its DICOM reader decodes only a short allow-list of technical tags (geometry, timing, pixel
scaling). It never reads patient name, ID, birth date, or other identifying fields, so they
can't appear in its outputs. Output files don't record your local paths.

**Don't commit real DICOM files or outputs to a public repository.** The included
`.gitignore` blocks common DICOM and output paths. If you open an issue, reproduce it with
synthetic data. The test suite shows how to generate it in [`tests/synthetic.py`](tests/synthetic.py).

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'
pytest
```

The tests use synthetic DICOM files generated on the fly, so no patient data is needed or included.

## License

Copyright 2026 Darian Garcia. Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE).

Apache 2.0 was chosen for its explicit patent grant and its "AS IS" warranty disclaimer, which
matter for health-adjacent software. Runtime dependencies are NumPy (BSD-3-Clause) and Pillow
(MIT-CMU). Both are compatible with Apache 2.0 and are installed separately, not bundled.
