# ROI file formats

Coordinates are in original image pixels (x = column, y = row, origin top-left).
The interactive pickers write these files for you as `rois.json`. Pass one back
with `--rois` to reproduce a measurement exactly.

| Command | File | Keys |
| --- | --- | --- |
| `volume` | [`volume-seeds.json`](volume-seeds.json) | `seeds`: one `LV` and one `RV` point per slice (0-based `slice`) |
| `flow` | [`flow-rois.json`](flow-rois.json) | `pulmonary`, optional `aorta`, optional `<vessel>_background` circles |
| `t1` | [`t1-rois.json`](t1-rois.json) | `native_myocardium`, `native_blood`, `post_myocardium`, `post_blood` circles |
| `lge` | [`lge-rois.json`](lge-rois.json) | `target` and `remote` circles |

The values here are placeholders, not taken from any real study.
