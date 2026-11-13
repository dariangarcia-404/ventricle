"""Command-line interface: ``ventricle <command> INPUT ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__, measure, render
from .measure import DISCLAIMER
from .series import Stack, by_number, discover, load


def _write(out: Path, name: str, data) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def _rois(args, pick):
    """Load ROIs from --rois, or open the picker and save what was clicked."""
    if args.rois:
        return json.loads(Path(args.rois).read_text())
    rois = pick()
    print(f"saved picks to {_write(args.out, 'rois.json', rois)} (re-run with --rois to reproduce)")
    return rois


def _slice(stack: Stack, index) -> int:
    s = stack.slices // 2 if index is None else index
    if not 0 <= s < stack.slices:
        raise SystemExit(f"--slice must be between 0 and {stack.slices - 1}")
    return s


def _brightest_phase(stack: Stack) -> np.ndarray:
    frames = stack.values[:, 0]
    return frames[int(np.argmax([np.percentile(f, 99) for f in frames]))]


def _finish(args, result: dict) -> None:
    path = _write(args.out, "results.json", result)
    print(json.dumps(result, indent=2))
    print(f"\nresults: {path}\n{DISCLAIMER}", file=sys.stderr)


# ----------------------------------------------------------------------------- commands

def cmd_inventory(args):
    series = discover(args.input)
    if not series:
        raise SystemExit(f"no uncompressed DICOM images found under {args.input}")
    rows = [{"series": s.number, "description": s.description, "images": len(s.files),
             "image_type": s.image_type, "rows": s.meta.get("Rows"), "columns": s.meta.get("Columns")}
            for s in series]
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    width = max(len(r["description"]) for r in rows)
    print(f"{'#':>4}  {'description':<{width}}  {'images':>6}  type")
    for r in rows:
        print(f"{r['series']:>4}  {r['description']:<{width}}  {r['images']:>6}  {r['image_type']}")


def cmd_preview(args):
    series = discover(args.input)
    chosen = series if args.series is None else [by_number(series, n) for n in args.series]
    for s in chosen:
        stack = load(s)
        by = args.by or ("phase" if stack.slices == 1 else "slice")
        print(render.contact_sheet(stack, args.out / f"series_{s.number:03d}_{by}.png", by=by))


def cmd_volume(args):
    stack = load(by_number(discover(args.input), args.series))
    if args.rois:
        seeds = json.loads(Path(args.rois).read_text())["seeds"]
    else:
        from .gui import pick_seeds
        seeds = pick_seeds([stack.values[0, s] for s in range(stack.slices)])
        _write(args.out, "rois.json", {"seeds": seeds})
    render.seed_overlay(stack, seeds, args.out / "seeds.png")
    _finish(args, measure.ventricular_volumes(stack, seeds, bsa_m2=args.bsa))


def _vessel(args, series, name, numbers, venc, rois):
    mag, phase = (load(by_number(series, n)) for n in numbers)
    v = venc or measure.venc_from_description(phase.series.description)
    if v is None:
        raise SystemExit(f"could not read VENC from {phase.series.label!r}; pass --venc-{name}")
    return mag, measure.flow_curve(phase, rois[name], v, phase.pixel_area_mm2, rois.get(f"{name}_background"))


def cmd_flow(args):
    series = discover(args.input)
    vessels = [("pulmonary", args.pulmonary, args.venc_pulmonary)]
    if args.aorta:
        vessels.append(("aorta", args.aorta, args.venc_aorta))
    mags = {name: load(by_number(series, nums[0])) for name, nums, _ in vessels}

    def pick():
        from .gui import pick_circles
        rois = {}
        for name, _, _ in vessels:
            rois.update(pick_circles(f"{name} flow ROI", _brightest_phase(mags[name]), [name, f"{name}_background"]))
        return rois

    rois = _rois(args, pick)
    curves = {}
    for name, nums, venc in vessels:
        _, curves[name] = _vessel(args, series, name, nums, venc, rois)
        shown = {k: v for k, v in rois.items() if k.startswith(name)}
        render.roi_overlay(_brightest_phase(mags[name]), shown, args.out / f"{name}_roi.png")
    _finish(args, measure.flow(curves["pulmonary"], curves.get("aorta")))


def cmd_t1(args):
    series = discover(args.input)
    native, post = load(by_number(series, args.native)), load(by_number(series, args.post))
    s = _slice(native, args.slice)
    n_img, p_img = native.values[0, s], post.values[0, min(s, post.slices - 1)]

    def pick():
        from .gui import pick_circles
        return {**pick_circles("native T1 map", n_img, ["native_myocardium", "native_blood"]),
                **pick_circles("post-contrast T1 map", p_img, ["post_myocardium", "post_blood"])}

    rois = _rois(args, pick)
    render.roi_overlay(n_img, {k: v for k, v in rois.items() if k.startswith("native")}, args.out / "native_roi.png")
    render.roi_overlay(p_img, {k: v for k, v in rois.items() if k.startswith("post")}, args.out / "post_roi.png")
    result = measure.t1(n_img, p_img, rois, hematocrit=args.hematocrit)
    _finish(args, {**result, "native_series": native.series.label, "post_series": post.series.label, "slice": s})


def cmd_lge(args):
    stack = load(by_number(discover(args.input), args.series))
    s = _slice(stack, args.slice)
    image = stack.values[0, s]

    def pick():
        from .gui import pick_circles
        return pick_circles("LGE: target then remote myocardium", image, ["target", "remote"])

    rois = _rois(args, pick)
    render.roi_overlay(image, rois, args.out / "lge_roi.png")
    result = measure.lge(image, rois["target"], rois["remote"], n_sd=args.n_sd)
    _finish(args, {**result, "series": stack.series.label, "slice": s})


# ----------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ventricle",
        description=f"Research-only cardiac MRI measurement toolkit. {DISCLAIMER}",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, func, help_text, out=True):
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("input", type=Path, help="folder, mounted CD, or single DICOM file")
        if out:
            p.add_argument("--out", type=Path, default=Path(f"ventricle-output/{name}"), help="output folder")
        p.set_defaults(func=func)
        return p

    p = command("inventory", cmd_inventory, "list the series in a study", out=False)
    p.add_argument("--json", action="store_true", help="print JSON instead of a table")

    p = command("preview", cmd_preview, "save contact-sheet PNGs of one or more series")
    p.add_argument("--series", type=int, nargs="+", help="series numbers (default: all)")
    p.add_argument("--by", choices=["slice", "phase"], help="tile across slices or cine phases")

    p = command("volume", cmd_volume, "LV/RV volumes and ejection fraction from a short-axis cine")
    p.add_argument("--series", type=int, required=True, help="short-axis cine series number")
    p.add_argument("--bsa", type=float, help="body-surface area (m^2) for indexed volumes")
    p.add_argument("--rois", type=Path, help="seeds JSON (skip the picker)")

    p = command("flow", cmd_flow, "phase-contrast flow, regurgitant fraction, and Qp:Qs")
    p.add_argument("--pulmonary", type=int, nargs=2, required=True, metavar=("MAG", "PHASE"), help="pulmonary magnitude and phase series")
    p.add_argument("--aorta", type=int, nargs=2, metavar=("MAG", "PHASE"), help="aortic magnitude and phase series (enables Qp:Qs)")
    p.add_argument("--venc-pulmonary", type=float, help="override VENC (cm/s)")
    p.add_argument("--venc-aorta", type=float, help="override VENC (cm/s)")
    p.add_argument("--rois", type=Path, help="ROI JSON (skip the picker)")

    p = command("t1", cmd_t1, "native/post-contrast T1 ROI values and ECV")
    p.add_argument("--native", type=int, required=True, help="native T1 map series")
    p.add_argument("--post", type=int, required=True, help="post-contrast T1 map series")
    p.add_argument("--slice", type=int, help="slice index (default: middle)")
    p.add_argument("--hematocrit", type=float, help="fraction, e.g. 0.42, to compute ECV")
    p.add_argument("--rois", type=Path, help="ROI JSON (skip the picker)")

    p = command("lge", cmd_lge, "late-gadolinium-enhancement target vs remote ROI comparison")
    p.add_argument("--series", type=int, required=True, help="LGE / PSIR series number")
    p.add_argument("--slice", type=int, help="slice index (default: middle)")
    p.add_argument("--n-sd", type=float, default=5.0, help="threshold = remote mean + N SD (default 5)")
    p.add_argument("--rois", type=Path, help="ROI JSON (skip the picker)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (LookupError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
