"""Command-line interface: ``lowcost3d calibrate | locate | tune``."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

from . import io
from . import pipeline as pl

_REF_IMAGE = Path(__file__).resolve().parent.parent / "data" / "markingOrder.png"


def _ask_file(title: str) -> str:
    """File dialog fallback (like MATLAB's uigetfile) when no path is given."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:  # pragma: no cover
        sys.exit(f"{title}: no path given and tkinter is not available")
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title,
                                      filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif"),
                                                 ("All files", "*.*")])
    root.destroy()
    if not path:
        sys.exit("cancelled")
    return path


def _range(text: str, as_int: bool = False) -> np.ndarray:
    """Parse 'start:stop:step' (inclusive stop) or a comma-separated list."""
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        start, stop = parts[0], parts[1]
        step = parts[2] if len(parts) > 2 else 1.0
        vals = np.arange(start, stop + step / 2, step)
    else:
        vals = np.array([float(p) for p in text.split(",")])
    return vals.astype(int) if as_int else np.round(vals, 6)


def _seeds(args, img_left, img_right, n_expected, label):
    """Seeds from CSV files or interactive clicking; optionally saved for reuse."""
    if args.seeds_left and args.seeds_right:
        sl, sr = io.load_points(args.seeds_left)[:, :2], io.load_points(args.seeds_right)[:, :2]
    elif args.seeds_left or args.seeds_right:
        sys.exit("give both --seeds-left and --seeds-right, or neither")
    else:
        from .picking import pick_points
        ref = None
        if label == "calibration" and _REF_IMAGE.exists():
            import matplotlib.image as mpimg
            ref = mpimg.imread(_REF_IMAGE)
        sl = pick_points(img_left, f"LEFT camera - click the {label} markers", n_expected, ref)
        sr = pick_points(img_right, f"RIGHT camera - click the same markers in the SAME order",
                         len(sl), ref)
    if getattr(args, "save_seeds", None):
        prefix = Path(args.save_seeds)
        io.save_points(f"{prefix}_left.csv", sl)
        io.save_points(f"{prefix}_right.csv", sr)
        print(f"seeds saved to {prefix}_left.csv / {prefix}_right.csv")
    return sl, sr


def _seg_params(args, defaults: pl.SegmentationParams) -> pl.SegmentationParams:
    return pl.SegmentationParams(
        max_dist=args.max_dist if args.max_dist is not None else defaults.max_dist,
        disk_radius=args.disk_radius if args.disk_radius is not None else defaults.disk_radius,
        fill_holes=not args.no_fill_holes,
        max_area=args.max_area,
        legacy_mean=args.legacy_mean,
    )


def _fmt(a, digits=2):
    return np.array2string(np.asarray(a, dtype=float), precision=digits, suppress_small=True,
                           max_line_width=120)


def _public(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def cmd_calibrate(args):
    left = args.left or _ask_file("Select LEFT calibration image")
    right = args.right or _ask_file("Select RIGHT calibration image")
    world = io.load_points(args.markers)[:, :3]
    img_l, img_r = io.load_gray(left), io.load_gray(right)
    sl, sr = _seeds(args, img_l, img_r, len(world), "calibration")
    params = _seg_params(args, pl.CALIBRATION_DEFAULTS)
    res = pl.calibrate(img_l, img_r, world, sl, sr, params, refine_lm=not args.no_refine)
    res.update(left_image=str(left), right_image=str(right), markers_file=str(args.markers))
    for name in ("left", "right"):
        cam = res["cameras"][name]
        print(f"\n{name.upper()} camera")
        print("  reprojection errors [px]:", _fmt(cam["reprojection_errors_px"]))
        if cam["loo_errors_px"] is not None:
            print("  leave-one-out errors [px]:", _fmt(cam["loo_errors_px"]))
        print("  camera centre [mm]:", _fmt(cam["center"], 1),
              " focal [px]:", _fmt(cam["focal_px"], 0))
    print(f"\nMean reprojection error: {res['mean_reprojection_error_px']:.3f} px")
    if res["mean_loo_error_px"] is not None:
        print(f"Mean leave-one-out error: {res['mean_loo_error_px']:.3f} px")
    io.save_json(args.out, _public(res))
    print(f"Calibration saved to {args.out}")
    _maybe_show(args, res, img_l, img_r, "Calibration")


def cmd_locate(args):
    calib = io.load_json(args.calibration)
    left = args.left or _ask_file("Select LEFT image")
    right = args.right or _ask_file("Select RIGHT image")
    img_l, img_r = io.load_gray(left), io.load_gray(right)
    model = io.load_points(args.model)[:, :3] if args.model else None
    n = len(model) if model is not None else None
    sl, sr = _seeds(args, img_l, img_r, n, "probe")
    params = _seg_params(args, pl.LOCATE_DEFAULTS)
    res = pl.locate(calib, img_l, img_r, sl, sr, params, model=model,
                    refine_lm=not args.no_refine)
    res.update(left_image=str(left), right_image=str(right),
               calibration_file=str(args.calibration))
    print("Marker positions [mm] (X, Y, Z):")
    for k, p in enumerate(res["points_mm"], 1):
        print(f"  {k}: {p[0]:9.2f} {p[1]:9.2f} {p[2]:9.2f}   "
              f"reprojection error {res['reprojection_errors_px'][k - 1]:.2f} px")
    print("Pairwise distances [mm]:\n", _fmt(res["pairwise_distances_mm"]))
    if "plane" in res:
        p = res["plane"]
        print(f"Plane: {p['a']:.5f} x + {p['b']:.5f} y + {p['c']:.5f} z + {p['d']:.3f} = 0")
        print("Coplanarity errors [mm]:", _fmt(res["coplanarity_errors_mm"], 3),
              f" RMS {res['coplanarity_rms_mm']:.3f}")
    if "rigid_fit" in res:
        rf = res["rigid_fit"]
        print("Rigid fit to model, residuals [mm]:", _fmt(rf["residuals_mm"], 3),
              f" RMS {rf['rms_mm']:.3f}")
    io.save_json(args.out, _public(res))
    print(f"Results saved to {args.out}")
    _maybe_show(args, res, img_l, img_r, "Markers")


def cmd_tune(args):
    world = io.load_points(args.markers)[:, :3]
    img_l, img_r = io.load_gray(args.left), io.load_gray(args.right)
    sl, sr = _seeds(args, img_l, img_r, len(world), "calibration")
    radii = _range(args.radii, as_int=True)
    dists = _range(args.dists)
    print(f"Testing {len(radii)} x {len(dists)} parameter combinations...")
    res = pl.tune(img_l, img_r, world, sl, sr, radii, dists,
                  fill_holes=not args.no_fill_holes, max_area=args.max_area)
    b = res["best"]
    print(f"Lowest leave-one-out error {b['mean_loo_error_px']:.3f} px for "
          f"max_dist={b['max_dist']}, disk_radius={b['disk_radius']} "
          f"(in-sample {b['mean_error_px']:.3f} px)")
    if args.out:
        io.save_json(args.out, res)
    if args.plot or not args.no_show:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, key, title in ((axes[0], "mean_error_px", "In-sample reprojection error [px]"),
                               (axes[1], "mean_loo_error_px", "Leave-one-out error [px]")):
            im = ax.imshow(res[key], aspect="auto", origin="lower",
                           extent=[radii[0] - .5, radii[-1] + .5,
                                   dists[0] - (dists[1] - dists[0]) / 2 if len(dists) > 1 else dists[0] - .005,
                                   dists[-1] + (dists[1] - dists[0]) / 2 if len(dists) > 1 else dists[0] + .005])
            ax.set_xlabel("Disk radius [px]")
            ax.set_ylabel("Region growing limit")
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        axes[1].plot(b["disk_radius"], b["max_dist"], "r*", ms=14)
        fig.tight_layout()
        if args.plot:
            fig.savefig(args.plot, dpi=120)
            print(f"Plot saved to {args.plot}")
        if not args.no_show:
            plt.show()


def _maybe_show(args, res, img_l, img_r, title):
    if args.plot or not args.no_show:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        from .picking import show_segmentation
        seg_l, seg_r = res["_segmentations"]
        show_segmentation([("Left", img_l, seg_l), ("Right", img_r, seg_r)], title,
                          block=not args.no_show, save=args.plot)


def _add_common(p, defaults: pl.SegmentationParams | None):
    p.add_argument("--left", help="left camera image (file dialog if omitted)")
    p.add_argument("--right", help="right camera image (file dialog if omitted)")
    p.add_argument("--seeds-left", help="CSV with x,y seed points in the left image "
                   "(skips clicking)")
    p.add_argument("--seeds-right", help="CSV with x,y seed points in the right image")
    p.add_argument("--save-seeds", metavar="PREFIX",
                   help="save the clicked seeds as PREFIX_left.csv / PREFIX_right.csv")
    if defaults is not None:
        p.add_argument("--max-dist", type=float, default=None,
                       help=f"region growing limit (default {defaults.max_dist})")
        p.add_argument("--disk-radius", type=int, default=None,
                       help=f"closing disk radius in px (default {defaults.disk_radius})")
        p.add_argument("--no-refine", action="store_true",
                       help="skip Levenberg-Marquardt refinement (plain DLT, as in MATLAB)")
        p.add_argument("--legacy-mean", action="store_true",
                       help="use the original (biased) region-mean update")
        p.add_argument("--plot", help="save the segmentation overlay to this image file")
        p.add_argument("--no-show", action="store_true", help="do not open figure windows")
    p.add_argument("--no-fill-holes", action="store_true",
                   help="do not fill holes in marker masks (original behaviour)")
    p.add_argument("--max-area", type=int, default=20000,
                   help="fail if a marker region exceeds this many pixels (default 20000)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lowcost3d", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("calibrate", help="estimate both camera projection matrices")
    _add_common(p, pl.CALIBRATION_DEFAULTS)
    p.add_argument("--markers", required=True,
                   help="calibration frame marker coordinates (.xlsx or .csv, X,Y,Z in mm)")
    p.add_argument("--out", default="calibration.json")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("locate", help="triangulate markers (probe or validation board)")
    _add_common(p, pl.LOCATE_DEFAULTS)
    p.add_argument("--calibration", default="calibration.json")
    p.add_argument("--model", help="optional CSV with the nominal marker coordinates "
                   "(same order); reports the rigid-fit residuals and pose")
    p.add_argument("--out", default="triangulation.json")
    p.set_defaults(func=cmd_locate)

    p = sub.add_parser("tune", help="grid search of segmentation parameters")
    _add_common(p, None)
    p.add_argument("--markers", required=True)
    p.add_argument("--radii", default="1:8", help="disk radii, 'start:stop[:step]' or list")
    p.add_argument("--dists", default="0.01:0.2:0.01", help="region growing limits")
    p.add_argument("--out", help="save the error tables as JSON")
    p.add_argument("--plot", help="save the error maps to this image file")
    p.add_argument("--no-show", action="store_true")
    p.set_defaults(func=cmd_tune)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "tune" and not (args.left and args.right):
        sys.exit("tune needs --left and --right")
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.formatwarning = lambda msg, *a, **k: f"WARNING: {msg}\n"
        args.func(args)


if __name__ == "__main__":
    main()
