"""High-level calibration, localisation and parameter tuning."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict

import numpy as np

from . import calibration as cal
from .geometry import fit_plane, fit_rigid, pairwise_distances, triangulate
from .segmentation import SegmentationError, close_mask, region_grow, segment_markers


@dataclass
class SegmentationParams:
    max_dist: float
    disk_radius: int = 6
    fill_holes: bool = True
    max_area: int | None = 20000
    legacy_mean: bool = False


# Defaults of the original scripts (setupCalibrator.m / probePositionEstimator.m).
CALIBRATION_DEFAULTS = SegmentationParams(max_dist=0.04, disk_radius=6)
LOCATE_DEFAULTS = SegmentationParams(max_dist=0.10, disk_radius=6)

# Flag points whose reprojection error exceeds this many pixels.
WARN_REPROJECTION_PX = 3.0


def _check_pairs(seeds_left: np.ndarray, seeds_right: np.ndarray, n: int | None = None):
    if len(seeds_left) != len(seeds_right):
        raise ValueError(
            f"{len(seeds_left)} markers clicked in the left view but "
            f"{len(seeds_right)} in the right view"
        )
    if n is not None and len(seeds_left) != n:
        raise ValueError(f"expected {n} markers, got {len(seeds_left)}")


def segment(image, seeds, params: SegmentationParams):
    return segment_markers(image, seeds, max_dist=params.max_dist,
                           disk_radius=params.disk_radius, fill_holes=params.fill_holes,
                           max_area=params.max_area, legacy_mean=params.legacy_mean)


def calibrate(img_left, img_right, world, seeds_left, seeds_right,
              params: SegmentationParams = CALIBRATION_DEFAULTS, *, refine_lm: bool = True) -> dict:
    """Calibrate both cameras from one image pair of the calibration frame."""
    world = np.asarray(world, dtype=float)
    _check_pairs(seeds_left, seeds_right, len(world))
    if img_left.shape != img_right.shape:
        warnings.warn("left and right images differ in size")
    seg_l = segment(img_left, seeds_left, params)
    seg_r = segment(img_right, seeds_right, params)
    res_l = cal.calibrate_camera(seg_l.centroids, world, refine_lm=refine_lm)
    res_r = cal.calibrate_camera(seg_r.centroids, world, refine_lm=refine_lm)
    cams = {}
    for name, res in (("left", res_l), ("right", res_r)):
        c = res.camera
        cams[name] = {
            "P": res.P,
            "reprojection_errors_px": res.errors,
            "loo_errors_px": res.loo_errors,
            "center": c.center,
            "focal_px": c.focal_px,
            "principal_point_px": c.principal_point,
            "skew": c.skew,
        }
        # A wrong click order or a wrong row in the marker table shows up as a
        # large in-sample residual (tens of pixels); segmentation noise gives
        # well below 1-2 px.
        for k in np.nonzero(res.errors > WARN_REPROJECTION_PX)[0]:
            warnings.warn(f"{name} camera: marker {k + 1} reprojection error "
                          f"{res.errors[k]:.1f} px - check the click order and the "
                          "marker coordinates")
    all_err = np.concatenate([res_l.errors, res_r.errors])
    return {
        "image_size": list(img_left.shape[::-1]),
        "segmentation": asdict(params),
        "refine_lm": refine_lm,
        "world_points": world,
        "seeds_left": np.asarray(seeds_left), "seeds_right": np.asarray(seeds_right),
        "centroids_left": seg_l.centroids, "centroids_right": seg_r.centroids,
        "cameras": cams,
        "mean_reprojection_error_px": float(all_err.mean()),
        "mean_loo_error_px": float(np.nanmean(np.concatenate(
            [res_l.loo_errors, res_r.loo_errors]))) if res_l.loo_errors is not None else None,
        "_segmentations": (seg_l, seg_r),
    }


def locate(calib: dict, img_left, img_right, seeds_left, seeds_right,
           params: SegmentationParams = LOCATE_DEFAULTS, *, model=None,
           refine_lm: bool = True) -> dict:
    """Triangulate markers from an image pair using a stored calibration."""
    _check_pairs(seeds_left, seeds_right)
    size = list(img_left.shape[::-1])
    if calib.get("image_size") and list(calib["image_size"]) != size:
        raise ValueError(f"image size {size} differs from the calibration "
                         f"{calib['image_size']}")
    P_l = np.asarray(calib["cameras"]["left"]["P"])
    P_r = np.asarray(calib["cameras"]["right"]["P"])
    seg_l = segment(img_left, seeds_left, params)
    seg_r = segment(img_right, seeds_right, params)
    # NOTE: the original probePositionEstimator.m triangulated the raw clicks
    # (x1, y1) instead of the segmented centroids; here the centroids are used.
    X, err = triangulate(P_l, P_r, seg_l.centroids, seg_r.centroids, refine_lm=refine_lm)
    for k in np.nonzero(err > WARN_REPROJECTION_PX)[0]:
        warnings.warn(f"marker {k + 1}: reprojection error {err[k]:.1f} px - "
                      "were the markers clicked in the same order in both views?")
    out = {
        "segmentation": asdict(params),
        "seeds_left": np.asarray(seeds_left), "seeds_right": np.asarray(seeds_right),
        "centroids_left": seg_l.centroids, "centroids_right": seg_r.centroids,
        "points_mm": X,
        "reprojection_errors_px": err,
        "pairwise_distances_mm": pairwise_distances(X),
        "_segmentations": (seg_l, seg_r),
    }
    if len(X) >= 3:
        cam_mid = 0.5 * (np.asarray(calib["cameras"]["left"]["center"])
                         + np.asarray(calib["cameras"]["right"]["center"]))
        plane = fit_plane(X, orient_towards=cam_mid)
        dist = plane.distances(X)
        out["plane"] = plane.as_dict()
        out["coplanarity_errors_mm"] = dist
        out["coplanarity_rms_mm"] = float(np.sqrt(np.mean(dist**2)))
    if model is not None:
        model = np.asarray(model, dtype=float)
        if model.shape != X.shape:
            raise ValueError(f"model has {len(model)} points, measured {len(X)}")
        fit = fit_rigid(model, X)
        out["rigid_fit"] = {"R": fit.R, "t": fit.t, "residuals_mm": fit.residuals,
                            "rms_mm": fit.rms}
    return out


def tune(img_left, img_right, world, seeds_left, seeds_right, radii, dists, *,
         fill_holes: bool = True, max_area: int | None = 20000, refine_lm: bool = True) -> dict:
    """Grid search of (disk radius, region-growing threshold).

    For each combination the mean in-sample and leave-one-out reprojection
    errors (both cameras) are recorded; combinations where segmentation
    fails are NaN. The recommended setting minimises the leave-one-out error,
    which does not reward over-fitting the 11-parameter DLT to 8 points.
    """
    world = np.asarray(world, dtype=float)
    _check_pairs(seeds_left, seeds_right, len(world))
    radii = np.asarray(radii, dtype=int)
    dists = np.asarray(dists, dtype=float)
    insample = np.full((len(dists), len(radii)), np.nan)
    loo = np.full_like(insample, np.nan)
    for i, d in enumerate(dists):
        grown = []
        try:
            for img, seeds in ((img_left, seeds_left), (img_right, seeds_right)):
                grown.append([region_grow(img, (int(round(y)), int(round(x))), d,
                                          max_area=max_area) for x, y in seeds])
        except SegmentationError:
            continue
        for j, r in enumerate(radii):
            errs, loos = [], []
            for masks in grown:
                cents = []
                for m in masks:
                    m = close_mask(m, int(r), fill_holes=fill_holes)
                    rows, cols = np.nonzero(m)
                    cents.append((cols.mean(), rows.mean()))
                res = cal.calibrate_camera(np.array(cents), world, refine_lm=refine_lm)
                errs.append(res.errors)
                loos.append(res.loo_errors)
            insample[i, j] = np.mean(np.concatenate(errs))
            loo[i, j] = np.nanmean(np.concatenate(loos))
    if np.all(np.isnan(loo)):
        raise SegmentationError("segmentation failed for every parameter combination")
    bi, bj = np.unravel_index(np.nanargmin(loo), loo.shape)
    return {"radii": radii, "dists": dists, "mean_error_px": insample,
            "mean_loo_error_px": loo,
            "best": {"disk_radius": int(radii[bj]), "max_dist": float(dists[bi]),
                     "mean_loo_error_px": float(loo[bi, bj]),
                     "mean_error_px": float(insample[bi, bj])}}
