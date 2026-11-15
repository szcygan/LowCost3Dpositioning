"""Camera projection matrix estimation (DLT) from 3-D/2-D correspondences.

Replaces MATLAB's ``estimateCameraProjection`` / ``estimateCameraMatrix``.
Compared with a plain DLT the estimate here

* uses Hartley's data normalisation (the plain DLT on pixel and millimetre
  coordinates is badly conditioned),
* is optionally refined by minimising the geometric reprojection error
  (Levenberg-Marquardt, the "gold standard" estimator),
* checks the input for degenerate configurations (coplanar or too few points),
* offers leave-one-out reprojection errors, an honest measure of calibration
  quality: with 8 markers and 11 unknowns the in-sample residual is small by
  construction and does not reveal a misplaced marker.

Projection matrices follow the usual 3x4 convention ``x ~ P @ [X, Y, Z, 1]``
with 0-based pixel coordinates (x = column, y = row).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize

MIN_POINTS = 6


def _normalisation(points: np.ndarray) -> np.ndarray:
    """Similarity transform moving the centroid to 0 and the RMS radius to sqrt(d)."""
    d = points.shape[1]
    centroid = points.mean(axis=0)
    rms = np.sqrt(((points - centroid) ** 2).sum(axis=1).mean())
    s = np.sqrt(d) / rms
    T = np.eye(d + 1)
    T[:d, :d] *= s
    T[:d, d] = -s * centroid
    return T


def _to_h(points: np.ndarray) -> np.ndarray:
    return np.hstack([points, np.ones((points.shape[0], 1))])


def project(P: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Project (N, 3) world points with a 3x4 matrix; returns (N, 2) pixels."""
    x = _to_h(np.asarray(world, dtype=float)) @ P.T
    return x[:, :2] / x[:, 2:3]


def reprojection_errors(P: np.ndarray, image_pts: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Euclidean distance (pixels) between observed and reprojected points."""
    return np.linalg.norm(project(P, world) - image_pts, axis=1)


def check_configuration(world: np.ndarray, tol: float = 1e-3) -> None:
    """Raise ValueError if the calibration points cannot determine P."""
    world = np.asarray(world, dtype=float)
    if world.ndim != 2 or world.shape[1] != 3:
        raise ValueError("world points must be an (N, 3) array")
    if world.shape[0] < MIN_POINTS:
        raise ValueError(f"at least {MIN_POINTS} calibration points are required")
    sv = np.linalg.svd(world - world.mean(axis=0), compute_uv=False)
    if sv[-1] / sv[0] < tol:
        raise ValueError("calibration points are (nearly) coplanar; DLT is degenerate")


def dlt(image_pts: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Normalised direct linear transform. Returns P with ||P||_F = 1."""
    image_pts = np.asarray(image_pts, dtype=float)
    world = np.asarray(world, dtype=float)
    if image_pts.shape[0] != world.shape[0]:
        raise ValueError(
            f"{image_pts.shape[0]} image points but {world.shape[0]} world points"
        )
    check_configuration(world)
    T2 = _normalisation(image_pts)
    T3 = _normalisation(world)
    xn = _to_h(image_pts) @ T2.T
    Xn = _to_h(world) @ T3.T
    rows = []
    for (u, v, w), X in zip(xn, Xn):
        u, v = u / w, v / w
        rows.append(np.concatenate([X, np.zeros(4), -u * X]))
        rows.append(np.concatenate([np.zeros(4), X, -v * X]))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Pn = vt[-1].reshape(3, 4)
    P = np.linalg.inv(T2) @ Pn @ T3
    return P / np.linalg.norm(P)


def refine(P0: np.ndarray, image_pts: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Minimise the sum of squared reprojection errors starting from ``P0``."""
    image_pts = np.asarray(image_pts, dtype=float)
    world = np.asarray(world, dtype=float)
    # Work in normalised coordinates for good conditioning.
    T2 = _normalisation(image_pts)
    T3 = _normalisation(world)
    Pn0 = T2 @ P0 @ np.linalg.inv(T3)
    Pn0 = Pn0 / np.linalg.norm(Pn0)
    xn = (_to_h(image_pts) @ T2.T)[:, :2]
    Xn = (_to_h(world) @ T3.T)[:, :3]

    def residuals(p):
        return (project(p.reshape(3, 4), Xn) - xn).ravel()

    method = "lm" if 2 * len(world) >= 12 else "trf"
    sol = optimize.least_squares(residuals, Pn0.ravel(), method=method,
                                 x_scale="jac", xtol=1e-12, ftol=1e-12)
    P = np.linalg.inv(T2) @ sol.x.reshape(3, 4) @ T3
    return P / np.linalg.norm(P)


def estimate_projection(image_pts: np.ndarray, world: np.ndarray, *, refine_lm: bool = True) -> np.ndarray:
    """Estimate a 3x4 projection matrix (normalised DLT, optional LM refinement)."""
    P = dlt(image_pts, world)
    if refine_lm:
        P = refine(P, image_pts, world)
    # fix the sign so that the calibration points lie in front of the camera
    depth = (_to_h(np.asarray(world, float)) @ P.T)[:, 2]
    if np.median(depth) < 0:
        P = -P
    return P


def leave_one_out_errors(image_pts: np.ndarray, world: np.ndarray, *, refine_lm: bool = True) -> np.ndarray:
    """Reprojection error of each point when it is excluded from the fit.

    Requires at least ``MIN_POINTS + 1`` points; returns NaN for a point whose
    removal leaves a degenerate configuration.
    """
    image_pts = np.asarray(image_pts, dtype=float)
    world = np.asarray(world, dtype=float)
    n = len(world)
    if n <= MIN_POINTS:
        raise ValueError(f"leave-one-out needs at least {MIN_POINTS + 1} points")
    out = np.full(n, np.nan)
    for i in range(n):
        keep = np.arange(n) != i
        try:
            P = estimate_projection(image_pts[keep], world[keep], refine_lm=refine_lm)
        except ValueError:
            continue
        out[i] = reprojection_errors(P, image_pts[i:i + 1], world[i:i + 1])[0]
    return out


@dataclass
class CameraParameters:
    """Decomposition P = K [R | -R C]."""

    K: np.ndarray
    R: np.ndarray
    center: np.ndarray

    @property
    def focal_px(self) -> tuple[float, float]:
        return float(self.K[0, 0]), float(self.K[1, 1])

    @property
    def principal_point(self) -> tuple[float, float]:
        return float(self.K[0, 2]), float(self.K[1, 2])

    @property
    def skew(self) -> float:
        return float(self.K[0, 1])


def decompose(P: np.ndarray) -> CameraParameters:
    """Split a projection matrix into intrinsics, rotation and camera centre."""
    M = P[:, :3]
    if np.linalg.det(M) < 0:
        P = -P
        M = -M
    K, R = linalg.rq(M)
    S = np.diag(np.sign(np.diag(K)))
    K, R = K @ S, S @ R
    K = K / K[2, 2]
    center = -np.linalg.solve(M, P[:, 3])
    return CameraParameters(K, R, center)


@dataclass
class CalibrationResult:
    P: np.ndarray
    errors: np.ndarray
    loo_errors: np.ndarray | None

    @property
    def camera(self) -> CameraParameters:
        return decompose(self.P)


def calibrate_camera(image_pts: np.ndarray, world: np.ndarray, *, refine_lm: bool = True,
                     with_loo: bool = True) -> CalibrationResult:
    P = estimate_projection(image_pts, world, refine_lm=refine_lm)
    err = reprojection_errors(P, image_pts, world)
    loo = None
    if with_loo and len(world) > MIN_POINTS:
        loo = leave_one_out_errors(image_pts, world, refine_lm=refine_lm)
    return CalibrationResult(P, err, loo)
