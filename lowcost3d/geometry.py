"""Stereo triangulation, plane fitting and rigid registration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from .calibration import project


def triangulate_linear(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation of corresponding points, as MATLAB ``triangulate``."""
    x1 = np.asarray(x1, dtype=float).reshape(-1, 2)
    x2 = np.asarray(x2, dtype=float).reshape(-1, 2)
    if x1.shape != x2.shape:
        raise ValueError(f"{len(x1)} points in view 1 but {len(x2)} in view 2")
    out = np.empty((len(x1), 3))
    for i, ((u1, v1), (u2, v2)) in enumerate(zip(x1, x2)):
        A = np.stack([
            u1 * P1[2] - P1[0],
            v1 * P1[2] - P1[1],
            u2 * P2[2] - P2[0],
            v2 * P2[2] - P2[1],
        ])
        # Row scaling improves conditioning without changing the solution set.
        A = A / np.linalg.norm(A[:, :3], axis=1, keepdims=True)
        _, _, vt = np.linalg.svd(A)
        X = vt[-1]
        out[i] = X[:3] / X[3]
    return out


def triangulate(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray, *,
                refine_lm: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate and return (points (N,3), mean reprojection error per point).

    With ``refine_lm`` the linear estimate is refined by minimising the
    reprojection error in both images (maximum-likelihood triangulation).
    The error is the mean of the two image distances, as in MATLAB.
    """
    X = triangulate_linear(P1, P2, x1, x2)
    x1 = np.asarray(x1, dtype=float).reshape(-1, 2)
    x2 = np.asarray(x2, dtype=float).reshape(-1, 2)
    if refine_lm:
        for i in range(len(X)):
            def res(p, i=i):
                return np.concatenate([
                    project(P1, p[None])[0] - x1[i],
                    project(P2, p[None])[0] - x2[i],
                ])
            X[i] = optimize.least_squares(res, X[i], method="lm").x
    err = 0.5 * (np.linalg.norm(project(P1, X) - x1, axis=1)
                 + np.linalg.norm(project(P2, X) - x2, axis=1))
    return X, err


@dataclass
class Plane:
    """Plane a*x + b*y + c*z + d = 0 with unit normal (a, b, c)."""

    normal: np.ndarray
    d: float
    centroid: np.ndarray

    def distances(self, points: np.ndarray) -> np.ndarray:
        """Signed point-to-plane distances."""
        return np.asarray(points) @ self.normal + self.d

    def as_dict(self) -> dict:
        a, b, c = self.normal
        return {"a": a, "b": b, "c": c, "d": self.d, "centroid": self.centroid}


def fit_plane(points: np.ndarray, orient_towards: np.ndarray | None = None) -> Plane:
    """Least-squares (total least squares) plane through 3 or more points.

    The normal sign is arbitrary in an SVD fit; if ``orient_towards`` (a 3-D
    point, e.g. a camera centre) is given the normal is flipped to point at it,
    so that results from repeated measurements are comparable.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError("a plane needs at least 3 points")
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    normal = vt[2]
    if orient_towards is not None and np.dot(np.asarray(orient_towards) - centroid, normal) < 0:
        normal = -normal
    return Plane(normal, float(-normal @ centroid), centroid)


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=2)


@dataclass
class RigidFit:
    """Rigid transform mapping model points onto measured points: X = R @ m + t."""

    R: np.ndarray
    t: np.ndarray
    residuals: np.ndarray  # per-marker distance after alignment

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2)))


def fit_rigid(model: np.ndarray, measured: np.ndarray) -> RigidFit:
    """Kabsch/Umeyama rigid registration (no scaling) of corresponding points."""
    model = np.asarray(model, dtype=float)
    measured = np.asarray(measured, dtype=float)
    if model.shape != measured.shape or model.shape[1] != 3:
        raise ValueError("model and measured points must both be (N, 3)")
    if len(model) < 3:
        raise ValueError("rigid registration needs at least 3 points")
    cm, cx = model.mean(axis=0), measured.mean(axis=0)
    H = (model - cm).T @ (measured - cx)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cx - R @ cm
    res = np.linalg.norm((model @ R.T + t) - measured, axis=1)
    return RigidFit(R, t, res)
