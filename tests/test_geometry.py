import numpy as np
import pytest

from lowcost3d.calibration import project
from lowcost3d.geometry import (fit_plane, fit_rigid, pairwise_distances, triangulate,
                                triangulate_linear)


def random_points(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform([50, 50, 0], [250, 300, 230], (n, 3))


def test_triangulation_exact(stereo):
    X = random_points()
    x1, x2 = project(stereo["P1"], X), project(stereo["P2"], X)
    np.testing.assert_allclose(triangulate_linear(stereo["P1"], stereo["P2"], x1, x2), X,
                               atol=1e-6)
    Xr, err = triangulate(stereo["P1"], stereo["P2"], x1, x2)
    np.testing.assert_allclose(Xr, X, atol=1e-6)
    assert err.max() < 1e-6


def test_refined_triangulation_reduces_reprojection_error(stereo):
    rng = np.random.default_rng(1)
    X = random_points()
    x1 = project(stereo["P1"], X) + rng.normal(0, 1, (20, 2))
    x2 = project(stereo["P2"], X) + rng.normal(0, 1, (20, 2))
    _, e_lin = triangulate(stereo["P1"], stereo["P2"], x1, x2, refine_lm=False)
    _, e_lm = triangulate(stereo["P1"], stereo["P2"], x1, x2)
    assert np.all(e_lm <= e_lin + 1e-9)


def test_mismatched_counts(stereo):
    with pytest.raises(ValueError):
        triangulate_linear(stereo["P1"], stereo["P2"], np.zeros((3, 2)), np.zeros((4, 2)))


def test_plane_fit_and_orientation():
    rng = np.random.default_rng(2)
    n = np.array([1.0, 2.0, 2.0]) / 3
    basis = np.linalg.svd(n[None])[2][1:]
    pts = rng.normal(size=(10, 2)) @ basis * 50 + 7 * n
    plane = fit_plane(pts, orient_towards=n * 100)
    np.testing.assert_allclose(plane.normal, n, atol=1e-10)
    np.testing.assert_allclose(plane.d, -7, atol=1e-10)
    assert np.abs(plane.distances(pts)).max() < 1e-10
    assert np.allclose(fit_plane(pts, orient_towards=-n * 100).normal, -n)


def test_rigid_fit():
    rng = np.random.default_rng(3)
    model = rng.normal(size=(5, 3)) * 50
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    t = np.array([10.0, -20, 30])
    fit = fit_rigid(model, model @ q.T + t)
    np.testing.assert_allclose(fit.R, q, atol=1e-10)
    np.testing.assert_allclose(fit.t, t, atol=1e-10)
    assert fit.rms < 1e-10


def test_pairwise_distances():
    d = pairwise_distances(np.array([[0, 0, 0], [3, 4, 0.0]]))
    np.testing.assert_allclose(d, [[0, 5], [5, 0]])
