import numpy as np
import pytest

from lowcost3d import calibration as cal
from conftest import FRAME


def test_exact_recovery(stereo):
    x = cal.project(stereo["P1"], FRAME)
    for refine in (False, True):
        P = cal.estimate_projection(x, FRAME, refine_lm=refine)
        assert cal.reprojection_errors(P, x, FRAME).max() < 1e-6
        test_pts = np.array([[100, 100, 100], [200, 250, 50.0]])
        np.testing.assert_allclose(cal.project(P, test_pts),
                                   cal.project(stereo["P1"], test_pts), atol=1e-6)


def test_decomposition(stereo):
    x = cal.project(stereo["P1"], FRAME)
    cam = cal.decompose(cal.estimate_projection(x, FRAME))
    np.testing.assert_allclose(cam.K, stereo["K1"], atol=1e-5)
    np.testing.assert_allclose(cam.R, stereo["R1"], atol=1e-8)
    np.testing.assert_allclose(cam.center, stereo["C1"], atol=1e-6)


def test_refinement_does_not_increase_error(stereo):
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = cal.project(stereo["P1"], FRAME) + rng.normal(0, 0.5, (8, 2))
        e_dlt = cal.reprojection_errors(cal.estimate_projection(x, FRAME, refine_lm=False), x, FRAME)
        e_lm = cal.reprojection_errors(cal.estimate_projection(x, FRAME), x, FRAME)
        assert np.sum(e_lm**2) <= np.sum(e_dlt**2) + 1e-9


def test_coplanar_points_rejected():
    world = FRAME.copy()
    world[:, 2] = 8
    with pytest.raises(ValueError, match="coplanar"):
        cal.dlt(np.random.rand(8, 2) * 1000, world)


def test_too_few_or_mismatched_points(stereo):
    x = cal.project(stereo["P1"], FRAME)
    with pytest.raises(ValueError):
        cal.dlt(x[:5], FRAME[:5])
    with pytest.raises(ValueError):
        cal.dlt(x[:7], FRAME)


def test_points_in_front_of_camera(stereo):
    x = cal.project(stereo["P1"], FRAME)
    P = cal.estimate_projection(x, FRAME)
    depth = np.hstack([FRAME, np.ones((8, 1))]) @ P[2]
    assert np.all(depth > 0)


def test_leave_one_out(stereo):
    x = cal.project(stereo["P1"], FRAME)
    loo = cal.leave_one_out_errors(x, FRAME)
    assert loo.shape == (8,) and np.nanmax(loo) < 1e-5
    x_bad = x.copy()
    x_bad[[3, 4]] = x_bad[[4, 3]]  # swapped click order
    res = cal.calibrate_camera(x_bad, FRAME)
    assert res.errors.max() > 5
