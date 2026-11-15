"""End-to-end checks on the example images shipped with the paper."""

import json
import warnings

import numpy as np
import pytest

from lowcost3d import io
from lowcost3d import pipeline as pl
from lowcost3d.cli import main


@pytest.fixture(scope="module")
def calib(data_dir=None):
    from conftest import DATA as d
    return pl.calibrate(io.load_gray(d / "calibrationLeft.jpg"),
                        io.load_gray(d / "calibrationRight.jpg"),
                        io.load_points(d / "markersFrame.xlsx"),
                        io.load_points(d / "seeds/calibration_left.csv"),
                        io.load_points(d / "seeds/calibration_right.csv"))


def _pair(d, name):
    return (io.load_gray(d / f"{name}Left.jpg"), io.load_gray(d / f"{name}Right.jpg"),
            io.load_points(d / f"seeds/{name}_left.csv"),
            io.load_points(d / f"seeds/{name}_right.csv"))


def test_load_gray_matches_matlab_rgb2gray(data_dir):
    img = io.load_gray(data_dir / "calibrationLeft.jpg")
    assert img.shape == (1080, 1920)
    assert np.allclose(img * 255, np.rint(img * 255))


def test_marker_table_formats_agree(data_dir):
    np.testing.assert_allclose(io.load_points(data_dir / "markersFrame.xlsx"),
                               io.load_points(data_dir / "markersFrame.csv"))


def test_calibration_quality(calib):
    assert calib["mean_reprojection_error_px"] < 1.0
    for cam in calib["cameras"].values():
        assert cam["reprojection_errors_px"].max() < 1.5
        assert abs(cam["skew"]) < 20
        fx, fy = cam["focal_px"]
        assert abs(fx / fy - 1) < 0.02


def test_validation_board_accuracy(calib, data_dir):
    board = io.load_points(data_dir / "validation_board.csv")
    res = pl.locate(calib, *_pair(data_dir, "validation"), model=board)
    assert res["rigid_fit"]["rms_mm"] < 1.0
    D = res["pairwise_distances_mm"]
    for (i, j), nominal in {(0, 1): 145, (3, 4): 145, (0, 3): 90, (1, 4): 90}.items():
        assert abs(D[i, j] - nominal) < 1.5
    assert res["coplanarity_rms_mm"] < 0.5
    assert res["reprojection_errors_px"].max() < 2


def test_result_does_not_depend_on_click_position(calib, data_dir):
    L, R, sl, sr = _pair(data_dir, "validation")
    ref = pl.locate(calib, L, R, sl, sr)["points_mm"]
    rng = np.random.default_rng(0)
    moved = pl.locate(calib, L, R, sl + rng.uniform(-4, 4, sl.shape),
                      sr + rng.uniform(-4, 4, sr.shape))["points_mm"]
    np.testing.assert_allclose(moved, ref, atol=0.05)


def test_probe_markers_coplanar(calib, data_dir):
    res = pl.locate(calib, *_pair(data_dir, "probe"))
    assert res["points_mm"].shape == (4, 3)
    assert res["coplanarity_rms_mm"] < 0.5


def test_wrong_click_order_warns(calib, data_dir):
    L, R, sl, sr = _pair(data_dir, "validation")
    with pytest.warns(UserWarning, match="same order"):
        pl.locate(calib, L, R, sl, sr[[1, 0, 2, 3, 4]])


def test_unequal_marker_counts_rejected(calib, data_dir):
    L, R, sl, sr = _pair(data_dir, "validation")
    with pytest.raises(ValueError, match="left view"):
        pl.locate(calib, L, R, sl, sr[:4])


def test_image_size_checked(calib, data_dir):
    L, R, sl, sr = _pair(data_dir, "validation")
    with pytest.raises(ValueError, match="image size"):
        pl.locate(calib, L[:-10], R[:-10], sl, sr)


def test_cli_roundtrip(tmp_path, data_dir, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    d = data_dir
    cal_json = tmp_path / "cal.json"
    main(["calibrate", "--left", str(d / "calibrationLeft.jpg"),
          "--right", str(d / "calibrationRight.jpg"), "--markers", str(d / "markersFrame.xlsx"),
          "--seeds-left", str(d / "seeds/calibration_left.csv"),
          "--seeds-right", str(d / "seeds/calibration_right.csv"),
          "--out", str(cal_json), "--no-show"])
    out = tmp_path / "val.json"
    main(["locate", "--calibration", str(cal_json),
          "--left", str(d / "validationLeft.jpg"), "--right", str(d / "validationRight.jpg"),
          "--seeds-left", str(d / "seeds/validation_left.csv"),
          "--seeds-right", str(d / "seeds/validation_right.csv"),
          "--model", str(d / "validation_board.csv"), "--out", str(out), "--no-show"])
    res = json.loads(out.read_text())
    assert res["rigid_fit"]["rms_mm"] < 1.0
    assert len(res["points_mm"]) == 5
