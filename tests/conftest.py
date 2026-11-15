from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parent.parent / "data"

# Calibration frame (markersFrame.xlsx), mm
FRAME = np.array([
    [246.1, 16.1, 8], [246.1, 311.1, 8], [51.1, 281.1, 8], [71.1, 166.1, 8],
    [51.1, 51.1, 8], [8, 21.1, 235], [8, 166.1, 235], [8, 311.1, 235],
])


def look_at(center, target, f=1450.0, pp=(960.0, 540.0), skew=0.0):
    """Synthetic pinhole camera looking from ``center`` at ``target``."""
    center = np.asarray(center, float)
    z = np.asarray(target, float) - center
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, -1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    K = np.array([[f, skew, pp[0]], [0, f, pp[1]], [0, 0, 1]])
    P = K @ np.hstack([R, -R @ center[:, None]])
    return P, K, R


@pytest.fixture
def stereo():
    target = [130, 160, 110]
    P1, K1, R1 = look_at([740, -90, 300], target)
    P2, K2, R2 = look_at([730, 430, 270], target, f=1430, pp=(950, 560))
    return {"P1": P1, "P2": P2, "K1": K1, "R1": R1, "C1": np.array([740, -90, 300.]),
            "K2": K2, "C2": np.array([730, 430, 270.])}


@pytest.fixture
def data_dir():
    return DATA
