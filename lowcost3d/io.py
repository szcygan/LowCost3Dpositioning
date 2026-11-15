"""Loading images, marker tables, point lists and result files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

# Luma weights used by MATLAB's rgb2gray (ITU-R BT.601).
_RGB2GRAY = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103])


def load_gray(path: str | Path) -> np.ndarray:
    """Read an image and return it as float grayscale in [0, 1].

    Matches MATLAB's ``im2double(rgb2gray(imread(path)))`` for 8-bit RGB
    images: the BT.601 weighted sum is rounded back to uint8 before scaling.
    """
    img = np.asarray(Image.open(path))
    if img.ndim == 3:
        rgb = img[..., :3].astype(float)
        if img.dtype == np.uint8:
            gray = np.clip(np.rint(rgb @ _RGB2GRAY), 0, 255).astype(np.uint8)
        else:
            gray = rgb @ _RGB2GRAY
    else:
        gray = img
    if gray.dtype == np.uint8:
        return gray.astype(float) / 255.0
    if gray.dtype == np.uint16:
        return gray.astype(float) / 65535.0
    gray = gray.astype(float)
    if gray.max() > 1:
        raise ValueError(f"cannot infer intensity range of {path}")
    return gray


def load_points(path: str | Path) -> np.ndarray:
    """Read an (N, 2) or (N, 3) table of coordinates.

    Accepts ``.xlsx``/``.xls`` (first sheet) and ``.csv``/``.txt``. A header
    row with non-numeric labels (e.g. ``X, Y, Z``) is skipped automatically.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        import pandas as pd

        df = pd.read_excel(path, header=None)
        data = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        arr = data.dropna(axis=1, how="all").to_numpy(dtype=float)
    else:
        arr = np.genfromtxt(path, delimiter=",", dtype=float, comments="#")
        arr = np.atleast_2d(arr)
        arr = arr[~np.all(np.isnan(arr), axis=1)]
    if np.isnan(arr).any():
        raise ValueError(f"{path}: table contains non-numeric entries")
    return arr


def save_points(path: str | Path, points: np.ndarray, header: str = "x,y") -> None:
    """Write an (N, k) array as CSV with a header line."""
    np.savetxt(path, np.asarray(points), delimiter=",", header=header, comments="",
               fmt="%.4f")


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(path: str | Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, cls=_Encoder, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
