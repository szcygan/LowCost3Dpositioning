"""Marker segmentation: seeded region growing followed by morphological closing.

Python port of ``regiongrowing.m`` (D. Kroon, University of Twente) as used in
the original MATLAB supplement, with the following corrections:

* the running region mean is updated correctly (the original weighted the old
  mean by ``n + 1`` instead of ``n``; reproducible with ``legacy_mean=True``),
* the frontier is kept in per-intensity buckets, so growing a region costs
  O(N) instead of O(N^2) and large regions no longer take minutes,
* growth can be capped with ``max_area`` so a seed placed outside a marker (or
  a threshold that is too permissive) fails loudly instead of returning the
  centroid of half of the image,
* an empty frontier (region filled the whole connected area) ends the loop
  instead of raising an indexing error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# Neighbour offsets (4-connectivity), same order as in the MATLAB code.
_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class SegmentationError(RuntimeError):
    """Raised when a marker cannot be segmented reliably."""


def _quantize(image: np.ndarray, levels: int) -> np.ndarray:
    """Map a float image in [0, 1] (or an integer image) to integer levels."""
    if np.issubdtype(image.dtype, np.integer):
        q = image.astype(np.int64)
        if q.min() < 0 or q.max() >= levels:
            raise ValueError(f"integer image values must lie in [0, {levels - 1}]")
        return q
    img = np.asarray(image, dtype=float)
    if img.min() < 0 or img.max() > 1:
        raise ValueError("float images must be scaled to [0, 1]")
    return np.rint(img * (levels - 1)).astype(np.int64)


def region_grow(
    image: np.ndarray,
    seed: tuple[int, int],
    max_dist: float = 0.2,
    *,
    max_area: int | None = None,
    levels: int = 256,
    legacy_mean: bool = False,
) -> np.ndarray:
    """Grow a region from ``seed`` (row, col) in a grayscale image.

    At every step the frontier pixel whose intensity is closest to the current
    region mean is added. Growth stops when that distance reaches ``max_dist``
    (in the [0, 1] intensity scale), when the frontier is empty, or raises
    :class:`SegmentationError` when the region exceeds ``max_area`` pixels.

    Parameters
    ----------
    image
        2-D grayscale image, float in [0, 1] or integer in [0, levels-1].
    seed
        (row, col) of the seed pixel.
    max_dist
        Maximum intensity distance between the region mean and a new pixel.
    max_area
        Optional upper bound on the region size in pixels.
    levels
        Number of grey levels used for the bucket queue. 256 is exact for
        8-bit images (all images produced by ``im2double(rgb2gray(uint8))``).
    legacy_mean
        Reproduce the mean-update formula of the original MATLAB code.

    Returns
    -------
    numpy.ndarray of bool
        Mask of the segmented region.
    """
    if image.ndim != 2:
        raise ValueError("image must be 2-D (grayscale)")
    q = _quantize(image, levels)
    scale = levels - 1
    h, w = q.shape
    r, c = int(seed[0]), int(seed[1])
    if not (0 <= r < h and 0 <= c < w):
        raise ValueError(f"seed {seed} lies outside the image of shape {q.shape}")

    # 0 = untouched, 1 = in frontier, 2 = in region
    state = np.zeros((h, w), dtype=np.uint8)
    buckets: list[list[int]] = [[] for _ in range(levels)]
    counts = np.zeros(levels, dtype=np.int64)
    n_frontier = 0

    reg_mean = q[r, c] / scale
    reg_size = 1
    flat = q.ravel()

    while True:
        state[r, c] = 2
        # add neighbours of the newest region pixel to the frontier
        for dr, dc in _NEIGHBOURS:
            rn, cn = r + dr, c + dc
            if 0 <= rn < h and 0 <= cn < w and state[rn, cn] == 0:
                state[rn, cn] = 1
                lvl = q[rn, cn]
                buckets[lvl].append(rn * w + cn)
                counts[lvl] += 1
                n_frontier += 1
        if n_frontier == 0:
            break

        # frontier pixel closest to the current mean (search outwards)
        centre = int(round(reg_mean * scale))
        best = None
        for off in range(levels):
            lo, hi = centre - off, centre + off
            cand = []
            if 0 <= lo < levels and counts[lo]:
                cand.append(lo)
            if hi != lo and 0 <= hi < levels and counts[hi]:
                cand.append(hi)
            if cand:
                best = min(cand, key=lambda v: abs(v / scale - reg_mean))
                break
        if best is None:  # pragma: no cover - n_frontier > 0 guarantees a hit
            break
        pixdist = abs(best / scale - reg_mean)
        if pixdist >= max_dist:
            break

        idx = buckets[best].pop()
        counts[best] -= 1
        n_frontier -= 1
        value = flat[idx] / scale
        if legacy_mean:
            reg_size += 1
            reg_mean = (reg_mean * reg_size + value) / (reg_size + 1)
        else:
            reg_mean = (reg_mean * reg_size + value) / (reg_size + 1)
            reg_size += 1
        r, c = divmod(idx, w)
        if max_area is not None and reg_size > max_area:
            raise SegmentationError(
                f"region grown from seed {seed} exceeded max_area={max_area} px; "
                "the seed is probably outside the marker or max_dist is too large"
            )

    return state == 2


def disk(radius: int) -> np.ndarray:
    """Flat disk structuring element (Euclidean, like ``strel('disk', r, 0)``)."""
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return xx**2 + yy**2 <= radius**2


def close_mask(mask: np.ndarray, radius: int, fill_holes: bool = False) -> np.ndarray:
    """Morphological closing with a disk (optionally followed by hole filling).

    Only the bounding box of the mask is processed, which makes this fast on
    full-resolution images. The image border does not erode the result.
    """
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    out = np.zeros_like(mask, dtype=bool)
    if rows.size == 0:
        return out
    pad = max(radius, 0) + 1
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    crop = np.pad(mask[r0:r1, c0:c1].astype(bool), pad, mode="constant")
    if radius > 0:
        se = disk(radius)
        crop = ndimage.binary_erosion(ndimage.binary_dilation(crop, structure=se),
                                      structure=se, border_value=1)
    if fill_holes:
        crop = ndimage.binary_fill_holes(crop)
    # a closing never extends beyond the bounding box of its input
    out[r0:r1, c0:c1] = crop[pad:-pad, pad:-pad]
    return out


@dataclass
class MarkerSegmentation:
    """Result of segmenting a set of markers in one image."""

    centroids: np.ndarray  # (N, 2) as (x, y) in pixels, 0-based
    masks: list[np.ndarray] = field(repr=False)
    areas: np.ndarray

    @property
    def combined_mask(self) -> np.ndarray:
        out = np.zeros_like(self.masks[0], dtype=bool)
        for m in self.masks:
            out |= m
        return out


def segment_markers(
    image: np.ndarray,
    seeds_xy: np.ndarray,
    *,
    max_dist: float,
    disk_radius: int,
    fill_holes: bool = True,
    max_area: int | None = 20000,
    legacy_mean: bool = False,
) -> MarkerSegmentation:
    """Segment every marker from its seed and return the mask centroids.

    ``seeds_xy`` holds (x, y) image coordinates (column, row), 0-based, as
    returned by the point picker. Seeds are rounded to the nearest pixel.
    """
    seeds_xy = np.asarray(seeds_xy, dtype=float).reshape(-1, 2)
    centroids, masks, areas = [], [], []
    for k, (x, y) in enumerate(seeds_xy):
        seed_rc = (int(round(y)), int(round(x)))
        mask = region_grow(
            image, seed_rc, max_dist, max_area=max_area, legacy_mean=legacy_mean
        )
        mask = close_mask(mask, disk_radius, fill_holes=fill_holes)
        rows, cols = np.nonzero(mask)
        if rows.size == 0:  # pragma: no cover - region always contains the seed
            raise SegmentationError(f"marker {k + 1}: empty segmentation")
        centroids.append((cols.mean(), rows.mean()))
        masks.append(mask)
        areas.append(rows.size)
    return MarkerSegmentation(np.array(centroids), masks, np.array(areas))
