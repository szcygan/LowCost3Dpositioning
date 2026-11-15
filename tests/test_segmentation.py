import numpy as np
import pytest

from lowcost3d.segmentation import (SegmentationError, close_mask, disk, region_grow,
                                    segment_markers)


def matlab_regiongrowing(I, x, y, reg_maxdist):
    """Line-by-line port of the original regiongrowing.m (0-based), for reference."""
    J = np.zeros(I.shape)
    reg_mean = I[x, y]
    reg_size = 1
    neg_list = []
    pixdist = 0
    neigb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while pixdist < reg_maxdist and reg_size < I.size:
        for dx, dy in neigb:
            xn, yn = x + dx, y + dy
            if 0 <= xn < I.shape[0] and 0 <= yn < I.shape[1] and J[xn, yn] == 0:
                neg_list.append([xn, yn, I[xn, yn]])
                J[xn, yn] = 1
        if not neg_list:
            J[x, y] = 2
            break
        dist = [abs(v[2] - reg_mean) for v in neg_list]
        index = int(np.argmin(dist))
        pixdist = dist[index]
        J[x, y] = 2
        reg_size += 1
        reg_mean = (reg_mean * reg_size + neg_list[index][2]) / (reg_size + 1)
        x, y = int(neg_list[index][0]), int(neg_list[index][1])
        neg_list[index] = neg_list[-1]
        neg_list.pop()
    return J > 1


def blob_image(shape=(120, 160), center=(60.3, 80.7), radius=12, fg=0.9, bg=0.3, noise=0.0,
               seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    img = np.full(shape, bg)
    img[(yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2] = fg
    img = img + rng.normal(0, noise, shape)
    return np.clip(np.rint(img * 255) / 255, 0, 1)


def test_legacy_mode_matches_matlab_port():
    rng = np.random.default_rng(3)
    # Unique grey levels so that the choice of the next pixel is unambiguous:
    # a noisy radial "blob" (smooth, large regions) and pure noise (small ones).
    yy, xx = np.mgrid[:16, :16]
    blob = np.hypot(yy - 7.3, xx - 8.1) + rng.normal(0, 0.8, (16, 16))
    images = [np.argsort(np.argsort(blob.ravel())).reshape(16, 16) / 255.0,
              rng.permutation(256).reshape(16, 16) / 255.0]
    sizes = []
    for img in images:
        for seed in [(3, 4), (7, 8), (15, 0), (0, 15)]:
            for t in (0.05, 0.1, 0.2, 0.4):
                ref = matlab_regiongrowing(img, *seed, t)
                ours = region_grow(img, seed, t, legacy_mean=True)
                assert np.array_equal(ref, ours), (seed, t)
                sizes.append(ref.sum())
    assert max(sizes) > 50  # the comparison covers large regions too


def test_region_on_blob_and_centroid():
    img = blob_image(noise=0.02)
    mask = region_grow(img, (60, 80), 0.2)
    yy, xx = np.nonzero(mask)
    assert abs(mask.sum() - np.pi * 12**2) < 40
    seg = segment_markers(img, [[78, 63]], max_dist=0.2, disk_radius=2)  # off-centre seed
    np.testing.assert_allclose(seg.centroids[0], [80.7, 60.3], atol=0.15)


def test_centroid_independent_of_seed_position():
    img = blob_image(noise=0.02)
    seeds = [[80, 60], [72, 57], [88, 64], [79, 49]]
    seg = segment_markers(img, seeds, max_dist=0.2, disk_radius=2)
    assert np.ptp(seg.centroids, axis=0).max() < 1e-9


def test_max_area_guard():
    img = blob_image()
    with pytest.raises(SegmentationError):
        region_grow(img, (5, 5), 0.2, max_area=500)  # seed on the background


def test_whole_image_region_terminates():
    img = np.full((20, 30), 0.5)
    assert region_grow(img, (10, 10), 0.1).all()


def test_seed_outside_image():
    with pytest.raises(ValueError):
        region_grow(np.zeros((10, 10)), (10, 3), 0.1)


def test_close_mask_fills_specular_hole():
    mask = disk(10)
    mask = np.pad(mask, 5)
    holed = mask.copy()
    holed[13:17, 13:17] = False
    assert np.array_equal(close_mask(holed, 0, fill_holes=True), mask)
    closed = close_mask(holed, 3)
    assert closed[15, 15]


def test_close_mask_at_image_border():
    mask = np.zeros((30, 30), bool)
    mask[:10, :10] = True
    np.testing.assert_array_equal(close_mask(mask, 4), mask)
