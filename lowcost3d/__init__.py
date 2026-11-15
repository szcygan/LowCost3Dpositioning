"""Low-cost stereo vision positioning of an ultrasonic probe.

Python implementation of the method described in
S. Cygan, M. Urban, A. Czyżyk, J. Żmigrodzki, "Low-cost 3D vision-based
triangulation system for ultrasonic probe positioning",
Pol J Med Phys Eng 2024;30(4):249-257, doi:10.2478/pjmpe-2024-0029.
"""

from .calibration import (calibrate_camera, decompose, estimate_projection,
                          leave_one_out_errors, project, reprojection_errors)
from .geometry import fit_plane, fit_rigid, pairwise_distances, triangulate
from .io import load_gray, load_points
from .pipeline import (CALIBRATION_DEFAULTS, LOCATE_DEFAULTS, SegmentationParams,
                       calibrate, locate, tune)
from .segmentation import SegmentationError, region_grow, segment_markers

__version__ = "1.0.0"
