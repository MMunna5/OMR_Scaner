"""Small compatibility fix for the two-block portrait OMR layout.

The scanner consumes 30 questions as 15 rows in the left block followed by
15 rows in the right block. The current scan() routine temporarily groups
coordinates by visual row; these adapters provide that exact intermediate
ordering so its existing regrouping restores the correct question sequence.
"""
import omr


def _row_interleave(points):
    if len(points) != 120:
        return points
    left = points[:60]
    right = points[60:]
    arranged = []
    for row in range(15):
        arranged.extend(left[row * 4:(row + 1) * 4])
        arranged.extend(right[row * 4:(row + 1) * 4])
    return arranged


_original_portrait_grid = omr._portrait_grid
_original_portrait_points = omr._portrait_points


def _portrait_grid_in_scan_order(image):
    return _row_interleave(_original_portrait_grid(image))


def _portrait_points_in_scan_order(width=1000, height=1400):
    return _row_interleave(_original_portrait_points(width, height))


omr._portrait_grid = _portrait_grid_in_scan_order
omr._portrait_points = _portrait_points_in_scan_order
