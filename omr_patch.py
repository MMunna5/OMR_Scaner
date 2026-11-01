"""Compatibility fixes for the two-block portrait OMR layout and multi-answer keys."""
import omr


class AnswerChoices(list):
    """A list of correct options that also matches the scanner's legacy int comparison."""
    def __eq__(self, other):
        if type(other) is int:
            return other in self
        return super().__eq__(other)


def normalize_key(key):
    return [AnswerChoices(answer) if isinstance(answer, list) else answer for answer in key]


def _row_interleave(points):
    if len(points) != 120:
        return points
    left, right = points[:60], points[60:]
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
