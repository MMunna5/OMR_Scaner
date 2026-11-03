"""OMR engine for the Shaheed Noor Memorial School & College answer sheet.
Pipeline: find 4 printed anchors (top black squares / bottom tick bars) -> rectify
-> snap to detected bubbles -> read marks. Layout constants were calibrated on the
real sheet (frame = 1000 px per anchor box, anchors start at 200,200)."""
import base64
import cv2
import numpy as np

L = ['ক', 'খ', 'গ', 'ঘ']
A, S, FW, FH = 200, 1000, 1500, 1400
U = [[.5551, .6024, .6522, .7010], [.8407, .8892, .9387, .9884]]   # answer columns (block 1, block 2)
V0, DV = .1553, .03316                                               # answer rows
RU = [-.005, .028, .061, .094, .126, .160]                           # roll-number columns
RV0, RDV = .143, .0258                                               # roll rows 0..9


class ScanError(Exception):
    pass


def _fit_line(pts, min_n):
    """RANSAC-ish: best near-horizontal line through pts; return (leftmost, rightmost) inliers."""
    p = np.array(pts, dtype=np.float32)
    if len(p) < min_n:
        return None
    best = None
    idx = range(len(p))
    for i in idx:
        d = p - p[i]
        dx, dy = d[:, 0], d[:, 1]
        for j in np.where((dx > 150) & (np.abs(dy) < 0.3 * np.abs(dx)))[0]:
            n = np.hypot(dx[j], dy[j])
            dist = np.abs(dx[j] * dy - dy[j] * dx) / n
            inl = np.where(dist < 8)[0]
            if best is None or len(inl) > len(best):
                best = inl
    if best is None or len(best) < min_n:
        return None
    q = p[best]
    return q[np.argmin(q[:, 0])], q[np.argmax(q[:, 0])]


def find_anchors(im):
    b, g, r = [im[:, :, i].astype(np.int16) for i in range(3)]
    for t1, t2 in ((40, 30), (28, 20), (18, 12)):
        m = ((r - g > t1) & (r - b > t2)).astype(np.uint8)
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        sq, bar = [], []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if 18 <= w <= 55 and 12 <= h <= 40 and a > .55 * w * h and .9 <= w / h <= 2.1:
                sq.append(cen[i])
            elif 2 <= w <= 16 and 12 <= h <= 60 and h >= 1.5 * w and a > .3 * w * h:
                bar.append(cen[i])
        top, bot = _fit_line(sq, 3), _fit_line(bar, 25)
        if top is not None and bot is not None:
            return top, bot
    return None


def _rectify_page(img):
    """Rectify a plain portrait answer sheet when calibrated anchors are absent."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best = None
    best_area = 0
    for threshold in (140, 120, 100):
        mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < img.shape[0] * img.shape[1] * .20 or area <= best_area:
                continue
            peri = cv2.arcLength(contour, True)
            poly = cv2.approxPolyDP(contour, .03 * peri, True)
            if len(poly) == 4 and cv2.isContourConvex(poly):
                best, best_area = poly[:, 0, :].astype(np.float32), area
            else:
                box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
                if area / cv2.contourArea(cv2.convexHull(box)) > .65:
                    best, best_area = box, area
        if best is not None:
            break
    if best is None:
        h, w = img.shape[:2]
        best = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    s = best.sum(1)
    d = np.diff(best, axis=1).ravel()
    src = np.float32([best[np.argmin(s)], best[np.argmin(d)],
                      best[np.argmax(s)], best[np.argmax(d)]])
    width = max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3]))
    height = max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1]))
    target_width = 1000
    target_height = max(1000, min(1800, round(target_width * height / max(width, 1))))
    dst = np.float32([[0, 0], [target_width, 0], [target_width, target_height],
                      [0, target_height]])
    return cv2.warpPerspective(
        img, cv2.getPerspectiveTransform(src, dst), (target_width, target_height),
        flags=cv2.INTER_CUBIC)


def rectify(img):
    """Find the calibrated anchor pair in any quarter-turn camera orientation."""
    candidates = [
        img,
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img, cv2.ROTATE_180),
        cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    detected = None
    for candidate in candidates:
        scale = 1400.0 / candidate.shape[1]
        small = cv2.resize(candidate, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        anchors = find_anchors(small)
        if anchors is None:
            continue
        (tl, tr), (bl, br) = anchors
        # The printed square row must be above the bottom alignment bars.
        if tl[1] > bl[1]:
            continue
        if (tr[0] - tl[0]) / scale < 500:
            continue
        detected = (candidate, scale, anchors)
        break

    if detected is None:
        # This school's 50-question sheet must be aligned from its printed
        # top and bottom registration marks. Never guess a different layout.
        raise ScanError(
            'Could not align the OMR sheet. Upload a clear photo showing the '
            'full page, including the top red alignment boxes and bottom red bars.'
        )

    candidate, scale, ((tl, tr), (bl, br)) = detected
    src = np.float32([tl, tr, br, bl]) / scale
    dst = np.float32([(A, A), (A + S, A), (A + S, A + S), (A, A + S)])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(candidate, M, (FW, FH), flags=cv2.INTER_CUBIC), 'standard'

def _fit_H(points, circles):
    """Homography expected-grid -> detected circles (corrects small anchor errors). Identity if unreliable."""
    pts = np.array(points, np.float32)
    H_tot = np.eye(3)
    if len(circles) < 12:
        return H_tot
    cc = circles[:, :2]
    for tol in (45, 22):
        cur = cv2.perspectiveTransform(pts[None], H_tot)[0]
        d = np.linalg.norm(cur[:, None, :] - cc[None, :, :], axis=2)
        j = d.argmin(1)
        ok = d[np.arange(len(pts)), j] < tol
        if ok.sum() < 30:
            break
        H, _ = cv2.findHomography(cur[ok], cc[j[ok]], cv2.RANSAC, 4.0)
        if H is None:
            break
        H_new = H @ H_tot
        moved = np.linalg.norm(cv2.perspectiveTransform(pts[None], H_new)[0] - pts, axis=1).max()
        if moved > 60:
            break
        H_tot = H_new
    return H_tot


def _snap(points, circles, H):
    pts = cv2.perspectiveTransform(np.array(points, np.float32)[None], H)[0]
    snapped = pts.copy()
    if len(circles):
        cc = circles[:, :2]
        d = np.linalg.norm(pts[:, None, :] - cc[None, :, :], axis=2)
        j = d.argmin(1)
        ok = d[np.arange(len(pts)), j] < 10
        snapped[ok] = cc[j[ok]]
    return pts, snapped


def _patch(R, x, y, r=10):
    x, y = int(round(x)), int(round(y))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    m = (xx * xx + yy * yy) <= r * r
    p = R[y - r:y + r + 1, x - r:x + r + 1]
    return p[m] if p.shape == m.shape else np.full(int(m.sum()), 255, np.uint8)


def _inner(R, x, y):
    return float(_patch(R, x, y).mean())


def _is_marked(R, pts, base):
    """Marked when at least 15% of the inner bubble or box is filled."""
    return any(_mark_fraction(R, x, y, base) >= .15 for x, y in pts)


def _mark_fraction(R, x, y, base):
    """Measure ink in the bubble and its four-corner box, excluding the border."""
    x, y = int(round(x)), int(round(y))
    patch = R[y - 11:y + 12, x - 11:x + 12]
    if patch.shape != (23, 23):
        return 0.0
    dark = patch < min(150, base * .72)
    yy, xx = np.ogrid[-11:12, -11:12]
    circle = (xx * xx + yy * yy) <= 9 * 9
    circle_fraction = float(dark[circle].mean())
    box = patch[4:19, 4:19]
    box_fraction = float((box < min(150, base * .72)).mean())
    return max(circle_fraction, box_fraction)


def _portrait_marked(R, x, y, base):
    """Detect ink inside a portrait-layout bubble, independently of its printed outline."""
    patch = R[y - 9:y + 10, x - 9:x + 10].astype(np.float32)
    if patch.shape != (19, 19):
        return False
    yy, xx = np.ogrid[-9:10, -9:10]
    radius = np.sqrt(xx * xx + yy * yy)
    # Ignore the printed bubble border: only inspect its central fill area.
    core = patch[radius <= 5]
    ring = patch[(radius >= 7) & (radius <= 9)]
    if not core.size or not ring.size:
        return False
    background = float(np.median(ring))
    threshold = min(145.0, max(55.0, float(base) * .72))
    dark_fraction = float((core < threshold).mean())
    core_mean = float(core.mean())
    # A real mark darkens the centre; an unfilled printed circle only darkens
    # the outer ring. Relative contrast also handles tinted/uneven paper.
    contrast = background - core_mean
    return dark_fraction >= .22 and contrast >= max(12.0, background * .16)


def _portrait_points(width=1000, height=1400):
    """Answer positions for the common 30-question portrait sheet: two blocks of 15."""
    xs = ([.19, .232, .276, .318], [.442, .480, .520, .560])
    return [(x * width, (.31 + .04 * (q % 15)) * height)
            for q in range(30) for x in xs[q // 15]]


def _fit_axis(values, count, start, end, spacing_min, spacing_max):
    values = np.asarray(values, dtype=np.float32)
    best = None
    for spacing in np.linspace(spacing_min, spacing_max, 41):
        for origin in np.arange(start, end - spacing * (count - 1), 2):
            expected = origin + spacing * np.arange(count)
            distance = np.abs(values[:, None] - expected[None, :]).min(1)
            score = int((distance < spacing * .28).sum())
            error = float(distance[distance < spacing * .28].mean()) if score else 999
            candidate = (score, -error, expected)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None or best[0] < count * .55:
        raise ScanError('Could not fit the answer grid.')
    return best[2].tolist()


def _cluster_centres(values, count, tolerance=18):
    values = sorted(float(v) for v in values)
    groups = []
    for value in values:
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    centres = [sum(group) / len(group) for group in groups]
    if len(centres) != count:
        data = np.float32(values).reshape(-1, 1)
        _, _, centers = cv2.kmeans(
            data, count, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, .5),
            5, cv2.KMEANS_PP_CENTERS)
        centres = sorted(float(x[0]) for x in centers)
    return centres


def _portrait_grid(R):
    gray = cv2.medianBlur(R, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=18, param1=70, param2=16,
        minRadius=7, maxRadius=20)
    if circles is None or len(circles[0]) < 40:
        raise ScanError('Could not locate the answer grid. Use a clearer, fully visible photo.')
    c = circles[0]
    height, width = R.shape
    c = c[(c[:, 0] > .15 * width) & (c[:, 0] < .62 * width) &
          (c[:, 1] > .25 * height) & (c[:, 1] < .95 * height)]
    if len(c) < 40:
        raise ScanError('Could not locate enough answer bubbles. Capture the full sheet without obstruction.')
    left = c[(c[:, 0] > .18 * width) & (c[:, 0] < .40 * width)]
    right = c[(c[:, 0] > .42 * width) & (c[:, 0] < .62 * width)]
    if len(left) < 20 or len(right) < 20:
        raise ScanError('Could not separate the two answer blocks.')
    left_x = _fit_axis(left[:, 0], 4, .18 * width, .40 * width,
                       .035 * width, .05 * width)
    right_x = _fit_axis(right[:, 0], 4, .42 * width, .62 * width,
                        .03 * width, .05 * width)
    ys = _fit_axis(c[:, 1], 15, .28 * height, .95 * height,
                   .035 * height, .06 * height)
    points = []
    for columns in (left_x, right_x):
        for row in ys:
            for col in columns:
                d = np.linalg.norm(c[:, :2] - np.array([col, row]), axis=1)
                points.append(c[d.argmin(), :2] if d.min() < 30 else (col, row))
    return points


def _standard_grid(R):
    """Fit the two 25-question answer blocks from the detected bubble circles."""
    gray = cv2.medianBlur(R, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=16, param1=70, param2=18,
        minRadius=8, maxRadius=20)
    if circles is None:
        raise ScanError('Could not locate the answer grid.')
    height, width = R.shape
    c = circles[0]
    # The leftmost option column can shift left on angled photos; the
    # rightmost column can shift beyond 83% after perspective correction.
    # Keep the answer-grid search separate from the roll-number area.
    c = c[(c[:, 0] > .43 * width) & (c[:, 0] < .89 * width) &
          (c[:, 1] > .24 * height) & (c[:, 1] < .86 * height)]
    if len(c) < 70:
        raise ScanError('Could not locate enough answer bubbles. Retake the photo with the full answer grid visible.')
    left = c[(c[:, 0] > .43 * width) & (c[:, 0] < .65 * width)]
    right = c[(c[:, 0] > .66 * width) & (c[:, 0] < .89 * width)]
    if len(left) < 25 or len(right) < 25:
        raise ScanError('Could not separate the two answer option blocks. Check that the whole sheet is visible.')
    left_x = _fit_axis(left[:, 0], 4, .43 * width, .65 * width,
                       .025 * width, .06 * width)
    right_x = _fit_axis(right[:, 0], 4, .66 * width, .89 * width,
                         .025 * width, .06 * width)
    ys = _fit_axis(c[:, 1], 25, .24 * height, .86 * height,
                   .016 * height, .04 * height)
    points = []
    for columns in (left_x, right_x):
        for row in ys:
            for col in columns:
                d = np.linalg.norm(c[:, :2] - np.array([col, row]), axis=1)
                points.append(c[d.argmin(), :2] if d.min() < 30 else (col, row))
    return points


def _bilinear_box(tl, tr, br, bl, u, v):
    """Map normalized (u, v) coordinates through a four-corner quadrilateral."""
    top = np.asarray(tl, np.float32) * (1.0 - u) + np.asarray(tr, np.float32) * u
    bottom = np.asarray(bl, np.float32) * (1.0 - u) + np.asarray(br, np.float32) * u
    return top * (1.0 - v) + bottom * v


def _four_corner_answer_grid():
    """Build Q1-25 and Q26-50 independently from each answer box's four corners."""
    # Corners enclose only the four answer-option columns, not the question-number column.
    boxes = (
        ((730, 330), (925, 330), (925, 1170), (730, 1170)),
        ((1015, 330), (1215, 330), (1215, 1170), (1015, 1170)),
    )
    option_u = (.128, .369, .626, .877)
    row_v = [.030 + i * (.947 / 24.0) for i in range(25)]
    points = []
    for corners in boxes:
        for v in row_v:
            for u in option_u:
                points.append(_bilinear_box(*corners, u, v))
    return np.asarray(points, np.float32)


def _four_corner_roll_grid():
    """Build the six-digit, ten-row roll grid from its own four-corner box."""
    corners = ((180, 330), (375, 330), (375, 590), (180, 590))
    col_u = (.077, .247, .416, .585, .753, .923)
    row_v = [.050 + i * (.896 / 9.0) for i in range(10)]
    return np.asarray([
        _bilinear_box(*corners, u, v)
        for u in col_u for v in row_v
    ], np.float32).reshape(6, 10, 2)


def _box_mark_fraction(R, x, y, base, half_w, half_h):
    """Ink ratio inside a rectangular bubble cell, excluding its printed border."""
    x, y = int(round(x)), int(round(y))
    patch = R[y - half_h:y + half_h + 1, x - half_w:x + half_w + 1]
    if patch.shape != (2 * half_h + 1, 2 * half_w + 1):
        return 0.0
    threshold = min(150.0, max(45.0, float(base) * .65))
    return float((patch < threshold).mean())


def _scan_box_answers(R, points, n, key):
    """Read the two 25-row answer boxes using fixed four-corner geometry and 15% ink."""
    pts = np.asarray(points, np.float32).reshape(50, 4, 2)
    inner = np.asarray([[_inner(R, *pts[q, k]) for k in range(4)]
                        for q in range(50)])
    answers, status, selected = [], [], []
    correct = wrong = blank = double = 0
    for q in range(n):
        # Each answer option is a full rectangular cell. Derive its boundaries
        # from neighbouring bubble centres, not a small circle-centred patch.
        block_start = (q // 25) * 25
        for_row = []
        for k in range(4):
            if q > block_start:
                for_row.append(abs(float(pts[q, k, 1] - pts[q - 1, k, 1])))
            if q < block_start + 24:
                for_row.append(abs(float(pts[q + 1, k, 1] - pts[q, k, 1])))
        half_h = max(1, int(min(for_row) * .46)) if for_row else 1

        # Keep a narrow safety gap between adjacent option cells.
        marked = []
        base = float(np.median(inner[max(block_start, q - 1):min(block_start + 25, q + 2)]))
        for k in range(4):
            neighbours = []
            if k > 0:
                neighbours.append(abs(float(pts[q, k, 0] - pts[q, k - 1, 0])))
            if k < 3:
                neighbours.append(abs(float(pts[q, k + 1, 0] - pts[q, k, 0])))
            half_w = max(1, int(min(neighbours) * .46))
            if _box_mark_fraction(R, *pts[q, k], base, half_w, half_h) >= .15:
                marked.append(k)
        selected.append(marked)
        expected = key[q] if isinstance(key[q], (list, tuple, set)) else [key[q]]
        if not marked:
            answers.append('-'); status.append('b'); blank += 1; wrong += 1
        elif len(marked) > 1:
            answers.append('*'); status.append('d'); double += 1; wrong += 1
        elif marked[0] in expected:
            answers.append(L[marked[0]]); status.append('c'); correct += 1
        else:
            answers.append(L[marked[0]]); status.append('w'); wrong += 1
    return pts, inner, answers, status, correct, wrong, blank, double, selected


def _read_box_roll(R, points):
    """Read each of six roll digits from its own ten-row rectangular bubble column."""
    roll_points = np.asarray(points, np.float32).reshape(6, 10, 2)
    digits, selected = [], []
    column_gaps = np.abs(np.diff(roll_points[:, 0, 0]))
    row_gaps = np.abs(np.diff(roll_points[0, :, 1]))
    half_w = max(1, int(float(column_gaps.min()) * .46))
    half_h = max(1, int(float(row_gaps.min()) * .46))
    for col in range(6):
        means = np.asarray([_inner(R, *roll_points[col, row]) for row in range(10)])
        base = float(np.median(means))
        marked = [
            row for row in range(10)
            if _box_mark_fraction(R, *roll_points[col, row], base, half_w, half_h) >= .15
        ]
        selected.append(marked)
        digits.append(str(marked[0]) if len(marked) == 1 else ('?' if not marked else '*'))
    return ''.join(digits), roll_points, selected


def _standard_roll_grid(R):
    """Fit the six roll-number columns in the upper-left identification block."""
    gray = cv2.medianBlur(R, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=16, param1=70, param2=18,
        minRadius=8, maxRadius=20)
    if circles is None:
        raise ScanError('Could not locate the roll-number grid.')
    height, width = R.shape
    c = circles[0]
    c = c[(c[:, 0] > .10 * width) & (c[:, 0] < .36 * width) &
          (c[:, 1] > .24 * height) & (c[:, 1] < .50 * height)]
    if len(c) < 40:
        raise ScanError('Could not locate enough roll-number bubbles.')
    xs = _fit_axis(c[:, 0], 6, .10 * width, .36 * width,
                   .02 * width, .06 * width)
    ys = _fit_axis(c[:, 1], 10, .24 * height, .50 * height,
                   .015 * height, .04 * height)
    return [(x, y) for x in xs for y in ys]


def _scan_answers(R, points, n, key, width, snapped=None):
    pts = np.array(points, np.float32).reshape(30 if width == 1000 else 50, 4, 2)
    inner = np.array([[_inner(R, *pts[q, k]) for k in range(4)]
                      for q in range(len(pts))])
    ans, st, selected = [], [], []
    cor = wr = bl = db = 0
    for q in range(n):
        block_size = 15 if width == 1000 else 25
        block = (q // block_size) * block_size
        lo, hi = max(block, q - 3), min(block + block_size, q + 4)
        base = np.median(inner[lo:hi])
        if width == 1000:
            mk = []
            for k in range(4):
                candidates = (pts[q, k],) if snapped is None else (pts[q, k], snapped[q, k])
                if any(_portrait_marked(R, x, y, base) for x, y in candidates):
                    mk.append(k)
        else:
            mk = [k for k in range(4) if _is_marked(
                R, (pts[q, k],) if snapped is None else (pts[q, k], snapped[q, k]), base)]
        selected.append(mk)
        expected = key[q] if isinstance(key[q], (list, tuple)) else [key[q]]
        if not mk:
            ans.append('-'); st.append('b'); bl += 1; wr += 1
        elif len(mk) > 1:
            ans.append('*'); st.append('d'); db += 1; wr += 1
        elif mk[0] in expected:
            ans.append(L[mk[0]]); st.append('c'); cor += 1
        else:
            ans.append(L[mk[0]]); st.append('w'); wr += 1
    return pts, inner, ans, st, cor, wr, bl, db, selected


def scan(img, n, key):
    w, profile = rectify(img)
    R = w[:, :, 2]
    if profile == 'portrait30':
        if n > 30:
            raise ScanError('This portrait sheet layout supports up to 30 questions.')
        try:
            points = _portrait_grid(R)
        except ScanError:
            points = _portrait_points(w.shape[1], w.shape[0])
        points = [
            points[row * 8 + col]
            for block_start in (0, 4)
            for row in range(15)
            for col in range(block_start, block_start + 4)
        ]
        gray = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)
        pts, inner, ans, st, cor, wr, bl, db, selected = _scan_answers(gray, points, n, key, 1000)
        ov = w.copy()
        for q in range(n):
            for k in range(4):
                x, y = map(int, pts[q, k])
                expected = key[q] if isinstance(key[q], (list, tuple, set)) else [key[q]]
                if k in expected and st[q] in ('w', 'b', 'd'):
                    cv2.circle(ov, (x, y), 15, (255, 140, 0), 3)
                if k in selected[q]:
                    color = (0, 170, 0) if st[q] == 'c' else ((0, 140, 255) if st[q] == 'd' else (0, 0, 230))
                    cv2.circle(ov, (x, y), 13, color, 4)
        crop = cv2.resize(ov, (800, 1120), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return dict(roll='??????', answers=ans, selected=selected, status=st, correct=cor, wrong=wr,
                    blank=bl, double=db, total=n, overlay='data:image/jpeg;base64,' +
                    base64.b64encode(buf).decode())

    # Detect the printed bubble centres from the rectified image. This adapts
    # to small scale/perspective differences between phone photos instead of
    # assuming that every camera image lands on identical hard-coded coordinates.
    points = _standard_grid(R)
    roll_points = _standard_roll_grid(R)
    pts, inner, ans, st, cor, wr, bl, db, selected = _scan_answers(
        R, points, n, key, 1500
    )
    roll, roll_pts, roll_selected = _read_box_roll(R, roll_points)

    ov = w.copy()
    for q in range(n):
        for k in range(4):
            x, y = map(int, pts[q, k])
            expected = key[q] if isinstance(key[q], (list, tuple, set)) else [key[q]]
            if k in expected and st[q] in ('w', 'b', 'd'):
                cv2.rectangle(ov, (x - 13, y - 10), (x + 13, y + 10), (255, 140, 0), 2)
            if k in selected[q]:
                color = (0, 170, 0) if st[q] == 'c' else ((0, 140, 255) if st[q] == 'd' else (0, 0, 230))
                cv2.rectangle(ov, (x - 12, y - 9), (x + 12, y + 9), color, 3)
    for col in range(6):
        for row in range(10):
            x, y = map(int, roll_pts[col, row])
            if row in roll_selected[col]:
                cv2.rectangle(ov, (x - 9, y - 9), (x + 9, y + 9), (0, 0, 230), 2)
    crop = ov[250:1200, 150:1300]
    crop = cv2.resize(crop, (800, int(800 * crop.shape[0] / crop.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return dict(roll=roll, answers=ans, selected=selected, status=st, correct=cor, wrong=wr, blank=bl, double=db, total=n,
                overlay='data:image/jpeg;base64,' + base64.b64encode(buf).decode())


def load_pages(name, data):
    """Yield (label, BGR image) for an image or PDF upload."""
    if name.lower().endswith('.pdf') or data[:4] == b'%PDF':
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        doc = fitz.open(stream=data, filetype='pdf')
        for i, p in enumerate(doc):          # lazily, one page at a time (saves RAM)
            pix = p.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), alpha=False)
            a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            yield f'{name} p{i + 1}', cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        return
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ScanError('This file could not be read as an image or PDF.')
    yield name, im
