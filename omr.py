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
    s = 1400.0 / img.shape[1]
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    f = find_anchors(small)
    if f is None:
        return _rectify_page(img), 'portrait30'
    (tl, tr), (bl, br) = f
    if tl[1] > bl[1]:
        raise ScanError('The sheet appears to be upside down. Rotate it and capture it again.')
    if (tr[0] - tl[0]) / s < 500:
        raise ScanError('The photo resolution is too low. Capture the sheet from a closer distance or use a higher-quality image.')
    src = np.float32([tl, tr, br, bl]) / s
    dst = np.float32([(A, A), (A + S, A), (A + S, A + S), (A, A + S)])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (FW, FH), flags=cv2.INTER_CUBIC), 'standard'


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
    """Marked when >=30% of the bubble disc is clearly darker than local paper (checks nominal and snapped centre)."""
    return max(float((_patch(R, x, y) < base * .72).mean()) for x, y in pts) >= .30


def _portrait_marked(R, x, y, base):
    """Detect a dark filled bubble while ignoring printed option text and paper tint."""
    patch = _patch(R, x, y, 9)
    mean = float(patch.mean())
    dark_fraction = float((patch < min(75, base * .65)).mean())
    return mean < min(40, base * .38) and dark_fraction >= .45


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


def _scan_answers(R, points, n, key, width, snapped=None):
    pts = np.array(points, np.float32).reshape(30 if width == 1000 else 50, 4, 2)
    inner = np.array([[_inner(R, *pts[q, k]) for k in range(4)]
                      for q in range(len(pts))])
    ans, st = [], []
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
        if not mk:
            ans.append('-'); st.append('b'); bl += 1
        elif len(mk) > 1:
            ans.append('*'); st.append('d'); db += 1
        elif mk[0] == key[q]:
            ans.append(L[mk[0]]); st.append('c'); cor += 1
        else:
            ans.append(L[mk[0]]); st.append('w'); wr += 1
    return pts, inner, ans, st, cor, wr, bl, db


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
        pts, inner, ans, st, cor, wr, bl, db = _scan_answers(gray, points, n, key, 1000)
        ov = w.copy()
        for q in range(n):
            for k in range(4):
                x, y = map(int, pts[q, k])
                if key[q] == k and st[q] in ('w', 'b', 'd'):
                    cv2.circle(ov, (x, y), 15, (255, 140, 0), 3)
                if st[q] in ('c', 'w') and ans[q] == L[k]:
                    cv2.circle(ov, (x, y), 13, (0, 170, 0) if st[q] == 'c' else (0, 0, 230), 4)
        crop = cv2.resize(ov, (800, 1120), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return dict(roll='??????', answers=ans, status=st, correct=cor, wrong=wr,
                    blank=bl, double=db, total=n, overlay='data:image/jpeg;base64,' +
                    base64.b64encode(buf).decode())

    g = cv2.medianBlur(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY), 5)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=70, param2=18, minRadius=9, maxRadius=17)
    circles = c[0] if c is not None else np.zeros((0, 3), np.float32)

    exp = [(A + U[q // 25][k] * S, A + (V0 + DV * (q % 25)) * S) for q in range(50) for k in range(4)]
    rexp = [(A + RU[c_] * S, A + (RV0 + RDV * r_) * S) for c_ in range(6) for r_ in range(10)]
    Hf = _fit_H(exp, circles)
    pts, spts = (a.reshape(50, 4, 2) for a in _snap(exp, circles, Hf))
    pts, inner, ans, st, cor, wr, bl, db = _scan_answers(
        R, pts.reshape(-1, 2), n, key, 1500, spts)

    rfit = cv2.perspectiveTransform(np.array(rexp, np.float32)[None], Hf)[0]
    if len(circles):   # small local shift for the roll block (it sits far from the answer grid)
        d = np.linalg.norm(rfit[:, None, :] - circles[None, :, :2], axis=2)
        j = d.argmin(1)
        ok = d[np.arange(60), j] < 30
        if ok.sum() >= 15:
            off = np.median(circles[j[ok], :2] - rfit[ok], axis=0)
            if np.abs(off).max() < 30:
                Hr = np.array([[1, 0, off[0]], [0, 1, off[1]], [0, 0, 1]])
                Hf_roll = Hr @ Hf
            else:
                Hf_roll = Hf
        else:
            Hf_roll = Hf
    else:
        Hf_roll = Hf
    rp, rsp = (a.reshape(6, 10, 2) for a in _snap(rexp, circles, Hf_roll))
    roll = ''
    for c_ in range(6):
        col = np.array([_inner(R, *rp[c_, r_]) for r_ in range(10)])
        base = np.median(col)
        mk = [r_ for r_ in range(10) if _is_marked(R, (rp[c_, r_], rsp[c_, r_]), base)]
        roll += str(mk[0]) if len(mk) == 1 else ('?' if not mk else '*')

    ov = w.copy()
    for q in range(n):
        for k in range(4):
            x, y = int(pts[q, k, 0]), int(pts[q, k, 1])
            if key[q] == k and st[q] in ('w', 'b', 'd'):
                cv2.circle(ov, (x, y), 15, (255, 140, 0), 3)
            if inner[q, k] < 255 and ans[q] == L[k] and st[q] in ('c', 'w'):
                cv2.circle(ov, (x, y), 13, (0, 170, 0) if st[q] == 'c' else (0, 0, 230), 4)
            elif st[q] == 'd' and _is_marked(R, (pts[q, k], spts[q, k]), np.median(inner[max(0, q - 3):q + 4])):
                cv2.circle(ov, (x, y), 13, (0, 140, 255), 4)
    crop = ov[250:1200, 150:1300]
    crop = cv2.resize(crop, (800, int(800 * crop.shape[0] / crop.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return dict(roll=roll, answers=ans, status=st, correct=cor, wrong=wr, blank=bl, double=db, total=n,
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
