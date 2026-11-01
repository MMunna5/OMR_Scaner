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


def rectify(img):
    s = 1400.0 / img.shape[1]
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    f = find_anchors(small)
    if f is None:
        raise ScanError('Could not find the sheet corners (anchors). Capture the entire sheet in good lighting and keep it straight.')
    (tl, tr), (bl, br) = f
    if tl[1] > bl[1]:
        raise ScanError('The sheet appears to be upside down. Rotate it and capture it again.')
    if (tr[0] - tl[0]) / s < 500:
        raise ScanError('The photo resolution is too low. Capture the sheet from a closer distance or use a higher-quality image.')
    src = np.float32([tl, tr, br, bl]) / s
    dst = np.float32([(A, A), (A + S, A), (A + S, A + S), (A, A + S)])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (FW, FH), flags=cv2.INTER_CUBIC)


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


def scan(img, n, key):
    w = rectify(img)
    R = w[:, :, 2]
    g = cv2.medianBlur(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY), 5)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=70, param2=18, minRadius=9, maxRadius=17)
    circles = c[0] if c is not None else np.zeros((0, 3), np.float32)

    exp = [(A + U[q // 25][k] * S, A + (V0 + DV * (q % 25)) * S) for q in range(50) for k in range(4)]
    rexp = [(A + RU[c_] * S, A + (RV0 + RDV * r_) * S) for c_ in range(6) for r_ in range(10)]
    Hf = _fit_H(exp, circles)
    pts, spts = (a.reshape(50, 4, 2) for a in _snap(exp, circles, Hf))
    inner = np.array([[_inner(R, *pts[q, k]) for k in range(4)] for q in range(50)])

    ans, st = [], []
    cor = wr = bl = db = 0
    for q in range(n):
        blk = (q // 25) * 25
        lo, hi = max(blk, q - 3), min(blk + 25, q + 4)
        base = np.median(inner[lo:hi])
        mk = [k for k in range(4) if _is_marked(R, (pts[q, k], spts[q, k]), base)]
        if not mk:
            ans.append('-'); st.append('b'); bl += 1
        elif len(mk) > 1:
            ans.append('*'); st.append('d'); db += 1
        elif mk[0] == key[q]:
            ans.append(L[mk[0]]); st.append('c'); cor += 1
        else:
            ans.append(L[mk[0]]); st.append('w'); wr += 1

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
