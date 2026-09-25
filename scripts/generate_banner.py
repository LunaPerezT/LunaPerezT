#!/usr/bin/env python3
"""
Generate the animated terminal banner for the LunaPerezT profile README.

    python scripts/generate_banner.py [path/to/photo.jpg]

Writes assets/banner-dark.svg and assets/banner-light.svg.

Two panels inside one terminal window:

  VISUAL.MAP    a stipple portrait - the source photo resampled into a cloud of
                dots whose local density follows image luminance. Pure vector:
                the photo itself is never embedded, so nothing identifiable is
                shipped and the file stays small.
  SYSTEM.INFO   a neofetch-style key/value table, revealed one row at a time.

Self-contained SVG: no external fonts, no network, no JavaScript. The reveal is
SMIL (<animate> on clip rects), which is what runs when an SVG is loaded through
an <img> tag - how GitHub serves README images.

Run with no argument and the portrait panel renders an "awaiting source" state.
"""
from __future__ import annotations

import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

# ---------------------------------------------------------------- content ---
ROWS = [
    ("Subject",        "Luna Pérez Troncoso"),
    ("Role",           "Data Scientist · AI Engineer"),
    ("Origin",         "Madrid · Spain"),
    ("Education",      "MSc Data Science & AI · MSc Data Engineering"),
    ("Focus",          "Medical & health AI"),
    ("Status",         "Building + Evaluating + Shipping"),
    ("Core.Lang",      "Python · SQL · R"),
    ("Core.DL",        "PyTorch · TensorFlow · Keras"),
    ("Core.Data",      "PySpark · Spark ML · Parquet · Databricks"),
    ("Core.ML",        "scikit-learn · XGBoost · LightGBM · Optuna"),
    ("Core.MLOps",     "MLflow · Docker · AWS · pytest"),
    ("Core.Forecast",  "Darts · LightGBM Tweedie · Optuna"),
    ("Grid.Portfolio", "lunaperezt.github.io"),
    ("Grid.LinkedIn",  "/in/luna-pérez-troncoso"),
    ("Grid.GitHub",    "LunaPerezT"),
]
TITLE   = "profile.sh --live"
HANDLE  = "@LunaPerezT"
STATUS  = "ALL SYSTEMS NOMINAL"
NODE    = "UTC+2 · MADRID NODE"

# ----------------------------------------------------------------- themes ---
THEMES = {
    "dark": dict(
        bg="#0D1117", chrome="#161B22", border="#30363D", panel="#0B0F16",
        panel_border="#243040", title="#8B949E", label="#58A6FF", dim="#6E7681",
        key="#8B949E", val="#E6EDF3", accent="#58A6FF", dot="#A371F7",
        live="#FF7B72", ok="#3FB950", chip_bg="#132A47", chip_fg="#79C0FF",
        d1="#FF5F57", d2="#FEBC2E", d3="#28C840",
    ),
    "light": dict(
        bg="#FFFFFF", chrome="#F6F8FA", border="#D1D9E0", panel="#F9FAFB",
        panel_border="#D8DEE4", title="#59636E", label="#0550AE", dim="#8C959F",
        key="#59636E", val="#1F2328", accent="#0550AE", dot="#6639BA",
        live="#CF222E", ok="#1A7F37", chip_bg="#DDF4FF", chip_fg="#0550AE",
        d1="#FF5F57", d2="#FEBC2E", d3="#28C840",
    ),
}

# ---------------------------------------------------------------- metrics ---
W, H, BAR = 1180, 560, 44
PAD = 26
PANEL_Y = BAR + 22
PANEL_H = H - PANEL_Y - 24
LW = 430                      # left panel width
GAP = 14
RX = PAD + LW + GAP           # right panel x
RW = W - RX - PAD

FS, ROW_H = 15, 24.5
CH = FS * 0.6
FONT = ("ui-monospace, SFMono-Regular, &apos;SF Mono&apos;, Menlo, Consolas, "
        "&apos;DejaVu Sans Mono&apos;, monospace")

N_POINTS = 9000               # dots in each of the three states
DOT = 2                       # side of every dot, in viewBox units
CHUNK = 250                   # dots per <path>; only affects file shape, not accuracy
# Blob size scales as sqrt(shape_area * GROUP / N_POINTS): drop the point count
# without dropping GROUP with it and every target shape smears. These two move
# together.
CYCLE = 14                    # seconds for the full portrait->py->nn->portrait loop
KEYTIMES = "0;.30;.40;.58;.68;.86;1"
SPLINES = ";".join([".4 0 .2 1"] * 6)
ROW_STEP = 0.16               # seconds between rows appearing


def _f(size, extra=""):
    return f'font-family="{FONT}" font-size="{size}px"{extra}'


# ------------------------------------------------------------- stipple ------
def stipple(photo: Path | None, box_w: float, box_h: float, seed: int = 7):
    """Return a list of (x, y, r) dots approximating the photo inside the box.

    Density follows luminance: a pixel is kept with probability proportional to
    how far it is from the darkest tone, so a lit subject on a darker background
    draws itself and the background stays empty. `--invert` flips that.
    """
    rng = np.random.default_rng(seed)

    if photo is None:
        return None                      # caller renders the placeholder state

    from PIL import Image, ImageOps, ImageFilter

    im = ImageOps.exif_transpose(Image.open(photo)).convert("L")

    # --- 1. separate subject from background ------------------------------
    # The dots stand for LIGHT falling on the subject, so density has to follow
    # brightness. On a photo shot against a bright wall that would draw the wall
    # instead, so the background is removed first: flood-fill inward from the
    # borders through everything that matches the border's own tone.
    from scipy import ndimage

    small = im.resize((360, max(1, round(360 * im.height / im.width))), Image.BILINEAR)
    g = np.asarray(small, dtype=np.float64) / 255.0
    Hs = g.shape[0]

    # Sample the background tone from the TOP band only. Sampling the whole
    # border fails on a normal portrait: the subject touches the bottom edge,
    # which inflates the spread until the flood fill swallows the whole frame.
    top = g[: max(2, int(Hs * 0.14))]
    tone = float(np.median(top))
    mad = float(np.median(np.abs(top - tone)))
    spread = float(np.clip(4.5 * mad, 0.045, 0.16))

    # Two independent reasons to call a pixel background, OR-ed together. Tone
    # alone is not enough: a shaded corner of the wall sits outside the
    # tolerance and stays attached to the subject, so the largest-component
    # trick cannot drop it. The wall is also perfectly flat, and the subject
    # never is, so local texture separates what tone cannot.
    loc_mean = ndimage.uniform_filter(g, 9)
    loc_std = np.sqrt(np.maximum(ndimage.uniform_filter(g * g, 9) - loc_mean ** 2, 0))
    flat = (loc_std < 0.020) & (g > tone - 0.30)

    hist, _ = np.histogram(g, 256, (0.0, 1.0))
    hist = hist.astype(float)
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    mu = np.cumsum(hist * np.arange(256))
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (mu[-1] * w0 / w0[-1] - mu) ** 2 / (w0 * w1 + 1e-9)
    otsu = float(np.nanargmax(between)) / 255.0

    seed = flat | (g > otsu) | (np.abs(g - tone) < spread)
    lab, _ = ndimage.label(seed)
    edge_labels = set(lab[0].tolist()) | set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
    edge_labels.discard(0)
    bg = ndimage.binary_closing(np.isin(lab, list(edge_labels)), np.ones((7, 7)))

    # NB: no binary_fill_holes here. The subject touches the bottom edge, so the
    # background is not an enclosed hole and fill_holes would return everything.
    subject = ndimage.binary_opening(~bg, np.ones((5, 5)))
    lab2, n2 = ndimage.label(subject)
    if n2 > 1:                                   # drop stray patches of wall
        sizes = ndimage.sum(subject, lab2, range(1, n2 + 1))
        subject = lab2 == (int(np.argmax(sizes)) + 1)
    subject = ndimage.binary_closing(subject, np.ones((9, 9)))

    if subject.sum() < 0.05 * subject.size:      # separation failed - keep all
        subject = np.ones_like(subject, dtype=bool)

    # --- 2. frame on the face ---------------------------------------------
    # Cropping to the whole subject leaves the face small: at panel size the
    # eyes and mouth fall below the resolution the dots can carry. Anchor the
    # frame on the detected face instead, so the head fills the panel.
    face = None
    try:
        import cv2
        det = cv2.resize(np.asarray(im), (720, max(1, round(720 * im.height / im.width))))
        cas = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        found = cas.detectMultiScale(cv2.equalizeHist(det), 1.08, 6, minSize=(60, 60))
        if len(found):
            k = im.width / det.shape[1]
            fx, fy, fw, fh = max(found, key=lambda r: r[2] * r[3])
            face = (fx * k, fy * k, fw * k, fh * k)
    except Exception:
        face = None

    want = box_w / box_h
    if face is not None:
        fx, fy, fw, fh = face
        fcx, fcy = fx + fw / 2, fy + fh / 2
        fh_frac = 0.42                       # face height as a share of the frame
        ch = fh / fh_frac
        cw = ch * want
        x0, y0 = fcx - cw / 2, fcy - 0.38 * ch     # eyes land near the upper third
        x1, y1 = x0 + cw, y0 + ch
    else:
        rs, cs = np.where(subject)
        sy, sx = im.height / subject.shape[0], im.width / subject.shape[1]
        x0, x1 = cs.min() * sx, cs.max() * sx
        y0, y1 = rs.min() * sy, rs.max() * sy
        pad = 0.05 * max(x1 - x0, y1 - y0)
        x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
        cw, ch = x1 - x0, y1 - y0
        if cw / ch < want:
            need, cx = ch * want, (x0 + x1) / 2
            x0, x1 = cx - need / 2, cx + need / 2
        else:
            need, cy = cw / want, (y0 + y1) / 2
            y0, y1 = cy - need / 2, cy + need / 2

    dx = max(0, -x0) - max(0, x1 - im.width)
    dy = max(0, -y0) - max(0, y1 - im.height)
    box = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)

    full_subject = np.asarray(
        Image.fromarray((subject * 255).astype("uint8")).resize(im.size, Image.BILINEAR)
    ) > 110

    im_c = im.crop(tuple(int(v) for v in box))
    sub_c = Image.fromarray((full_subject * 255).astype("uint8")).crop(
        tuple(int(v) for v in box))

    # --- 3. scale into the panel ------------------------------------------
    scale = min(box_w / im_c.width, box_h / im_c.height)
    tw, th = max(1, int(im_c.width * scale)), max(1, int(im_c.height * scale))
    im_c = im_c.resize((tw, th), Image.LANCZOS)
    sub_c = sub_c.resize((tw, th), Image.BILINEAR)
    tw_, th_ = tw, th

    v = np.asarray(im_c, dtype=np.float64) / 255.0
    m = (np.asarray(sub_c, dtype=np.float64) / 255.0)

    # face box in panel pixels, used to sharpen the features harder than the rest
    face_w = np.zeros_like(v)
    if face is not None:
        k = tw / (box[2] - box[0])
        fx0 = (face[0] - box[0]) * k; fy0 = (face[1] - box[1]) * k
        fx1 = fx0 + face[2] * k;      fy1 = fy0 + face[3] * k
        yy, xx = np.mgrid[0:th, 0:tw]
        inside = ((xx > fx0 - 6) & (xx < fx1 + 6) & (yy > fy0 - 6) & (yy < fy1 + 10))
        face_w = ndimage.gaussian_filter(inside.astype(float), 9)
        face_w /= max(face_w.max(), 1e-6)

    # --- 4. tone curve -----------------------------------------------------
    # Density has to read as light: bright skin dense, eyes and lips sparse.
    # An earlier version ADDED edge energy to the density, which inverted that -
    # it made the eyes the densest thing on the face. Local histogram
    # equalisation is the right tool: it stretches contrast inside the face
    # without touching the global relationship between subject and background.
    inside = v[m > 0.5]
    if inside.size:
        lo, hi = np.percentile(inside, [3, 97])
        v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)

    try:
        import cv2
        clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
        v = clahe.apply((v * 255).astype("uint8")).astype(np.float64) / 255.0
    except Exception:
        pass

    # a light unsharp on top, pushed harder over the face
    v = np.clip(v + (0.30 + 0.55 * face_w) * (v - ndimage.gaussian_filter(v, 1.6)), 0, 1)

    # hair strands are fine high-frequency detail that tone alone thins out;
    # a small edge term keeps them, too small to fill the features back in
    gx = ndimage.sobel(v, axis=1); gy = ndimage.sobel(v, axis=0)
    e = np.hypot(gx, gy)
    e = e / max(e.max(), 1e-6)

    # 0.86 ceiling: above that the densest areas fill in solid and stop reading
    # as a stipple. The 0.08 floor keeps dark hair present as a silhouette.
    a = np.clip(0.05 + 0.90 * v + 0.14 * e, 0, 0.95) * m
    return a, (tw, th)


# The official Python mark, rasterised once from the simple-icons path and
# packed here as a 320x320 1-bit mask (zlib + base64, ~1.2 KB). Embedding the
# bitmap rather than re-drawing the shape by hand keeps the logo correct, and
# rather than rasterising an SVG at build time keeps the script dependency-free.
PYTHON_MASK_B64 = (
    "eNrt202SsyAQBmCtLFx6BI/i0fRoHoXdbFm6oOjJT01AtF9elRmrvi+un0qkabCDnapKrlpEbJW9BnlcPsd6eV0Os0Z+LgPdm+FvboODHzhEDtzhTeJrpL4WBbFfOEeM9nlxtycyZYOMb7BL3EwNQx/IINxAUqZEul65iQqLlgprZ6nwaYFpV86RzpNOqGlDriIC3T0G2OcD3T8C1lJuSoJoVHej3Jgkg1VcmlyK8zvckE0E3rlfcP0R5867oaDzF7lVvjgy/866NO81Z5J1pLk52RQ055I9WnPyJZyTj/u4/8xtFDIf91cOFuaJm1HJFDnRK86lm/RKbeGsVhAnzmW+9sd5rTBNnODRBjfh23s7s12vr5zFw3i7ebsOXzkHoxycx8N9O1HW96YbCDfCsAQ3kc7A8C3cjXEWZcsBN8PpWLiOcY50Hk7bfidweg85odzIuppzE+tQutyna68zzTXOsq69xs0one/LrLCr9zqHloeYi1zIOtZ56MbghhIubI5FnCedC5stdDPclPc7Czfl+JiGcxPclONjKc7hTTmuJsmwVORwa3IYN4Z5/MyNb6/hokxNr8el0uJMr9eHiE9yt1mtjzBzQrt9zNiq+ZE9Kd38taEuh2O3152LipDM4Zp7EeWBWv1c/houjx5u4PKXXV4NuUt0pOvJ5UruOtwwHBfl+7Q1pCu7uWMXnkcWhi84A8N3xAnlcIV8wMFpC/UL60bWNYUdrmi7va4q7XBF217jTGnXlHUh68q4kJ14lwy/V7GrWFdRzhd2LmzyRdwcHi5FnA0PNehMKI6KuCnUoNCNpIuqqJyrCeej6g2f73Fujs7/kLPReSJyJioHkRujqha5+BgYOB+Xq8DNcfkLnImPs4Gb4hcpuvOLMl5386Lu1t20qONVl7wOiJz2PqEmXfr+6OP+xum/6D7u4/5x98U5/+r7yLpnCwnh7OP5Q7hniwvlasqNpHs+BvLu1SJ0jesJ5/7OdUec/wU3nHNtmgesE8bp+deQeZ+2RGrulqy3lnLu1ceZ76OtRGtozR/Qas5y/cCK60nXaYVp9nh4Iht92QZjtmGZbID2JxuvO2ra1oExJxvN2cZ1uhGebayvudtbfrGn/peQ+VPEIFyV2nAfF+4w+w+LgWMvuJ7Zb9XbW7g="
)
PYTHON_MASK_SIZE = 320


def _python_mask():
    import base64
    import zlib

    raw = zlib.decompress(base64.b64decode(PYTHON_MASK_B64))
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    n = PYTHON_MASK_SIZE
    return bits[: n * n].reshape(n, n).astype(np.float64)


def _shapes(W, H):
    """Density maps for the two non-photographic states of VISUAL.MAP."""
    from PIL import Image, ImageDraw

    S = 1000

    # --- Python: the real mark -------------------------------------------
    py = Image.fromarray((_python_mask() * 255).astype("uint8")).resize(
        (S, S), Image.LANCZOS)

    # --- neural net ------------------------------------------------------
    # Solid nodes, not rings: at the resolution a translated group of dots can
    # carry, a ring's hole collapses and the whole thing reads as noise. Few
    # layers, generous spacing, and edges thick enough to survive the same
    # limit.
    nn = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(nn)
    layers = [4, 5, 3]
    xs = np.linspace(155, 845, len(layers))
    pos = []
    for x, n in zip(xs, layers):
        ys = np.linspace(500 - (n - 1) * 175 / 2, 500 + (n - 1) * 175 / 2, n)
        pos.append([(x, y) for y in ys])
    # draw the mesh first and faint, so the solid nodes stay the brightest
    # thing in the frame and survive the group-translation blur
    for a_, b_ in zip(pos, pos[1:]):
        for (x1, y1) in a_:
            for (x2, y2) in b_:
                d.line((x1, y1, x2, y2), fill=58, width=14)
    for layer in pos:
        for (x, y) in layer:
            d.ellipse((x - 60, y - 60, x + 60, y + 60), fill=255)

    out = []
    for img, fit in ((py, 0.78), (nn, 0.92)):
        k = min(W / S, H / S) * fit
        t = img.resize((int(S * k), int(S * k)), Image.LANCZOS)
        canvas = np.zeros((H, W))
        ox, oy = (W - t.width) // 2, (H - t.height) // 2
        canvas[oy:oy + t.height, ox:ox + t.width] = np.asarray(t) / 255.0
        out.append(np.clip(canvas, 0, 1) * 0.92)
    return out


def _sample(a, n, rng, cell=1):
    """n points placed by Floyd-Steinberg error diffusion over the density map.

    Rejection sampling was the obvious choice and the wrong one: independent
    random draws are Poisson, so they clump and leave holes, and a logo with
    large solid areas comes out looking eaten. Error diffusion spreads the same
    number of points evenly - blue noise rather than white - which is what a
    stipple wants. Returns (x, y, density) per point.
    """
    a = np.asarray(a, dtype=np.float64)
    # Dither on a grid whose cell equals the dot size. A 1px dot renders below
    # one device pixel once the banner is scaled into a README column and fades
    # out; a bigger dot on a matching grid keeps ink coverage exactly equal to
    # the density, so the tone mapping stays linear instead of saturating where
    # neighbouring dots would overlap.
    if cell > 1:
        h0, w0 = a.shape
        hh, ww = h0 // cell, w0 // cell
        a = a[: hh * cell, : ww * cell].reshape(hh, cell, ww, cell).mean(axis=(1, 3))
    total = a.sum()
    if total <= 0:
        return []
    work = a * (n / total)                      # so the dither yields ~n dots
    work = np.clip(work, 0, 1)
    # renormalise once after clipping, or bright areas silently lose their share
    if work.sum() > 0:
        work *= min(3.0, n / work.sum())
        work = np.clip(work, 0, 1)

    h, w = work.shape
    err = work.copy()
    pts = []
    for y in range(h):
        row = err[y]
        for x in range(w):
            old = row[x]
            new_v = 1.0 if old > 0.5 else 0.0
            if new_v:
                pts.append((x * cell, y * cell, float(a[y, x])))
            e = old - new_v
            if x + 1 < w:
                row[x + 1] += e * 7 / 16
            if y + 1 < h:
                if x > 0:
                    err[y + 1][x - 1] += e * 3 / 16
                err[y + 1][x] += e * 5 / 16
                if x + 1 < w:
                    err[y + 1][x + 1] += e * 1 / 16

    if len(pts) > n:
        keep = rng.choice(len(pts), n, replace=False)
        pts = [pts[i] for i in sorted(keep)]
    while len(pts) < n and pts:                 # pad so all states match in count
        pts.append(pts[rng.integers(0, len(pts))])
    return pts


def _hilbert_d(x, y, order=8):
    """Index of (x, y) along a Hilbert curve on a 2**order grid."""
    n = 1 << order
    rx = ry = 0
    d = 0
    s_ = n >> 1
    while s_ > 0:
        rx = 1 if (x & s_) > 0 else 0
        ry = 1 if (y & s_) > 0 else 0
        d += s_ * s_ * ((3 * rx) ^ ry)
        # rotate
        if ry == 0:
            if rx == 1:
                x = s_ - 1 - x
                y = s_ - 1 - y
            x, y = y, x
        s_ >>= 1
    return d


def _compact_order(pts, w, h, order=8):
    """Sort points along a Hilbert curve.

    Consecutive indices then form COMPACT clusters. A serpentine scan also keeps
    neighbours together, but its groups come out as long flat strips: translate
    a strip onto a thin diagonal edge of a neural-net diagram and it smears. A
    Hilbert group is a small square blob, which lands cleanly in any target.
    """
    n = 1 << order
    keyed = []
    for p in pts:
        gx = min(n - 1, max(0, int(p[0] / max(w, 1e-9) * (n - 1))))
        gy = min(n - 1, max(0, int(p[1] / max(h, 1e-9) * (n - 1))))
        keyed.append((_hilbert_d(gx, gy, order), p))
    keyed.sort(key=lambda t: t[0])
    return [p for _, p in keyed]


def visual_map_svg(c, photo, px, py, pw, ph):
    """The animated point cloud: portrait -> Python -> neural net -> portrait.

    Built the way a large cloud has to be if the file is to stay sane: the dots
    are chunked into ~1k groups and each group is translated between the three
    layouts with one <animateTransform>. Animating every dot individually, or
    morphing path data, multiplies the file by the number of points.
    """
    rng = np.random.default_rng(11)
    res = stipple(photo, pw, ph)
    if res is None:
        cx, cy = px + pw / 2, py + ph / 2
        return ("\n".join([
            f'<text {_f(13)} x="{cx:.0f}" y="{cy - 6:.0f}" text-anchor="middle" '
            f'fill="{c["dim"]}">[ awaiting source image ]</text>',
            f'<text {_f(11)} x="{cx:.0f}" y="{cy + 16:.0f}" text-anchor="middle" '
            f'fill="{c["dim"]}">python scripts/generate_banner.py photo.jpg</text>']),
            0, "NO SIGNAL")

    a, (tw, th) = res
    ox, oy = (pw - tw) / 2, (ph - th) / 2

    home_pts = [(ox + x, oy + y, r) for x, y, r in _sample(a, N_POINTS, rng, DOT)]
    states = [_compact_order(home_pts, pw, ph)]
    for dens in _shapes(int(pw), int(ph)):
        states.append(_compact_order(_sample(dens, N_POINTS, rng, DOT), pw, ph))

    # Morph the path data itself rather than translating groups of dots.
    # A rigid <animateTransform> per group carries the portrait's local
    # arrangement into every target, so each group lands as a blob roughly its
    # own size off the outline and the logo comes out fuzzy. Animating "d"
    # moves every dot to its exact position in each state - the structure of
    # the path (one M/m + h v h z per dot) is identical across states, which is
    # what SMIL needs to interpolate it.
    def encode(pts):
        # Each dot is a DOT-long vertical stroke of width DOT, i.e. a DOT x DOT
        # square - four times shorter to write than the equivalent filled box
        # ("m5 -3v2" against "m5 -3h2v2h-2z"), and that matters when the same
        # geometry is repeated once per keyframe. Deltas are relative and the
        # Hilbert order keeps them to one or two digits.
        out_, px_, py_ = [], 0, 0
        for i, (x, y, _) in enumerate(pts):
            X, Y = round(px + x), round(py + y)
            if i == 0:
                out_.append(f"M{X} {Y}v{DOT}")
            else:
                out_.append(f"m{X - px_} {Y - py_}v{DOT}")
            px_, py_ = X, Y + DOT      # v leaves the pen DOT below Y
        return "".join(out_)

    out = []
    for g in range(0, N_POINTS, CHUNK):
        ds = [encode(st[g:g + CHUNK]) for st in states]
        if not ds[0]:
            break
        vals = ";".join([ds[0], ds[0], ds[1], ds[1], ds[2], ds[2], ds[0]])
        out.append(
            f'<path d="{ds[0]}">'
            f'<animate attributeName="d" values="{vals}" keyTimes="{KEYTIMES}" '
            f'dur="{CYCLE}s" calcMode="spline" keySplines="{SPLINES}" '
            f'repeatCount="indefinite"/></path>')

    body = "\n".join(out)
    return (f'<g fill="none" stroke="{c["dot"]}" stroke-width="{DOT}" '
            f'stroke-linecap="butt" opacity="0.9" shape-rendering="crispEdges">{body}</g>',
            N_POINTS, "SRC 3-STATE / 1-BIT")


# ------------------------------------------------------------------ build ---
def build(theme: str, photo: Path | None) -> str:
    c = THEMES[theme]
    clips, rows, t0 = [], [], 0.45

    # ---- left panel ----
    lx, ly, lh = PAD, PANEL_Y, PANEL_H
    inner_pad_top, inner_pad = 34, 18
    pts_svg, n_pts, src_label = visual_map_svg(
        c, photo, lx + inner_pad, ly + inner_pad_top,
        LW - 2 * inner_pad, lh - inner_pad_top - 30)

    # ---- right panel rows ----
    ry = ly + 52
    key_w = max(len(k) for k, _ in ROWS) + 1
    for i, (k, v) in enumerate(ROWS):
        cid = f"r{i}"
        clips.append(
            f'<clipPath id="{cid}"><rect x="{RX}" y="{ry - FS - 3:.1f}" '
            f'width="0" height="{FS + 9}">'
            f'<animate attributeName="width" from="0" to="{RW}" '
            f'begin="{t0 + i * ROW_STEP:.2f}s" dur="0.34s" fill="freeze"/>'
            f'</rect></clipPath>')
        leader_x1 = RX + 18 + len(k) * CH + 8
        leader_x2 = RX + RW - 18 - len(v) * CH - 8
        leader = ("" if leader_x2 - leader_x1 < 12 else
                  f'<line x1="{leader_x1:.0f}" y1="{ry - 4:.0f}" x2="{leader_x2:.0f}" '
                  f'y2="{ry - 4:.0f}" stroke="{c["dim"]}" stroke-width="1" '
                  f'stroke-dasharray="1 4" opacity="0.5"/>')
        rows.append(
            f'<g clip-path="url(#{cid})">{leader}'
            f'<text {_f(FS)} x="{RX + 18}" y="{ry:.1f}" fill="{c["key"]}">{escape(k)}</text>'
            f'<text {_f(FS)} x="{RX + RW - 18}" y="{ry:.1f}" text-anchor="end" '
            f'fill="{c["val"]}">{escape(v)}</text></g>')
        ry += ROW_H

    t_end = t0 + len(ROWS) * ROW_STEP + 0.4

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" \
role="img" aria-label="Terminal banner: portrait and system information for Luna Perez Troncoso, \
Data Scientist and AI Engineer, Madrid">
<title>profile.sh --live — Luna Pérez Troncoso</title>
<defs>
{chr(10).join(clips)}
<linearGradient id="edge" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="#0981F7"/><stop offset="1" stop-color="#7B2FBE"/>
</linearGradient>
</defs>

<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="12" fill="{c['bg']}" stroke="{c['border']}"/>
<path d="M1 13 A12 12 0 0 1 13 1 H{W-13} A12 12 0 0 1 {W-1} 13 V{BAR} H1 Z" fill="{c['chrome']}"/>
<line x1="1" y1="{BAR}" x2="{W-1}" y2="{BAR}" stroke="{c['border']}"/>
<rect x="1" y="{H-5}" width="{W-2}" height="4" fill="url(#edge)" opacity="0.9"/>
<circle cx="26" cy="{BAR//2}" r="6" fill="{c['d1']}"/>
<circle cx="47" cy="{BAR//2}" r="6" fill="{c['d2']}"/>
<circle cx="68" cy="{BAR//2}" r="6" fill="{c['d3']}"/>
<text {_f(13)} x="{W//2}" y="{BAR//2 + 5}" text-anchor="middle" fill="{c['title']}">{escape(TITLE)}</text>

<!-- VISUAL.MAP -->
<rect x="{lx}" y="{ly}" width="{LW}" height="{lh}" rx="7" fill="{c['panel']}" stroke="{c['panel_border']}"/>
<text {_f(12, ' letter-spacing="1.2"')} x="{lx + 16}" y="{ly + 22}" fill="{c['label']}" font-weight="700">VISUAL.MAP</text>
<text {_f(10)} x="{lx + LW - 16}" y="{ly + 22}" text-anchor="end" fill="{c['dim']}">{escape(src_label)}</text>
<path d="M{lx+18} {ly+44} h14 M{lx+18} {ly+44} v14" stroke="{c['accent']}" stroke-width="1.4" fill="none" opacity="0.7"/>
<path d="M{lx+LW-18} {ly+44} h-14 M{lx+LW-18} {ly+44} v14" stroke="{c['accent']}" stroke-width="1.4" fill="none" opacity="0.7"/>
<path d="M{lx+18} {ly+lh-24} h14 M{lx+18} {ly+lh-24} v-14" stroke="{c['accent']}" stroke-width="1.4" fill="none" opacity="0.7"/>
<path d="M{lx+LW-18} {ly+lh-24} h-14 M{lx+LW-18} {ly+lh-24} v-14" stroke="{c['accent']}" stroke-width="1.4" fill="none" opacity="0.7"/>
{pts_svg}
<text {_f(10)} x="{lx + 16}" y="{ly + lh - 10}" fill="{c['dim']}">PTS {n_pts} · FS/HILBERT-MORPH</text>

<!-- SYSTEM.INFO -->
<rect x="{RX}" y="{ly}" width="{RW}" height="{lh}" rx="7" fill="{c['panel']}" stroke="{c['panel_border']}"/>
<text {_f(12, ' letter-spacing="1.2"')} x="{RX + 18}" y="{ly + 22}" fill="{c['label']}" font-weight="700">SYSTEM.INFO</text>
<circle cx="{RX + RW - 172}" cy="{ly + 18}" r="4" fill="{c['live']}">
  <animate attributeName="opacity" values="1;0.25;1" dur="2.2s" repeatCount="indefinite"/></circle>
<text {_f(11)} x="{RX + RW - 162}" y="{ly + 22}" fill="{c['live']}" font-weight="700">LIVE</text>
<rect x="{RX + RW - 130}" y="{ly + 7}" width="112" height="22" rx="11" fill="{c['chip_bg']}"/>
<text {_f(11)} x="{RX + RW - 74}" y="{ly + 22}" text-anchor="middle" fill="{c['chip_fg']}" font-weight="700">{escape(HANDLE)}</text>
<line x1="{RX + 18}" y1="{ly + 34}" x2="{RX + RW - 18}" y2="{ly + 34}" stroke="{c['panel_border']}"/>
{chr(10).join(rows)}
<line x1="{RX + 18}" y1="{ly + lh - 28}" x2="{RX + RW - 18}" y2="{ly + lh - 28}" stroke="{c['panel_border']}"/>
<circle cx="{RX + 22}" cy="{ly + lh - 14}" r="3.5" fill="{c['ok']}" opacity="0">
  <animate attributeName="opacity" values="0;1" begin="{t_end:.2f}s" dur="0.3s" fill="freeze"/></circle>
<text {_f(10)} x="{RX + 32}" y="{ly + lh - 10}" fill="{c['ok']}" opacity="0">{escape(STATUS)}
  <animate attributeName="opacity" values="0;1" begin="{t_end:.2f}s" dur="0.3s" fill="freeze"/></text>
<text {_f(10)} x="{RX + RW - 18}" y="{ly + lh - 10}" text-anchor="end" fill="{c['dim']}">{escape(NODE)}</text>
</svg>
'''


if __name__ == "__main__":
    photo = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if photo and not photo.exists():
        sys.exit(f"photo not found: {photo}")
    out = Path(__file__).resolve().parent.parent / "assets"
    out.mkdir(parents=True, exist_ok=True)
    for name in THEMES:
        p = out / f"banner-{name}.svg"
        p.write_text(build(name, photo), encoding="utf-8")
        print(f"wrote {p}  ({p.stat().st_size/1024:.0f} KB)")
