"""Synthetic ceramic-image generator for testing and validation.

Generates BGR images of "ceramic sherds" with non-overlapping inclusions and
voids, where the ground-truth count and per-feature metadata are known
exactly.  Shapes and colors are calibrated to fall inside the
``amfora.core.detection`` envelopes (size, aspect, solidity, compactness,
circularity / convexity, and intensity contrast), so any detector failure
on these images points at a real algorithmic issue rather than a synthetic-
image quirk.

This module is intentionally separate from the package's auto-import surface
(``amfora.__init__``) — pull it in explicitly with::

    from amfora.testing import generate_ceramic_image

The package's own pytest suite uses it via ``tests/conftest.py``; user
notebooks can use it for their own validation experiments.
"""

import cv2
import numpy as np
from numpy.random import default_rng


def _sample_ceramic_bg(rng):
    """Sample a plausible ceramic matrix color in LAB; return BGR uint8 tuple."""
    if rng.random() < 0.1:  # 10% chance for dark grey
        L = rng.integers(30, 70)
        a = rng.integers(120, 135)
        b = rng.integers(120, 135)
    else:
        L = rng.integers(120, 220)
        a = rng.integers(128, 165)
        b = rng.integers(135, 175)
    lab = np.array([[[L, a, b]]], dtype=np.uint8)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return tuple(int(c) for c in bgr[0, 0])


def _keep_largest_component(mask):
    """Reduce a binary mask to its single largest connected component."""
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_lbl <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


# --- Shape generators -------------------------------------------------------
# Calibrated against amfora.core.detection envelopes.
#
# Inclusions (blob + polygon): solidity >~0.85, compactness >~0.4, aspect <=1.5:1.
#   Comfortably inside the inclusion gates (0.45 / 0.125 / 4:1).
# Voids: sinuous traces with thickness >= the detector's blur kernel so
#   blackhat morphology can detect them as continuous contours rather than
#   breaking them into fragments.  The split between void and inclusion lists
#   rests on contour_detection's symmetric void_intensity_max brightness
#   exclusion.

def _blob_mask(rng, radius):
    """Round-ish inclusion: ellipse with mild axis ratio, optional rotation.

    Designed to pass blob and contour inclusion gates by construction.
    """
    side = 2 * radius + 4
    mask = np.zeros((side, side), dtype=np.uint8)
    cx, cy = side // 2, side // 2
    rx = radius
    ry = max(2, int(radius * rng.uniform(0.75, 1.0)))
    rot = float(rng.uniform(0, 180))
    cv2.ellipse(mask, (cx, cy), (rx, ry), rot, 0, 360, 255, -1)
    return mask


def _polygon_mask(rng, radius):
    """Angular inclusion: 4-8 sided polygon with modest vertex jitter.

    Jitter range chosen so aspect ratio stays under ~1.5:1 and solidity stays
    above ~0.85, well inside inclusion shape gates.
    """
    side = 2 * radius + 4
    mask = np.zeros((side, side), dtype=np.uint8)
    n_sides = int(rng.integers(4, 9))
    rot = rng.uniform(0, 2 * np.pi)
    cx, cy = side // 2, side // 2
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + rot
    jitter = rng.uniform(0.85, 1.15, size=n_sides)
    xs = (cx + radius * jitter * np.cos(angles)).astype(np.int32)
    ys = (cy + radius * jitter * np.sin(angles)).astype(np.int32)
    pts = np.stack([xs, ys], axis=1)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _void_mask(rng, radius):
    """Sinuous organic-burnout trace.

    Renders a thick curved polyline rather than a closed convex shape.
    Length factor 3.5-6.5x radius (min absolute 40 px so even small voids
    look elongated), thickness factor 0.35-0.55x radius (min absolute 10 px
    so the trace stays thicker than contour_detection's ~11 px Gaussian
    blur kernel at 1200 DPI — thinner traces get smoothed into the paste
    and the blackhat morphology fragments them, producing trace-piece
    inclusions that escape the void brightness gate).  Sinuous amplitude
    15-30% of length.  Visual L/T ~5-15.

    Void/inclusion split rests on contour_detection's symmetric
    void_intensity_max brightness exclusion, not on these shape proportions.
    """
    length = max(40, int(radius * rng.uniform(3.5, 6.5)))
    thickness = max(10, int(radius * rng.uniform(0.35, 0.55)))
    amplitude = int(length * rng.uniform(0.15, 0.30))

    n_pts = max(20, length // 4)
    xs = np.linspace(-length / 2, length / 2, n_pts)
    ys = np.zeros(n_pts)
    n_harm = int(rng.integers(2, 5))
    for _ in range(n_harm):
        f = rng.uniform(0.5, 2.0)
        a = amplitude / n_harm * rng.uniform(0.5, 1.5)
        p = rng.uniform(0, 2 * np.pi)
        ys += a * np.sin(2 * np.pi * f * xs / length + p)

    theta = rng.uniform(0, 2 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    xr = cos_t * xs - sin_t * ys
    yr = sin_t * xs + cos_t * ys

    pad = thickness + 4
    w_extent = int(xr.max() - xr.min()) + 2 * pad
    h_extent = int(yr.max() - yr.min()) + 2 * pad
    side = max(w_extent, h_extent)
    cx_off = side // 2 - int((xr.max() + xr.min()) / 2)
    cy_off = side // 2 - int((yr.max() + yr.min()) / 2)

    mask = np.zeros((side, side), dtype=np.uint8)
    pts = np.stack(
        [(xr + cx_off).astype(np.int32), (yr + cy_off).astype(np.int32)],
        axis=1,
    )
    cv2.polylines(mask, [pts], isClosed=False, color=255,
                  thickness=thickness, lineType=cv2.LINE_8)
    return _keep_largest_component(mask)


# --- Color sampling ---------------------------------------------------------
# Material palettes for non-void inclusions, in OpenCV-scaled LAB (a,b
# centered at 128). Higher a = redder, higher b = yellower.
#
# All ranges are calibrated so the MINIMUM per-channel BGR intensity stays
# above the detector's void_intensity_max=60 cutoff with margin (~10+).
# This is what stops dark inclusions from being mis-classified as voids by
# the symmetric brightness gate — on real sherds dark mineral inclusions
# sit at interior >=73 even on light-grey paste; the palettes match that
# distribution rather than producing "blacker than any real mineral" colors.
# Heavy a/b saturation redistributes brightness across channels (warm
# colors suppress B, cool colors suppress R), so saturation upper bounds
# are tightened in tandem with lower L bounds.
_INCLUSION_PALETTES = [
    ((210, 250), (118, 135), (118, 138)),  # white shell / limestone
    ((80, 115),  (124, 132), (124, 132)),  # dark mineral grain (basalt, sand)
    ((100, 150), (120, 134), (120, 134)),  # mid grey stone
    ((120, 170), (138, 155), (140, 155)),  # terracotta grog
    ((105, 140), (130, 140), (135, 145)),  # dark brown organics / iron-rich
]

# Inclusions need pop >= ~25 (max-BGR core-vs-annulus delta) to pass the
# paste-anchored pop gate.  Sample-resampling target uses a comfortable margin.
_MIN_INCLUSION_BG_DELTA = 35

# Voids are sampled as a darkened version of the matrix color (matrix a/b
# kept with mild jitter, L pulled well below the matrix and capped so
# every channel comfortably clears the void_intensity_max=60 cutoff).
# This matches the physical model: a void in a thick section reads as a
# pocket of darker paste-hued pixels, not a pure-black hole, because the
# back wall of the cavity is the same fabric color.
_VOID_L_DROP_RANGE = (80, 160)  # L_void = L_bg - uniform(80, 160)
_VOID_L_CAP = 35                # absolute cap (LAB scale) so BGR median <60
_VOID_L_FLOOR = 5
_VOID_AB_JITTER = 3             # small a/b wiggle around matrix hue


def _sample_inclusion_color(bg_bgr, kind, rng):
    """Sample BGR.  Voids = darkened matrix tint; inclusions resample if too close to bg."""
    if kind == "void":
        bg_arr = np.array([[bg_bgr]], dtype=np.uint8)
        bg_lab = cv2.cvtColor(bg_arr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.int32)
        L_bg, a_bg, b_bg = int(bg_lab[0]), int(bg_lab[1]), int(bg_lab[2])
        L_void = max(_VOID_L_FLOOR,
                     min(_VOID_L_CAP,
                         L_bg - int(rng.integers(_VOID_L_DROP_RANGE[0],
                                                 _VOID_L_DROP_RANGE[1] + 1))))
        a_void = int(np.clip(a_bg + rng.integers(-_VOID_AB_JITTER, _VOID_AB_JITTER + 1), 0, 255))
        b_void = int(np.clip(b_bg + rng.integers(-_VOID_AB_JITTER, _VOID_AB_JITTER + 1), 0, 255))
        lab = np.array([[[L_void, a_void, b_void]]], dtype=np.uint8)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return tuple(int(c) for c in bgr[0, 0])

    bg_arr = np.array(bg_bgr, dtype=np.int32)
    best = None
    best_delta = -1
    for _ in range(12):
        L_r, a_r, b_r = _INCLUSION_PALETTES[int(rng.integers(0, len(_INCLUSION_PALETTES)))]
        L = int(rng.integers(L_r[0], L_r[1] + 1))
        a = int(rng.integers(a_r[0], a_r[1] + 1))
        b = int(rng.integers(b_r[0], b_r[1] + 1))
        lab = np.array([[[L, a, b]]], dtype=np.uint8)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cand = tuple(int(c) for c in bgr[0, 0])
        delta = int(np.max(np.abs(np.array(cand, dtype=np.int32) - bg_arr)))
        if delta >= _MIN_INCLUSION_BG_DELTA:
            return cand
        if delta > best_delta:
            best_delta = delta
            best = cand
    # Fallback: if no palette draw cleared the threshold, push to whichever
    # high-contrast extreme (white or dark-grey) is farther from the matrix.
    # Note: "dark" fallback is dark grey (90), not pure black, so it stays
    # above void_intensity_max on every channel.
    bg_mean = int(bg_arr.mean())
    return (250, 250, 250) if bg_mean < 128 else (90, 90, 90)


# --- Kind partition (deterministic) ----------------------------------------
# Per-image kind mix is a hard split, not a probabilistic sample, so ground
# truth is exactly recoverable: a target of 100 features at the default
# weights produces exactly 60 blobs, 25 polygons, 15 voids regardless of
# seed.  Order of placement is still shuffled per seed so spatial layout
# varies normally.
KINDS = np.array(["blob", "polygon", "void"])
KIND_WEIGHTS = np.array([0.60, 0.25, 0.15])
_MAX_TRIES = 50


def _partition_kinds(n_total, weights):
    """Split n_total into integer per-kind counts that sum exactly to n_total.

    Uses largest-remainder rounding: floor each ``n_total * weight``, then
    distribute the leftover one-at-a-time to the kinds with the largest
    fractional parts.  Stable for any weight vector summing to 1.
    """
    raw = n_total * np.asarray(weights, dtype=float)
    counts = np.floor(raw).astype(int)
    leftover = n_total - int(counts.sum())
    if leftover > 0:
        frac = raw - counts
        order = np.argsort(-frac)
        for i in range(leftover):
            counts[order[i % len(order)]] += 1
    return counts


def generate_ceramic_image(image_size, n_inclusions, size_range, seed,
                            paste_noise_std=0):
    """Create a synthetic ceramic image with non-overlapping features.

    Per-image kind mix is deterministic (largest-remainder split of
    ``n_inclusions`` across ``KIND_WEIGHTS``), so 100 features at the
    default weights always yields exactly 60 blobs / 25 polygons /
    15 voids.  Spatial placement and per-feature appearance still vary
    by seed.

    Shapes and colors are calibrated to fall inside the
    ``amfora.core.detection`` envelopes (size, aspect, solidity,
    compactness, circularity/convexity, and intensity contrast).  Voids
    are rendered as sinuous traces (organic-burnout signatures) with a
    darkened matrix tint and rely on the detector's symmetric
    ``void_intensity_max`` brightness exclusion to avoid being double-
    counted as inclusions.

    Parameters
    ----------
    image_size : tuple of int
        ``(H, W)`` in pixels.
    n_inclusions : int
        Total target feature count (split into kinds per ``KIND_WEIGHTS``).
    size_range : tuple of int
        ``(min_radius, max_radius)`` in pixels.  Used directly for blob /
        polygon inclusions; for voids the same value acts as a length-
        scale factor (the trace is 3.5-6.5x longer than this, clamped to
        a 40 px minimum, with thickness 0.35-0.55x clamped to a 10 px
        minimum so the trace stays thicker than ``contour_detection``'s
        11 px blur kernel at 1200 DPI).
    seed : int
        RNG seed for reproducibility.
    paste_noise_std : float, optional
        Standard deviation of per-pixel Gaussian noise added to paste
        pixels before features are placed (default: 0, no noise).
        Real flatbed scans show paste MAD of ~5-18 depending on the
        clay body.  For Gaussian noise the relationship is
        σ ≈ 1.4826 · MAD, so ``paste_noise_std=10`` yields MAD ~7
        (mid-range real sherd) and ``paste_noise_std=18`` yields
        MAD ~12 (heavily-mottled grog-tempered fabric).  Either value
        exercises the K * MAD branch of the paste-anchored pop gate.
        Leaving the default of 0 produces a zero-MAD paste, which
        instead exercises the ``paste_pop_floor`` fallback branch —
        both are useful regimes to test.

    Returns
    -------
    image_bgr : (H, W, 3) uint8 ndarray
    label_mask : (H, W) int32 ndarray
        Per-pixel feature ID (0 = background).  Useful for IoU evaluation.
    metadata : dict
        Carries ``inclusions`` (per-feature records),
        ``kind_target_counts`` (the deterministic split for this run),
        per-kind ``kind_placed_counts``, ``skipped``, and the
        ``paste_noise_std`` setting used.
    """
    H, W = image_size
    rng = default_rng(seed)
    bg = _sample_ceramic_bg(rng)
    image = np.full((H, W, 3), bg, dtype=np.uint8)
    if paste_noise_std > 0:
        # Per-channel Gaussian noise on paste pixels.  Applied BEFORE features
        # are placed so inclusion / void pixels stay at their sampled colors
        # (matching the real-scan model where the sensor's noise floor sits
        # on the paste, not on dark mineral grain interiors).
        noise = rng.normal(0, paste_noise_std, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    label_mask = np.zeros((H, W), dtype=np.int32)
    inclusions = []
    skipped = 0
    rmin, rmax = size_range

    counts = _partition_kinds(n_inclusions, KIND_WEIGHTS)
    kinds_sequence = np.repeat(KINDS, counts).tolist()
    rng.shuffle(kinds_sequence)
    placed_counts = {k: 0 for k in KINDS.tolist()}

    for i, kind in enumerate(kinds_sequence, start=1):
        radius = int(rng.integers(rmin, rmax + 1))
        if kind == "blob":
            shape = _blob_mask(rng, radius)
        elif kind == "polygon":
            shape = _polygon_mask(rng, radius)
        else:
            shape = _void_mask(rng, radius)
        sh, sw = shape.shape
        if sh >= H or sw >= W or shape.sum() == 0:
            skipped += 1
            continue
        color = _sample_inclusion_color(bg, kind, rng)

        placed = False
        for _ in range(_MAX_TRIES):
            x0 = int(rng.integers(0, W - sw))
            y0 = int(rng.integers(0, H - sh))
            region = label_mask[y0:y0 + sh, x0:x0 + sw]
            if np.any((shape > 0) & (region > 0)):
                continue
            ys, xs = np.where(shape > 0)
            image[y0 + ys, x0 + xs] = color
            label_mask[y0 + ys, x0 + xs] = i
            cx = x0 + int(xs.mean())
            cy = y0 + int(ys.mean())
            inclusions.append({
                "id": i,
                "kind": kind,
                "center": (cx, cy),
                "radius": radius,
                "color_bgr": color,
            })
            placed_counts[kind] += 1
            placed = True
            break
        if not placed:
            skipped += 1

    metadata = {
        "seed": seed,
        "image_size": (H, W),
        "background_bgr": bg,
        "inclusions": inclusions,
        "skipped": skipped,
        "kind_target_counts": dict(zip(KINDS.tolist(), counts.tolist())),
        "kind_placed_counts": placed_counts,
        "paste_noise_std": paste_noise_std,
    }
    return image, label_mask, metadata


def generate_ceramic_image_batch(n_images, seeds, image_size, n_inclusions,
                                  size_range, paste_noise_std=0):
    """Run ``generate_ceramic_image`` once per seed.  Requires ``len(seeds) == n_images``."""
    if len(seeds) != n_images:
        raise ValueError(f"len(seeds)={len(seeds)} does not match n_images={n_images}")
    return [
        generate_ceramic_image(image_size, n_inclusions, size_range, s,
                                paste_noise_std=paste_noise_std)
        for s in seeds
    ]
