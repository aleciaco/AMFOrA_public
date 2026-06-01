# User guide: the pipeline, stage by stage

`amfora.analyze_single_sherd` is the one-call interface, but understanding what it does internally helps when you need to tune parameters or debug unexpected results. This page walks through the same pipeline broken into its individual stages, with tuning notes for each.

## The full pipeline

For a single sherd, `analyze_single_sherd` runs roughly this sequence:

```python
import cv2, amfora
from amfora.core.detection import (
    sherd_mask, apply_mask, full_image_mask,
    sherd_blobs, contour_detection,
)

image = cv2.imread("sherd.jpg")

# 1. Segment the sherd from the scanner background
mask, crop, sherd_contour = sherd_mask(image, scan_dpi=1200)

# 2. Apply the mask (crop + bitwise_and against the mask)
masked = apply_mask(image, mask, crop)

# 3. Detect features two complementary ways
inc_blobs, void_blobs = sherd_blobs(masked, scan_dpi=1200)
contour_result        = contour_detection(masked, scan_dpi=1200)

# 4. Compute per-sherd statistics
#    (analyze_single_sherd does this part for you)
```

The rest of this page covers each stage in more detail.

## Stage 1: Masking

`sherd_mask` segments the sherd from the scanner background using GrabCut, then crops to the sherd's bounding box (plus a configurable buffer). Returns a 3-channel mask, the crop bounds, and the largest contour around the sherd outline.

```python
mask, crop, sherd_contour = sherd_mask(
    image,
    scan_dpi=1200,
    crop_buffer=125,   # px of padding around the sherd
    auto_crop=True,    # set False to keep original image dimensions
)
```

**Common issue:** the mask "eats" part of the sherd or leaves an overhanging fragment from a broken edge. The downstream **edge-band gate** (4 % of the shorter image dimension, default) is the protection layer — overhanging dark splotches near the mask boundary get rejected automatically. See {doc}`../troubleshooting`.

**Skip masking** when you already have a pre-cropped, background-removed image: pass `pre_masked=True` to `analyze_single_sherd` (it uses `full_image_mask` instead).

## Stage 2: Apply the mask

`apply_mask(image, mask, crop)` slices the image to the crop region, pads if needed, and zeros out non-sherd pixels via `bitwise_and`. The result is the input to both detectors.

```python
masked = apply_mask(image, mask, crop)
```

The masked image has the sherd's BGR values inside the mask and exact zeros everywhere else. The downstream detectors and the paste-anchored gate rely on this zero-background convention.

## Stage 3a: Blob detection

`sherd_blobs` runs three internal `cv2.SimpleBlobDetector` instances (light inclusions, dark inclusions, dark voids), one for each native BGR channel by default, then pools results. Good for round / compact grains; less good for angular ones.

```python
inc_blobs, void_blobs = amfora.sherd_blobs(
    masked,
    scan_dpi=1200,
    paste_pop_k=2.0,        # default for blobs (looser than contour)
    paste_pop_floor=8.0,
    edge_band_px=None,      # None → 4% of shorter dim
)
```

Returned blobs are `cv2.KeyPoint` objects. `kp.pt` is the (x, y) center; `kp.size` is the diameter in pixels.

## Stage 3b: Contour detection

`contour_detection` thresholds each channel into dark/light masks, traces contours with `cv2.findContours(RETR_TREE)`, filters by size, shape (aspect ratio, solidity, compactness), edge-band rejection, and the paste-anchored pop gate. Then runs two watershed passes: cluster recovery (oversize merged blobs) and multigrain split (lumpy accepted contours).

```python
cr = amfora.contour_detection(
    masked,
    scan_dpi=1200,
    paste_pop_k=2.5,                  # tighter than blob default
    paste_pop_floor=8.0,
    watershed_enabled=True,           # cluster recovery
    multigrain_split_enabled=True,    # split lumpy contours
)
```

Returns a dict with `inclusions` (list of contour arrays), `voids`, per-feature `inclusion_areas` and `void_areas` in cm², total counts, and a `debug_info` dict with paste reference, MAD, and per-stage rejection counters.

## Stage 4: Per-sherd statistics

`analyze_single_sherd` wraps everything above and computes:

- `sherd_area_cm2` — full mask area in cm².
- `effective_detection_area_cm2` — sherd area minus the 4 % edge band. This is what density / area-percentage metrics divide by, so cross-sherd comparisons stay valid.
- `*_inclusion_count`, `*_inclusion_density_per_cm2`, `*_inclusion_area_percentage` for both detectors.
- Same for voids.
- Inclusion orientation summary, color summary, sherd color summary.
- Optional core-periphery firing-atmosphere analysis (slow; off by default in tests).

## Key parameters and when to tune them

### `scan_dpi`

The single most important parameter. Must match your actual scan resolution or every cm² value will be wrong. AMFOrA's defaults (filter sizes, blur kernels, edge band) are calibrated at 1200 DPI but auto-scale with `scan_dpi`.

### `paste_pop_k` and `paste_pop_floor`

These set the threshold a candidate has to clear to count as a feature. The threshold per channel is `max(K · MAD, floor)`, where MAD is the per-channel Median Absolute Deviation of the paste pixels.

- Defaults work for most pottery (cream, terracotta, grey-bodied wares).
- **Raise K** (e.g. 3.0–3.5) when you're getting too many noise detections on visually-uniform paste.
- **Lower K** (e.g. 1.5–2.0) when you're missing visibly-present fine grains.
- The **floor** rarely needs tuning — it's a safety net for synthetic / perfectly-smooth inputs where MAD collapses to zero.

### `watershed_enabled` and `multigrain_split_enabled`

On by default. The watershed cluster-recovery and multigrain-split passes are what handle the dense-inclusion case (e.g. heavily-tempered sherds where adjacent grains touch). Turn off to get legacy single-contour behavior on cluster regions.

### `void_intensity_max`

Maximum interior intensity (0–255) for a feature to count as a void. Lower (e.g. 45) for stricter void detection; raise (e.g. 90) for low-contrast scans where pores don't quite reach near-black.

## Reading the debug output

If you pass `debug_mode=True` to `contour_detection`, you get a printed breakdown of how many candidates each filter rejected:

```python
amfora.contour_detection(masked, scan_dpi=1200, debug_mode=True)
```

```
[contour_detection debug]
  Channels                           : B, G, R
  Combine mode                       : union
  Size-filtered inclusion candidates : 248
  Accepted inclusions                : 138
  Rejected – boundary band (40 px)   : 12
  Rejected – solidity < 0.45         : 0
  Rejected – compactness < 0.12      : 0
  Rejected – nested in larger contour: 35
  ...
```

This is the fastest way to figure out *why* a candidate didn't make it through.

## Where to go next

- {doc}`../api-reference/index` — full parameter documentation for every public function.
- {doc}`../troubleshooting` — common failure modes (no features, too many false positives, void-vs-inclusion confusion).
