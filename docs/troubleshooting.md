# Troubleshooting

Common failure modes with concrete fixes. If you hit something not covered here, please open an issue: [github.com/aleciaco/AMFOrA_public/issues](https://github.com/aleciaco/AMFOrA_public/issues).

## Installation problems

### `Package 'amfora' requires a different Python: 3.x not in '>=3.10'`

Your active interpreter is too old. AMFOrA requires Python 3.10+. Create a fresh environment:

```bash
conda create -n amfora python=3.12 -y
conda activate amfora
pip install -e .
```

### `ModuleNotFoundError: No module named 'cv2'`

The `opencv-python` wheel didn't install. On unusual platforms try the headless variant:

```bash
pip uninstall -y opencv-python
pip install opencv-python-headless
```

### `ModuleNotFoundError` for `seaborn`, `sklearn`, or `plotly` after `pip install`

You're on an older version of AMFOrA. v1.0.0 declared these as required dependencies. Update:

```bash
git pull
pip install -e .
```

## Detection problems

### No features detected on a sherd that visibly has them

Most common cause: **`scan_dpi` doesn't match your actual scan resolution**. AMFOrA's size filters and blur kernels scale from `scan_dpi`; a wrong value can put real grains below the minimum-size threshold.

```python
# Sanity check
print(image.shape)           # (H, W, 3) — if your scan is supposed to be
                             # 1 inch wide at 1200 dpi, H or W should be ~1200
dpcm = scan_dpi * 0.3937     # dots per cm; ~472 at 1200 DPI
```

Second most common: **`paste_pop_k` too high**. Try lowering to 1.5 for both detectors and see whether the missing grains return:

```python
amfora.contour_detection(masked, scan_dpi=1200, paste_pop_k=1.5)
amfora.sherd_blobs(masked, scan_dpi=1200, paste_pop_k=1.5)
```

If grains come back at low K but the rest of the sherd doesn't look like noise, you can leave K low for that fabric type.

### Too many false detections on a uniform paste

Symptoms: tens of small inclusions reported on a sherd you know has no temper (e.g. R01-series control bars).

Most common cause: **unmasked overhang at the sherd edge**. A broken corner or weathered edge that GrabCut couldn't fully exclude reads as a dark splotch. AMFOrA's edge-band gate (4 % of the shorter image dimension, default) covers this — verify it's not disabled:

```python
# These are the defaults; pass explicitly to be sure
amfora.contour_detection(masked, scan_dpi=1200)   # edge_band default = 4 %
amfora.sherd_blobs(masked, scan_dpi=1200)         # edge_band default = 4 %
```

If the false positives are mid-paste rather than at edges, they're likely real surface micro-features (chips, dust, micro-cracks) rather than inclusions. Raising `paste_pop_k` to 3.0 will suppress them but may also drop genuine fine grains.

### "Voids" that look like dark mineral inclusions in the output

The void-vs-inclusion split is brightness-based: features with median interior intensity below `void_intensity_max` (default 60) get classified as voids. Dark mineral grains in light paste can fall just under this threshold.

- Lower `void_intensity_max` (e.g. 45) to make voids stricter.
- The `analyze_inclusion_angularity` metrics may help you distinguish populations after the fact.

### Big lumpy contours that should be many individual grains

You probably have an old version. v1.0.0 introduced `multigrain_split_enabled` (default `True`) which post-processes lumpy contours via distance-transform watershed. Confirm it's running:

```python
cr = amfora.contour_detection(masked, scan_dpi=1200, debug_mode=True)
# Look for: "inclusion_added_by_multigrain_split: <nonzero>"
```

If you genuinely want the legacy behavior (one contour per visible cluster), pass `multigrain_split_enabled=False`.

## Area / density looks wrong

### Density much higher than expected

You're probably comparing against historical numbers computed with the **full sherd area** as denominator. AMFOrA v1.0.0 switched to the **effective detection area** (sherd minus edge band) as the denominator for densities and area-percentages, so the results reflect the area the detectors actually searched.

```python
r = amfora.analyze_single_sherd(image, scan_dpi=1200)
print(r["sherd_area_cm2"])                    # full mask area, for reference
print(r["effective_detection_area_cm2"])      # denominator for densities
```

Multiply density by `effective_detection_area_cm2 / sherd_area_cm2` if you need the legacy convention.

### Area in pixels, not cm²

You called `cv2.contourArea` directly on a contour from `contour_detection`. Those areas are in pixels. Use the `inclusion_areas` and `void_areas` keys in the returned dict — those are already in cm² (divided by `(scan_dpi · 0.3937)²`).

## Performance

### Single-sherd analysis takes >30 s

Likely causes:

- **Image dimensions are huge** (e.g. a full uncropped flatbed scan). `sherd_mask` does its own crop, but if you pass a 10000×10000 image, GrabCut still has to chew through it. Pre-crop to roughly the sherd's bounding box first.
- **`analyze_core_periphery=True`** (the default in `analyze_single_sherd`). This stage is computationally expensive. Set `analyze_core_periphery=False` if you don't need firing-atmosphere metrics.
- **Watershed split firing on many large contours**. Pass `multigrain_split_enabled=False` to skip it (you'll get fewer, lumpier contours).

### Memory errors on a folder of sherds

`full_analysis` keeps all results in memory until it writes the CSV at the end. For thousands of sherds, process in batches and append to a CSV:

```python
import pandas as pd

paths = [...]  # all your image paths
for chunk in [paths[i:i+50] for i in range(0, len(paths), 50)]:
    # save each chunk to its own folder, run full_analysis, append
    ...
```

## Getting more help

If you're still stuck, the most useful information for a bug report is:

1. **Python and AMFOrA versions:** `python --version` and `python -c "import amfora; print(amfora.__version__)"`
2. **OS and OpenCV version:** `python -c "import cv2; print(cv2.__version__)"`
3. **A minimal reproducer:** one sherd image plus the exact call you made.
4. **What you expected vs what happened:** counts, traceback, etc.

Open an issue with that info at [github.com/aleciaco/AMFOrA_public/issues](https://github.com/aleciaco/AMFOrA_public/issues).
