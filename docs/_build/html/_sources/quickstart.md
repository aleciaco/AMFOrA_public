# Quick start

This page gets you from "AMFOrA installed" to "first per-sherd results" in under a minute. It mirrors the example notebooks under `examples/`.

## Prerequisites

- AMFOrA installed (see {doc}`installation`).
- One ceramic sherd scan in JPEG, PNG, or TIFF format. A flatbed scan at 1200 DPI is the calibration default; lower DPI works if you set `scan_dpi` to match.

## The two-line version

```python
import cv2, amfora

image = cv2.imread("sherd.jpg")
result = amfora.analyze_single_sherd(image, scan_dpi=1200)
```

`result` is a dict containing inclusion / void counts, areas in cm², densities per cm², orientation summaries, color summaries, and the analysis status. The relevant keys for a typical workflow:

```python
print(f"Sherd area:           {result['sherd_area_cm2']:.2f} cm²")
print(f"Effective area:       {result['effective_detection_area_cm2']:.2f} cm²")
print(f"Inclusions (blob):    {result['blob_inclusion_count']}")
print(f"Inclusions (contour): {result['contour_inclusion_count']}")
print(f"Voids (contour):      {result['contour_void_count']}")
print(f"Inclusion density:    {result['contour_inclusion_density_per_cm2']:.1f} per cm²")
```

> **Why two inclusion counts?** AMFOrA runs two complementary detectors. The blob detector (`cv2.SimpleBlobDetector`) is good at round, compact grains; the contour detector handles angular, elongated, or clustered grains better. The numbers are usually within 10–20 % of each other; large discrepancies are diagnostic — see {doc}`troubleshooting`.

## Process a folder of sherds

```python
import amfora

df = amfora.full_analysis(
    "path/to/sherd_scans/",
    scan_dpi=1200,
    save_csv=True,           # also writes results.csv
)

# df is a pandas DataFrame with one row per sherd
df.head()
```

`full_analysis` is the same pipeline as `analyze_single_sherd` applied to every image in the folder, with a few aggregate columns added (filename, file path, scan_dpi). The CSV is suitable for downstream analysis in R, Python, or Excel.

## Visualize one sherd's detections

```python
import cv2, matplotlib.pyplot as plt
import amfora
from amfora.core.detection import sherd_mask, apply_mask

image = cv2.imread("sherd.jpg")

# Same pipeline analyze_single_sherd uses internally
mask, crop, _ = sherd_mask(image, scan_dpi=1200)
masked = apply_mask(image, mask, crop)

cr = amfora.contour_detection(masked, scan_dpi=1200)

# Overlay green inclusions and red voids on the sherd
overlay = masked.copy()
cv2.drawContours(overlay, cr["inclusions"], -1, (0, 255, 0), -1)
cv2.drawContours(overlay, cr["voids"], -1, (0, 0, 255), -1)

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"inclusions={cr['total_inclusions']}  voids={cr['total_voids']}")
plt.axis("off")
plt.show()
```

## Important parameters at a glance

| Parameter | Default | What it controls |
|---|---|---|
| `scan_dpi` | 1200 | Pixel-to-cm conversion. Set this to match your scanner — wrong values produce wrong areas. |
| `paste_pop_k` | 2.0 (blob), 2.5 (contour) | How far a candidate must stand out from paste, in MAD-multiples. Raise for tighter precision, lower for more recall. |
| `paste_pop_floor` | 8.0 | Absolute brightness floor under the MAD-scaled threshold. Protects against zero-MAD synthetic inputs. |
| `watershed_enabled` | `True` | Recover merged dark-cluster grains by distance-transform watershed (contour detector only). |
| `multigrain_split_enabled` | `True` | Re-split lumpy accepted contours into individual grains (contour detector only). |
| `void_intensity_max` | 60 | Maximum interior intensity (0–255) for a feature to qualify as a void. |

See the {doc}`api-reference/index` for the full parameter list with rationale.

## Next steps

- {doc}`user-guide/basic-workflow` — the same pipeline broken into individual stages, with tuning notes for each.
- {doc}`troubleshooting` — what to do when counts look off.
- {doc}`api-reference/index` — every public function with full docstring.
