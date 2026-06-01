# API reference

The API is split across four modules under `amfora.core`. Every public function is documented here with the same NumPy-style docstrings that appear in the source. If you spot stale or unclear documentation, please open an issue (see {doc}`../contributing`).

## Module layout

| Module | Purpose | Highlights |
|---|---|---|
| {doc}`detection` | Image preprocessing, masking, feature detection | `sherd_mask`, `apply_mask`, `sherd_blobs`, `contour_detection`, the paste-anchored pop gate helpers |
| {doc}`analysis` | Per-sherd and batch metric calculation | `analyze_single_sherd`, `full_analysis`, `size_count_summary_single` |
| {doc}`visualization` | Inspection / plotting helpers | `inclusion_viewer` |
| {doc}`statistics` | Cross-sherd statistical summaries | clustering, PCA, group comparison helpers |

## Top-level convenience imports

All headline functions are re-exported at the package root, so you can write `amfora.contour_detection(...)` instead of `amfora.core.detection.contour_detection(...)`:

```python
import amfora

# All of these work and refer to the same function objects:
amfora.sherd_mask
amfora.sherd_blobs
amfora.contour_detection
amfora.analyze_single_sherd
amfora.full_analysis
```

The full re-export list lives in [`src/amfora/__init__.py`](https://github.com/aleciaco/AMFOrA_public/blob/main/src/amfora/__init__.py).

## Reading the docstrings

Each function's docstring includes:

- A one-line summary.
- A longer description of *what it does and why*, including algorithmic notes.
- A `Parameters` section with type, default, and explanation for every argument.
- A `Returns` section describing the return shape and meaning.
- Often a `Notes` or `Limitations` section with calibration data or known failure modes.

When in doubt about a parameter, the docstring is the source of truth — it's regenerated into these pages on every documentation build.

## Synthetic test helper

In addition to the four core modules, the package exposes `amfora.testing`, a deterministic synthetic-sherd generator used by the test suite. It's useful for validating detection parameters against known ground truth:

```python
from amfora.testing import generate_ceramic_image

image, label_mask, metadata = generate_ceramic_image(
    image_size=(700, 700),
    n_inclusions=50,
    size_range=(6, 14),
    seed=42,
    paste_noise_std=10,   # realistic paste noise
)

print(metadata["kind_placed_counts"])  # exact ground-truth counts
```

See `src/amfora/testing.py` for the full signature and parameter documentation.
