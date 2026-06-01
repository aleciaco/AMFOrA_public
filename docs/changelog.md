# Changelog

All notable changes to AMFOrA are documented here. The format roughly follows [Keep a Changelog](https://keepachangelog.com/), and versions follow [semantic versioning](https://semver.org/).

## v1.0.1 update Docs

Docs still had instructions to clone the github and install -e instead of pointing to the pypi pip install amfora. This should fix that issue.

## v1.0.0 — Unreleased

First release. Package renamed from the original `AMFOrA_public` working name to `amfora` for PyPI publication.

### Detection pipeline

- Replaced the local-ring inclusion pop gate with a **paste-anchored MAD-scaled** version. The threshold per channel is `max(K · MAD, floor)`, where MAD is the per-channel Median Absolute Deviation of paste pixels. The same `K` works across paste types because MAD tracks the paste's own noise floor.
- Added **watershed cluster recovery** for dark connected components too big for the size cap.
- Added a **multigrain split** pass that re-splits lumpy accepted contours into individual grain sub-contours via distance-transform watershed seeded by local-maxima.
- Added an **edge-band gate** to both detectors (4 % of shorter image dimension by default). Covers CLAHE tile-boundary leakage and unmasked-overhang artifacts.
- Added **`effective_detection_area_cm2`** to `analyze_single_sherd` results. Density and area-percentage metrics now divide by this instead of `sherd_area_cm2`, so they reflect the area the detectors actually searched.

### Synthetic image generator

- New `amfora.testing` module exposing the calibrated synthetic-sherd generator (previously inline in `examples/pro_gen_images.ipynb`). Adds a `paste_noise_std` parameter so the same image can exercise either the `K · MAD` or the `paste_pop_floor` branch of the pop gate.

### Packaging

- Migrated to modern `src/amfora` layout with `pyproject.toml` (hatchling backend). `setup.py` and `requirements.txt` removed.
- Python 3.10+ required.
- All previously-undeclared dependencies (seaborn, scikit-learn, plotly) added to the install requirements.
- Optional extras: `[test]`, `[dev]`, `[docs]`.

### Tests and CI

- Added `tests/` with 17 pytest tests covering imports, paste reference / MAD, false-positive rejection, recall on synthetic ground truth, and analysis result shape.
- Added `.github/workflows/test.yml` (pytest on Py 3.10/3.11/3.12 + macOS Py 3.12, coverage upload).
- Added `.github/workflows/lint.yml` (ruff check + format).

### Bugs fixed during test-suite buildout

- **OpenCV 4.10+ blob param validation crash on bright pastes.** The hardcoded `min(200, max_thresh)` cap pushed `max < min` on cream / white channels, which OpenCV 4.10+ rejects. Cap raised to 255 with `max > min` enforced.
- **Cross-channel `_drop_nested` over-firing.** The per-channel-pool dedup used centroid-in-polygon, which dropped adjacent features when one's centroid fell inside another's outline. Replaced with bbox-IoU + asymmetric containment, which correctly distinguishes same-feature duplicates from adjacent features.
- **`cv2.erode` borderValue default.** The edge-band erosion was a silent no-op on all-foreground masks (pre-masked / synthetic inputs) because OpenCV's default border handling preserved image-edge pixels. Both call sites now pass `borderType=BORDER_CONSTANT, borderValue=0` explicitly.
- **Docstring escape-sequence deprecation warnings under Python 3.12.** Stripped `\*` and `\_` backslash escapes from docstrings.

### Documentation

- Repo-root files: `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), generative-AI disclosure.
- Jupyter Book under `docs/` rewritten to reflect the current API; auto-published via `.github/workflows/docs.yml`.
- API reference auto-generated from NumPy-style docstrings via Sphinx autodoc.

---

## Prior history

The pre-1.0 work was developed under the names AMACFA+ and AMFOrA_public. That history is preserved in the git log; see [github.com/aleciaco/AMFOrA_public/commits/main](https://github.com/aleciaco/AMFOrA_public/commits/main).
