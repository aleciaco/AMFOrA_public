# AMFOrA

**Automated Macroscopic Fabric and Orientation Analysis**

AMFOrA is a Python package for automated, reproducible quantitative analysis of ceramic sherd fabric from flatbed-scanner images. It detects inclusions and voids, measures their sizes and orientations, characterizes paste and inclusion color, and aggregates per-sherd results into ready-to-publish CSVs.

The package targets archaeological and materials-science workflows where consistent, DPI-aware measurements across many sherds matter — petrographic surveys, temper provenance studies, firing-atmosphere reconstruction.

## When to use AMFOrA

- You have flatbed scans of ceramic sherds (typically 600–1200 DPI cross-sections).
- You want quantitative measurements (inclusion counts, size distributions, void densities) rather than visual classification alone.
- You need to process a batch of sherds with the same parameters and write the results to a single table.

## What's in the box

- **`analyze_single_sherd`** — masking + blob and contour detection + size, color, orientation analysis for one sherd; returns a dict.
- **`full_analysis`** — runs the above over a directory of images and returns a pandas DataFrame.
- **`sherd_mask` / `sherd_blobs` / `contour_detection`** — the building blocks if you want to skip the all-in-one entry points.
- **`amfora.testing.generate_ceramic_image`** — a deterministic synthetic-sherd generator used by the test suite; also useful for validating your own detection parameters against known ground truth.

## A note on scope and limits

AMFOrA finds features by their optical contrast against the surrounding paste. Grains whose color overlaps the matrix (e.g. iron-bearing sand in a terracotta paste) are invisible to scanning regardless of detection parameters. Reported inclusion counts represent an **optically-visible lower bound** on the true grain population, not a complete census. See {doc}`api-reference/analysis` (the `analyze_single_sherd` docstring) for the full discussion and the R08G/R08TC calibration data showing the ~2.3× density gap on bars sharing identical sand temper.

## Documentation map

- {doc}`installation` — install via `pip install -e .` (PyPI release coming).
- {doc}`quickstart` — five-line example that mirrors the test suite.
- {doc}`user-guide/basic-workflow` — the full pipeline from raw scan to CSV, with parameter notes.
- {doc}`api-reference/index` — auto-generated reference for every public function.
- {doc}`troubleshooting` — common failure modes and fixes.
- {doc}`contributing` — development setup and PR guidelines.
- {doc}`changelog` — release notes.

## License and citation

MIT licensed. If you use AMFOrA in published work, please cite:

> Iacobucci, A. (2026). *AMFOrA: Automated Macroscopic Fabric and Orientation Analysis for ceramic sherds*. https://github.com/aleciaco/AMFOrA_public

A DOI / Zenodo archive will be added with the first PyPI release.
