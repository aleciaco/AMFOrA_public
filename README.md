# AMFOrA (Automated Macroscopic Fabric and Orientation Analysis)

A Python package for automated quantitative analysis of ceramic sherd fabric from flatbed-scanner images — inclusion detection, void analysis, size and orientation measurements, and color characterization.

Built for archaeological and materials-science workflows that need reproducible, DPI-aware measurements across many sherds.

## Features

- **Inclusion detection** — Blob and contour-based detectors with a paste-anchored MAD-scaled pop gate that handles dense clusters, fine grains on uniform paste, and mottled tempers consistently.
- **Void analysis** — Brightness-gated dark feature detection that separates pores from dark mineral inclusions.
- **Watershed cluster recovery** — Dense touching grains are split into individual sub-grains rather than being counted as one lumpy contour.
- **Size and density metrics** — Areas in cm², densities per cm², with the denominator computed over the actually-searched area (not the full sherd) so cross-sherd comparisons stay valid.
- **Orientation and color** — Per-inclusion orientation distributions and CIELAB color summaries.
- **Batch processing** — Run a whole folder of sherd scans into a single CSV via `full_analysis`.

## Installation

### From source

```bash
git clone https://github.com/aleciaco/AMFOrA_public.git
cd AMFOrA_public
pip install -e .
```

PyPI installation is planned for the next release.

### Dependencies

Python 3.10+, OpenCV, NumPy, pandas, matplotlib, SciPy, scikit-image, Pillow. All pulled in automatically by `pip install`.

## Quickstart

```python
import cv2
import amfora

# Load a raw scanned sherd image (background still present)
image = cv2.imread('sherd.jpg')

# One-call analysis: masking, blob + contour detection, color, orientation
results = amfora.analyze_single_sherd(image, scan_dpi=1200)

print(f"Blob inclusions: {results['blob_inclusion_count']}")
print(f"Contour inclusions: {results['contour_inclusion_count']}")
print(f"Inclusion density: {results['contour_inclusion_density_per_cm2']:.1f} per cm²")
```

For a batch of sherds in a folder:

```python
df = amfora.full_analysis('path/to/sherd_scans/', scan_dpi=1200)
df.to_csv('results.csv')
```

See `examples/` for working notebooks and `docs/` for the full user guide.

## Documentation

The Jupyter Book under `docs/` builds locally with `jupyter-book build docs/` and covers installation, quickstart, the user guide, full API reference, and tutorials. Online publishing via GitHub Pages is planned (link will go here once the site is live).

## Core entry points

- `analyze_single_sherd(image, scan_dpi)` — Mask + detect + measure a single sherd; returns a dict of metrics.
- `full_analysis(folder_path, scan_dpi)` — Run `analyze_single_sherd` over a directory; returns a pandas DataFrame and writes CSV.
- `sherd_mask(image, scan_dpi)` — GrabCut-based mask isolating the sherd from scanner background.
- `sherd_blobs(masked_image, scan_dpi)` — SimpleBlobDetector with adaptive thresholds and paste-anchored pop gating.
- `contour_detection(masked_image, scan_dpi)` — Contour-based detection with watershed cluster recovery and multigrain split.

Every function has a NumPy-style docstring with parameter and return-value details.

## DPI and units

All measurements are DPI-aware. The default is `scan_dpi=1200`. Conversion factor is `dpcm = scan_dpi * 0.3937`; areas are returned in cm². Set `scan_dpi` to match your scanner.

## Scientific applications

Designed for:

- Quantitative petrographic analysis of ceramic fabrics
- Inclusion size and shape distribution studies
- Void analysis (organic burnout, shrinkage cracks)
- Orientation analysis of elongated inclusions
- Color characterization (paste, inclusion populations, core-periphery firing-atmosphere indicators)

Cross-fabric inclusion-density comparisons are limited by optical contrast: grains that share the matrix's color are invisible to scanning regardless of detection parameters. See `analyze_single_sherd`'s docstring for the full discussion and the R08G/R08TC calibration data showing the ~2.3× density gap on bars sharing identical sand temper.

## Contributing

Pull requests welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR guidelines. By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Disclosure: use of generative AI tools

Parts of this package have been developed and refined with assistance from generative AI tools (Anthropic Claude, primarily for testing, repackaging in modern Python standards, docstring and documentation drafting, code review, and iterative tuning of detector parameters). All algorithmic design decisions, validation against real sherd data, and final acceptance of changes were performed by the human maintainer. As a solo archaeologist who knows Python, I had a glaring gap in my ability to actually format the github in the modern standard where it would be useful by other researchers, so I turned to Anthropic Claude to help format the documentation, make docstrings make enough sense for other users, write tests, and do linting/CI because I had never learned how to do that. If there are any issues with this content, please reach out or suggest better ways of maintaining it!

## License

MIT License — see [LICENSE](LICENSE).

## Citation

If you use AMFOrA in published work, please cite:

> Iacobucci, A. (2026). *AMFOrA: Automated Macroscopic Fabric and Orientation Analysis for ceramic sherds*. https://github.com/aleciaco/AMFOrA_public

A formal citation (DOI / Zenodo archive / JOSS paper) will be added when the package is released to PyPI.

## Support

- Bug reports and feature requests: [GitHub issues](https://github.com/aleciaco/AMFOrA_public/issues)
- Maintainer: [aleciaco@uw.edu](mailto:aleciaco@uw.edu)
