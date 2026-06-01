# Installation

## Requirements

- **Python 3.10 or newer.** Older versions are not supported.
- **OpenCV** (>= 4.5.0). On modern Python this comes from the `opencv-python` wheel; no system OpenCV needed.
- A recent **scientific-Python stack** — NumPy, SciPy, pandas, matplotlib, scikit-image, scikit-learn, Pillow, seaborn, plotly. All pulled in automatically by `pip install`.

## Install from source

PyPI publication is on the roadmap. For now, install from the GitHub repo:

```bash
git clone https://github.com/aleciaco/AMFOrA_public.git
cd AMFOrA_public
pip install -e .
```

The `-e` flag installs the package in *editable mode* — changes to the source are picked up immediately without reinstalling. Drop the `-e` if you only want to use the package (not develop it).

## Recommended: use a fresh virtual environment

Create an isolated environment so AMFOrA's dependencies don't conflict with anything else on your system. With conda:

```bash
conda create -n amfora python=3.12 -y
conda activate amfora
pip install -e .
```

With `venv` (built into Python):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
pip install -e .
```

## Verify the install

```python
import amfora
print(amfora.__version__)
print(amfora.analyze_single_sherd)
```

If both lines run without an `ImportError`, you're set.

## Development install

If you intend to contribute (run the test suite, lint with ruff, build the docs locally), install the optional dev extras:

```bash
pip install -e ".[dev]"
```

That pulls in pytest, pytest-cov, ruff, jupyter-book, and sphinx-autodoc-typehints. See {doc}`contributing` for the development workflow.

## Troubleshooting installation

- **`ERROR: Package 'amfora' requires a different Python: 3.x not in '>=3.10'`** — your active interpreter is too old. Create a fresh env on Python 3.10+ as shown above.
- **`ModuleNotFoundError: No module named 'cv2'`** after install — the `opencv-python` wheel sometimes fails silently on unusual platforms. Try `pip install opencv-python` directly to see the real error.
- **`cv2.error: ... validateParameters`** on bright pastes — you're on a version of AMFOrA before the OpenCV 4.10+ compatibility fix. Update to v1.0.0 or later.

For runtime issues (no features detected, suspicious counts), see {doc}`troubleshooting`.
