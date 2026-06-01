# Contributing

The canonical contributing guide lives in the repo root: [`CONTRIBUTING.md`](https://github.com/aleciaco/AMFOrA_public/blob/main/CONTRIBUTING.md). It covers:

- Development setup (`pip install -e ".[dev]"`)
- How to run the test suite (`pytest`)
- Code style (ruff)
- Pull-request workflow
- Bug-report template
- Disclosure of generative-AI assistance

By participating you agree to the [Code of Conduct](https://github.com/aleciaco/AMFOrA_public/blob/main/CODE_OF_CONDUCT.md). Report unacceptable behavior to [aleciaco@uw.edu](mailto:aleciaco@uw.edu).

## Where to start

- **Documentation bugs** — typo fixes, clarifications, examples that drifted out of sync with the code. These are the most welcome PRs and the easiest to review.
- **Test fixtures** — additional synthetic-image scenarios in `tests/conftest.py` to exercise edge cases.
- **New detection parameters** — if you've calibrated a new parameter on a specific fabric type, open an issue describing the calibration data first; we'll discuss before code.
- **Performance** — if you have a profile showing a hot spot, that's a great starting point for an optimization PR.

If you're not sure where to begin, open a discussion or email the maintainer.
