# Contributing to AMFOrA

Thanks for taking the time to contribute! This document covers how to set up a development environment, run the test suite, and submit changes.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to uphold its terms. Report unacceptable behavior to [aleciaco@uw.edu](mailto:aleciaco@uw.edu).

## Ways to contribute

- **Report a bug.** Open an issue with a minimal reproducer (sherd image + the call you made + what you expected vs what you got).
- **Suggest a feature.** Open an issue describing the use case. Discussion before code keeps reviews fast.
- **Submit a pull request.** Fork, branch, commit, push, open a PR against `main`. See below for setup.
- **Improve documentation.** Doc-only PRs are very welcome — typo fixes, clearer examples, better cross-links.

## Development setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/aleciaco/AMFOrA_public.git
cd AMFOrA_public
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### 2. Install in editable mode with dev dependencies

```bash
pip install -e ".[dev]"
```

(After Phase 2 of the modernization plan lands; until then, use `pip install -e .` and install dev tools manually.)

### 3. Verify the install

```python
python -c "import AMFOrA_public; print(AMFOrA_public.__version__)"
```

## Running tests

```bash
pytest
```

Tests live under `tests/` and use small fixture images checked into `tests/fixtures/`. Adding a new detection feature? Add a fixture image and a test asserting the expected behavior.

## Code style

- Follow [PEP 8](https://peps.python.org/pep-0008/) — 4-space indents, snake_case for functions, ALL_CAPS for module-level constants.
- Run `ruff check .` and `ruff format .` before committing.
- Docstrings use [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html). The existing detection / analysis modules are the reference.
- Prefer narrow, scoped changes over broad reworks. If you find yourself touching unrelated code, split the PR.

## Submitting a pull request

1. **Fork** the repo on GitHub.
2. **Branch** from `main` with a descriptive name: `git checkout -b fix/edge-band-overhang` or `feat/watershed-tuning`.
3. **Commit** with clear messages — describe *why* the change is needed, not just *what* changed.
4. **Test** locally: `pytest` should pass.
5. **Push** and open a PR against `main`. Link any related issues.
6. CI will run automatically. Address any failures before requesting review.
7. A maintainer will review. Be prepared for feedback — iteration is normal.

## Reporting bugs

Good bug reports include:

- The version of AMFOrA you're running (`AMFOrA_public.__version__`)
- The Python version (`python --version`) and OS
- A minimal sherd image that reproduces the issue (if image-dependent)
- The exact code you ran
- What you expected vs what actually happened
- Any traceback

## Disclosure: use of generative AI tools

Parts of this package have been developed and refined with the assistance of generative AI tools (Anthropic Claude, primarily for docstring drafting, code review, and iterative tuning of detector parameters). All algorithmic design decisions, validation against real sherd data, and final acceptance of changes were performed by the human maintainer. If you contribute, please disclose any non-trivial use of AI assistance in your PR description.

## Questions

Not sure where to start? Open an issue or email the maintainer at [aleciaco@uw.edu](mailto:aleciaco@uw.edu).
