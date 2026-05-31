"""Shared test fixtures.

A pytest fixture is a helper that builds something a test needs (an image,
a temporary directory, etc.).  Tests "request" a fixture by listing its
name as a parameter; pytest injects it automatically.

We use ``amfora.testing.generate_ceramic_image`` (the same generator the
notebooks use) so tests exercise images calibrated to fall inside the
detector's documented envelopes — any failure points at a real algorithmic
issue, not at a synthetic-image quirk.  Each fixture pins a seed so test
results are reproducible.
"""

import pytest

from amfora.testing import generate_ceramic_image


@pytest.fixture
def synthetic_sherd():
    """500x500 sherd with 50 inclusions/voids at the default 60/25/15 mix.

    Returns (image_bgr, label_mask, metadata).  ``metadata['inclusions']``
    is the per-feature ground truth; ``metadata['kind_placed_counts']``
    gives the per-kind tally (blob, polygon, void).
    """
    return generate_ceramic_image(
        image_size=(500, 500),
        n_inclusions=50,
        size_range=(6, 14),
        seed=42,
    )


@pytest.fixture
def synthetic_uniform_sherd():
    """Same paste as synthetic_sherd but with zero features placed.

    Used for false-positive tests — a well-calibrated detector should
    return very few inclusions on a uniform paste.
    """
    return generate_ceramic_image(
        image_size=(500, 500),
        n_inclusions=0,
        size_range=(6, 14),
        seed=7,
    )


@pytest.fixture
def synthetic_dense_sherd():
    """Larger sherd with 100 features for tests that need a denser field."""
    return generate_ceramic_image(
        image_size=(800, 800),
        n_inclusions=100,
        size_range=(6, 14),
        seed=123,
    )


@pytest.fixture
def synthetic_noisy_sherd():
    """700x700 sherd with realistic per-pixel paste noise (MAD ~7).

    Exercises the K * MAD branch of the paste-anchored pop gate — the
    everyday code path on real flatbed scans, distinct from the
    zero-noise fixtures above which exercise the ``paste_pop_floor``
    fallback branch.  Image is large enough that the 4 % edge band
    doesn't dominate inclusion placement (~14 % area loss vs. 16 % on
    the 500 px fixture).
    """
    return generate_ceramic_image(
        image_size=(700, 700),
        n_inclusions=50,
        size_range=(6, 14),
        seed=99,
        paste_noise_std=10,
    )


@pytest.fixture
def synthetic_noisy_uniform_sherd():
    """Noisy paste with NO features placed — for noisy-regime false-positive tests."""
    return generate_ceramic_image(
        image_size=(500, 500),
        n_inclusions=0,
        size_range=(6, 14),
        seed=11,
        paste_noise_std=10,
    )
