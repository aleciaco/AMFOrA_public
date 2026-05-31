"""Smoke tests: the package and its key entry points import cleanly.

These are the cheapest, most-likely-to-catch-something tests in the suite.
If a future change breaks an import, every other test will also fail with
a less-useful error — so checking imports first surfaces the real root
cause.  Failure here usually means a typo in __init__ or a missing
dependency in pyproject.toml.
"""


def test_package_imports():
    import amfora
    assert hasattr(amfora, "__version__"), "amfora.__version__ should exist"
    assert isinstance(amfora.__version__, str)


def test_top_level_api_exposed():
    """The headline functions should be reachable as `amfora.<name>` directly."""
    import amfora
    expected = [
        "sherd_mask",
        "apply_mask",
        "sherd_blobs",
        "contour_detection",
        "analyze_single_sherd",
        "full_analysis",
    ]
    missing = [name for name in expected if not hasattr(amfora, name)]
    assert not missing, f"missing top-level names: {missing}"


def test_testing_module_imports():
    """The synthetic-image helper used by the test suite (and notebooks)."""
    from amfora.testing import generate_ceramic_image, generate_ceramic_image_batch
    assert callable(generate_ceramic_image)
    assert callable(generate_ceramic_image_batch)
