"""Analysis-pipeline tests for analyze_single_sherd.

Confirm the function returns the documented dict shape, that the
``effective_detection_area_cm2`` derivation is consistent with the
edge-band gate inside the detectors, and that density / area-percentage
metrics divide by the effective area (not the full sherd area).
"""

import amfora


def test_analyze_single_sherd_returns_expected_keys(synthetic_sherd):
    """The result dict should contain the documented metric keys."""
    image, _label_mask, _meta = synthetic_sherd
    r = amfora.analyze_single_sherd(
        image, scan_dpi=1200, analyze_core_periphery=False, pre_masked=True
    )
    expected = {
        "sherd_area_cm2",
        "effective_detection_area_cm2",
        "blob_inclusion_count",
        "blob_void_count",
        "blob_inclusion_density_per_cm2",
        "blob_void_density_per_cm2",
        "blob_inclusion_area_percentage",
        "contour_inclusion_count",
        "contour_void_count",
        "contour_inclusion_density_per_cm2",
        "contour_void_density_per_cm2",
        "contour_inclusion_area_percentage",
        "analysis_status",
    }
    missing = expected - set(r.keys())
    assert not missing, f"missing keys in analyze_single_sherd result: {missing}"


def test_effective_area_is_less_than_sherd_area(synthetic_sherd):
    """Edge erosion should shrink the denominator by the expected ~25-35 %."""
    image, _label_mask, _meta = synthetic_sherd
    r = amfora.analyze_single_sherd(
        image, scan_dpi=1200, analyze_core_periphery=False, pre_masked=True
    )
    sa = r["sherd_area_cm2"]
    ea = r["effective_detection_area_cm2"]
    assert ea < sa, "effective area should be smaller than full sherd area after edge erosion"
    # 4 % erosion of a 500x500 sherd gives ~80-85 % of the area; loose bound
    # for safety against image-size variation.
    assert 0.5 * sa < ea < sa, (
        f"effective area {ea:.2f} cm² unexpectedly far from sherd area {sa:.2f} cm² "
        f"(expected 50-100 % of sherd area)"
    )


def test_density_uses_effective_area_not_full_sherd(synthetic_sherd):
    """blob_inclusion_density should equal count / effective_area (not sherd_area)."""
    image, _label_mask, _meta = synthetic_sherd
    r = amfora.analyze_single_sherd(
        image, scan_dpi=1200, analyze_core_periphery=False, pre_masked=True
    )
    count = r["blob_inclusion_count"]
    density = r["blob_inclusion_density_per_cm2"]
    ea = r["effective_detection_area_cm2"]
    if ea > 0 and count > 0:
        expected = count / ea
        assert abs(density - expected) < 1e-6, (
            f"density {density:.4f} doesn't match count/effective_area = {expected:.4f}"
        )


def test_analysis_succeeds_status_field(synthetic_sherd):
    """A normal run should report analysis_status == 'success'."""
    image, _label_mask, _meta = synthetic_sherd
    r = amfora.analyze_single_sherd(
        image, scan_dpi=1200, analyze_core_periphery=False, pre_masked=True
    )
    assert r["analysis_status"] == "success", f"unexpected status: {r['analysis_status']}"
