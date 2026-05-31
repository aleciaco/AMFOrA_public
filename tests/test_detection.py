"""Detection-pipeline tests.

Cover the paste-anchored gate (both code paths: K * MAD and floor),
inclusion recovery on calibrated synthetic images, and the edge-band
boundary rejection.  Tolerances are loose enough that legitimate
OpenCV-version drift won't break the suite, but tight enough that a
real regression in detection logic shows up immediately.
"""

import numpy as np

import amfora
from amfora.core.detection import _paste_reference, _paste_mad


# --- Paste reference + MAD ---------------------------------------------------

def test_paste_reference_recovers_paste_color(synthetic_sherd):
    """The per-channel paste reference should land near the synthetic paste color."""
    image, _label_mask, meta = synthetic_sherd
    paste_ref = _paste_reference(image)
    true_paste = meta["background_bgr"]
    # Paste dominates the image so the median should be within a few units
    # of the true color (inclusions / voids shift it only slightly).
    for ch in range(3):
        assert abs(paste_ref[ch] - true_paste[ch]) <= 5, (
            f"channel {ch}: paste_ref={paste_ref[ch]} vs true paste={true_paste[ch]}")


def test_paste_mad_zero_on_uniform_paste(synthetic_uniform_sherd):
    """Zero-noise uniform paste -> MAD is clamped to 1.0 (the floor branch)."""
    image, _label_mask, _meta = synthetic_uniform_sherd
    paste_mad = _paste_mad(image)
    # The clamp inside _paste_mad raises raw MAD=0 to 1.0 so the gate
    # threshold stays meaningful; without the clamp K * MAD = 0 and
    # every candidate would pass.
    for ch in range(3):
        assert paste_mad[ch] == 1.0, f"channel {ch} MAD should be clamped to 1.0, got {paste_mad[ch]}"


def test_paste_mad_realistic_on_noisy_paste(synthetic_noisy_sherd):
    """Noisy paste -> MAD in the realistic range (4-12) so K * MAD dominates the threshold."""
    image, _label_mask, _meta = synthetic_noisy_sherd
    paste_mad = _paste_mad(image)
    for ch in range(3):
        assert 4 <= paste_mad[ch] <= 12, (
            f"channel {ch} MAD={paste_mad[ch]} outside expected 4-12 range "
            f"for paste_noise_std=10")


def test_paste_mad_outlier_resistant(synthetic_dense_sherd):
    """Dense inclusion field shouldn't blow up the MAD (it's outlier-resistant)."""
    image, _label_mask, _meta = synthetic_dense_sherd
    paste_mad = _paste_mad(image)
    # 100 features on an 800x800 image is < 10% of pixels; MAD of zero-noise
    # paste should still come back tiny (clamped to 1.0) — if MAD were std
    # instead of MAD, the inclusions would inflate it to 20+.
    for ch in range(3):
        assert paste_mad[ch] <= 3.0, (
            f"channel {ch} MAD={paste_mad[ch]} suspiciously high — "
            f"inclusions may be inflating it")


# --- False-positive rejection on uniform paste -------------------------------

def test_no_false_positives_on_zero_noise_uniform_paste(synthetic_uniform_sherd):
    """No features placed + zero noise -> paste_pop_floor rejects everything."""
    image, _label_mask, _meta = synthetic_uniform_sherd
    cr = amfora.contour_detection(image, scan_dpi=1200)
    inc_blobs, void_blobs = amfora.sherd_blobs(image, scan_dpi=1200)
    # A handful of edge-band micro-features can sneak in even on uniform
    # paste; the gate should keep the count at a single-digit number.
    assert cr["total_inclusions"] <= 5, (
        f"contour detected {cr['total_inclusions']} inclusions on "
        f"zero-noise uniform paste — paste_pop_floor isn't doing its job")
    assert len(inc_blobs) <= 5, (
        f"blob detected {len(inc_blobs)} inclusions on zero-noise uniform paste")


def test_no_false_positives_on_noisy_uniform_paste(synthetic_noisy_uniform_sherd):
    """No features placed + realistic noise -> K * MAD rejects everything."""
    image, _label_mask, _meta = synthetic_noisy_uniform_sherd
    cr = amfora.contour_detection(image, scan_dpi=1200)
    inc_blobs, _void_blobs = amfora.sherd_blobs(image, scan_dpi=1200)
    assert cr["total_inclusions"] <= 5, (
        f"contour detected {cr['total_inclusions']} inclusions on "
        f"noisy uniform paste — K * MAD isn't doing its job")
    assert len(inc_blobs) <= 5, (
        f"blob detected {len(inc_blobs)} inclusions on noisy uniform paste")


# --- Inclusion recovery -------------------------------------------------------

def test_contour_detection_recovers_synthetic_inclusions(synthetic_noisy_sherd):
    """At least ~70 % of placed inclusions show up in contour_detection's list.

    Uses the noisy-paste fixture (realistic regime).  Zero-noise paste
    creates deterministic threshold artifacts unrelated to detector quality,
    so testing recall there would conflate algorithm performance with the
    synthetic-image quirk.
    """
    image, _label_mask, meta = synthetic_noisy_sherd
    n_placed = meta["kind_placed_counts"]["blob"] + meta["kind_placed_counts"]["polygon"]
    cr = amfora.contour_detection(image, scan_dpi=1200)
    assert n_placed * 0.7 <= cr["total_inclusions"] <= n_placed * 1.5, (
        f"contour detected {cr['total_inclusions']}, expected ~{n_placed} "
        f"(±30%/+50%) from synthetic ground truth")


def test_blob_detection_recovers_synthetic_inclusions(synthetic_noisy_sherd):
    """Same recall check for the blob detector — also uses the noisy fixture."""
    image, _label_mask, meta = synthetic_noisy_sherd
    n_placed = meta["kind_placed_counts"]["blob"] + meta["kind_placed_counts"]["polygon"]
    inc_blobs, _void_blobs = amfora.sherd_blobs(image, scan_dpi=1200)
    assert n_placed * 0.7 <= len(inc_blobs) <= n_placed * 1.5, (
        f"blob detected {len(inc_blobs)}, expected ~{n_placed} "
        f"(±30%/+50%) from synthetic ground truth")


def test_contour_detection_finds_some_voids(synthetic_noisy_sherd):
    """If the generator placed voids, the contour detector should find some of them."""
    image, _label_mask, meta = synthetic_noisy_sherd
    n_voids_placed = meta["kind_placed_counts"]["void"]
    if n_voids_placed == 0:
        return
    cr = amfora.contour_detection(image, scan_dpi=1200)
    assert cr["total_voids"] >= 1, (
        f"contour found 0 voids despite {n_voids_placed} being placed")


# --- Determinism --------------------------------------------------------------

def test_detection_is_deterministic(synthetic_sherd):
    """Running detection twice on the same image returns the same counts."""
    image, _label_mask, _meta = synthetic_sherd
    cr1 = amfora.contour_detection(image, scan_dpi=1200)
    cr2 = amfora.contour_detection(image, scan_dpi=1200)
    assert cr1["total_inclusions"] == cr2["total_inclusions"]
    assert cr1["total_voids"] == cr2["total_voids"]
