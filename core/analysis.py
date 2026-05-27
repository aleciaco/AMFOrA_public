"""
Analysis functions for AMACFA+ ceramic fabric analysis.

This module contains functions for analyzing inclusions, voids, orientations,
and colors in ceramic sherds with enhanced accuracy and robustness.
"""

import cv2
import numpy as np
import pandas as pd
import math
from pathlib import Path
from scipy.stats import skew
from .detection import setup_robust_blob_params, sherd_mask, apply_mask, sherd_blobs, contour_detection

__all__ = [
    'size_count_summary_single', 'analyze_single_sherd', 'full_analysis',
    'size_count_summary', 'void_counter', 'contour_counter',
    'sacredsquare', 'inclusion_colors', 'inclusion_colors_from_contours',
    'inclusion_orientation',
    'inclusion_orientation2', 'sherd_color_analysis', 'sherd_color_summary',
    'extract_core_periphery_colors', 'analyze_inclusion_angularity',
    'analyze_orientation_for_pca', 'analyze_manufacturing_technique',
]


def _pad_crop(image_slice, crop):
    """Apply edge-padding to an image slice to match a padded mask."""
    if len(crop) == 8:
        pt, pb, pl, pr = crop[4:]
        if pt or pb or pl or pr:
            return np.pad(image_slice, ((pt, pb), (pl, pr), (0, 0)),
                          mode='constant', constant_values=0)
    return image_slice


def _prioritize_columns(df, primary_cols):
    """Move *primary_cols* to the front of *df*, preserving remaining order."""
    present = [c for c in primary_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in present]
    return df[present + remaining]


def _interleave_method_columns(df, primary_cols):
    """Reorder DataFrame so blob/contour columns are interleaved by metric.

    For every metric suffix (e.g. ``inclusion_count``), the blob\_ and
    contour\_ variants are placed side-by-side. Metric order follows
    first-seen insertion order. Columns without a method prefix are
    appended at the end.
    """
    present = list(df.columns)
    primary = [c for c in primary_cols if c in present]
    used = set(primary)

    # Map metric suffix → column name for each method
    blob_metrics = {}
    contour_metrics = {}
    for col in present:
        if col.startswith('blob_'):
            blob_metrics[col[5:]] = col
        elif col.startswith('contour_'):
            contour_metrics[col[8:]] = col

    # Unique metrics in first-seen order
    all_metrics = []
    seen = set()
    for col in present:
        if col.startswith('blob_'):
            metric = col[5:]
        elif col.startswith('contour_'):
            metric = col[8:]
        else:
            continue
        if metric not in seen:
            all_metrics.append(metric)
            seen.add(metric)

    # Build interleaved list
    interleaved = []
    for metric in all_metrics:
        if metric in blob_metrics:
            interleaved.append(blob_metrics[metric])
            used.add(blob_metrics[metric])
        if metric in contour_metrics:
            interleaved.append(contour_metrics[metric])
            used.add(contour_metrics[metric])

    remaining = [c for c in present if c not in used]
    return df[primary + interleaved + remaining]


def size_count_summary_single(blobs_light, blobs_dark, scan_dpi=1200):
    """
    Analyze size distributions for a single image's detected blobs.
    
    Parameters
    ----------
    blobs_light : list
        List of light blob keypoints (inclusions)
    blobs_dark : list  
        List of dark blob keypoints (voids)
    scan_dpi : int, optional
        Scan resolution in dots per inch (default: 1200)
        
    Returns
    -------
    dict
        Dictionary containing comprehensive size statistics for inclusions and voids
    """
    # Validate DPI input
    if scan_dpi < 150 or scan_dpi > 2400:
        print(f"Warning: scan_dpi {scan_dpi} is outside recommended range (150-2400). Results may be unreliable.")
    
    # Convert DPI to dots per centimeter
    dpcm = scan_dpi * 0.3937
    
    # Calculate areas for light blobs (inclusions)
    inclusion_areas = [np.pi * ((blob.size/2)/dpcm)**2 for blob in blobs_light]
    
    # Calculate areas for dark blobs (voids) 
    void_areas = [np.pi * ((blob.size/2)/dpcm)**2 for blob in blobs_dark]
    
    # Calculate comprehensive statistics
    results = {}
    
    # Inclusion statistics
    if inclusion_areas:
        results['inclusion_count'] = len(inclusion_areas)
        results['inclusion_total_area_cm2'] = np.sum(inclusion_areas)
        results['inclusion_mean_area_cm2'] = np.mean(inclusion_areas)
        results['inclusion_std_area_cm2'] = np.std(inclusion_areas)
        results['inclusion_min_area_cm2'] = np.min(inclusion_areas)
        results['inclusion_max_area_cm2'] = np.max(inclusion_areas)
        results['inclusion_median_area_cm2'] = np.median(inclusion_areas)
        results['inclusion_25pct_cm2'] = np.percentile(inclusion_areas, 25)
        results['inclusion_75pct_cm2'] = np.percentile(inclusion_areas, 75)
        results['inclusion_skewness'] = float(skew(inclusion_areas)) if len(inclusion_areas) >= 3 else 0.0
        results['inclusion_cv'] = float(np.std(inclusion_areas) / np.mean(inclusion_areas)) if np.mean(inclusion_areas) > 0 else 0.0
    else:
        results['inclusion_count'] = 0
        for key in ['inclusion_total_area_cm2', 'inclusion_mean_area_cm2', 'inclusion_std_area_cm2',
                   'inclusion_min_area_cm2', 'inclusion_max_area_cm2', 'inclusion_median_area_cm2',
                   'inclusion_25pct_cm2', 'inclusion_75pct_cm2', 'inclusion_skewness', 'inclusion_cv']:
            results[key] = 0
    
    # Void statistics
    if void_areas:
        results['void_count'] = len(void_areas)
        results['void_total_area_cm2'] = np.sum(void_areas)
        results['void_mean_area_cm2'] = np.mean(void_areas)
        results['void_std_area_cm2'] = np.std(void_areas)
        results['void_min_area_cm2'] = np.min(void_areas)
        results['void_max_area_cm2'] = np.max(void_areas)
        results['void_median_area_cm2'] = np.median(void_areas)
        results['void_25pct_cm2'] = np.percentile(void_areas, 25)
        results['void_75pct_cm2'] = np.percentile(void_areas, 75)
        results['void_skewness'] = float(skew(void_areas)) if len(void_areas) >= 3 else 0.0
        results['void_cv'] = float(np.std(void_areas) / np.mean(void_areas)) if np.mean(void_areas) > 0 else 0.0
    else:
        results['void_count'] = 0
        for key in ['void_total_area_cm2', 'void_mean_area_cm2', 'void_std_area_cm2',
                   'void_min_area_cm2', 'void_max_area_cm2', 'void_median_area_cm2',
                   'void_25pct_cm2', 'void_75pct_cm2', 'void_skewness', 'void_cv']:
            results[key] = 0

    return results


def analyze_single_sherd(image, scan_dpi=1200, analyze_inclusions=True, analyze_voids=True,
                         analyze_core_periphery=True, use_blob=True, use_contour=True,
                         enhance_contrast=True, clahe_clip=2.0, clahe_grid=(8, 8),
                         channels=('B', 'G', 'R'), combine_mode='union', vote_min=2,
                         void_intensity_max=60.0, inclusion_pop_min=25.0):
    """
    Comprehensive analysis of a single ceramic sherd image.

    Parameters
    ----------
    image : numpy.ndarray
        Image array of a scanned sherd
    scan_dpi : int, optional
        Scan resolution in dots per inch (default: 1200)
    analyze_inclusions : bool, optional
        Whether to analyze inclusions (default: True)
    analyze_voids : bool, optional
        Whether to analyze voids (default: True)
    analyze_core_periphery : bool, optional
        Whether to perform core-periphery color analysis for firing atmosphere
        interpretation. This is computationally intensive. (default: True)
    use_blob : bool, optional
        Whether to use blob detection method (default: True)
    use_contour : bool, optional
        Whether to use contour detection method (default: True)
    enhance_contrast : bool, optional
        If True (default), apply CLAHE before running blob/contour detection.
        When ``channels == ('L',)`` CLAHE is applied once to the masked image's
        L* channel via the BGR round-trip; when multiple channels are requested
        CLAHE is instead applied to each channel inside the detectors so every
        channel benefits from the contrast enhancement.  Set to False to
        disable entirely.
    clahe_clip : float, optional
        CLAHE clip limit when ``enhance_contrast=True`` (default: 2.0).
    clahe_grid : tuple of int, optional
        CLAHE tile grid size when ``enhance_contrast=True`` (default: (8, 8)).
    channels : tuple of str, optional
        Image channels passed to both detectors.  Default
        ``('B', 'G', 'R')`` runs detection independently on each of OpenCV's
        native BGR channels (not RGB) and combines the results, so inclusions
        that only contrast strongly in one channel (e.g. iron-rich grains in
        R, organic dark cores in B) are still picked up.  L\* is excluded
        from the default because it is a perceptually-weighted blend of B,
        G, and R, so including it gives features visible in L\* an extra
        redundant vote in the combination step.  Set ``channels=('L',)`` to
        recover the pre-multi-channel behavior.
    combine_mode : {'union', 'vote'}, optional
        How to combine per-channel detections when ``len(channels) > 1``.
        Default ``'union'`` pools detections across channels and removes
        spatial duplicates without requiring cross-channel agreement.
        This catches monochromatic features (e.g. reddish-brown grog has
        near-zero contrast in the R channel against a warm cream paste
        and only appears in B) that the old ``'vote'`` default with
        ``vote_min=2`` dropped — roughly half of legitimate inclusions
        on iron-rich tempered fabrics were lost to the agreement
        requirement.  Noise rejection is instead handled by
        ``inclusion_pop_min``, which directly measures true intensity
        contrast on the raw (pre-CLAHE) pixels and is a stronger
        discriminator than per-channel voting.  Both detectors accept
        this parameter; the contour detector's noise rejection is less
        voting-dependent (shape filters do more of the work), so the
        ``'union'`` default trades a small contour-recall hit (~5%) for
        a large blob-recall gain (~27%) on iron-tempered fabrics.  Use
        ``'vote'`` only if you have a specific reason to require
        cross-channel agreement (e.g. very noisy scans).
    vote_min : int, optional
        Minimum number of channels that must agree for a feature to be kept
        when ``combine_mode='vote'`` (default: 2 of 3 BGR channels).
        Ignored under the default ``combine_mode='union'``.
    void_intensity_max : float in 0..255, optional
        Brightness gate applied to void detections in both detectors.  A
        candidate void's interior must read below this mean pixel intensity
        on its channel (default: 60).  This is the primary
        inclusion-vs-void discriminator on real scans, because the
        DPI-scaled blur smooths shape concavities that would otherwise
        distinguish a pore (hole, near-black inside) from a dark mineral
        grain (just darker paste).  Lower (e.g. 45) for stricter void
        detection; raise (e.g. 90) for low-contrast scans where genuine
        pores don't quite reach black.
    inclusion_pop_min : float in 0..255, optional
        Minimum "pop" required for a candidate inclusion to be kept
        (default: 25).  "Pop" here is informal shorthand for **how
        much the feature visually stands out against its immediate
        surrounding paste** — concretely, the absolute difference
        between the candidate's core disc mean intensity and the mean
        intensity of a surrounding annulus, computed on the **raw
        (pre-CLAHE) BGR channels** of the masked input image and taken
        as the maximum across the three native channels:

            pop = max over BGR of |mean(core disc) - mean(annulus)|

        Higher values mean a clearer intensity discontinuity between
        the inclusion and the paste around it; a low pop value means
        the "blob" the detector found is actually flat against its
        surround — almost certainly CLAHE-amplified noise dressed up to
        look like a feature.  Applied to both blob and contour
        detectors as the final inclusion filter.  Under the default
        ``combine_mode='union'`` this is the primary noise rejection
        mechanism, replacing the old cross-channel voting requirement.
        Calibrated on AMFOrA_Test_Bars: at the default, ~43% reduction
        in R01-series (uniform paste, no true inclusions) false
        positives vs the prior vote=2 + pop=20 configuration, with
        ~27% MORE true positives caught on iron-tempered fabrics
        (R03G_4 jumps from 37 → 53 detections, matching visual
        inspection).  Set to 0 to disable; raise (e.g. 30–35) for very
        uniform paste or to further tighten precision; lower (e.g.
        15–20) when chasing subtle features in fine-grained fabrics
        (pair with ``combine_mode='vote'`` for additional noise control
        if needed).

    Returns
    -------
    dict
        Dictionary containing comprehensive analysis results
    """
    from .detection import sherd_mask, apply_mask, sherd_blobs, clahe_enhance

    results = {}

    try:
        # Create sherd mask and apply it (mask is auto-cropped; crop carries the bounds)
        # best_contour is the sherd boundary contour (image_cropped coords) used for masking
        mask, crop, best_contour = sherd_mask(image, scan_dpi=scan_dpi)
        masked_image = apply_mask(image, mask, crop)

        # CLAHE strategy:
        #   - Single-channel L* mode keeps the legacy BGR-roundtrip path
        #     (clahe_enhance on the masked image) so results match prior runs.
        #   - Multi-channel mode pushes CLAHE into the detectors so every
        #     requested channel — not just L* — receives contrast enhancement.
        multi_channel = len(channels) > 1
        if enhance_contrast and not multi_channel:
            masked_image = clahe_enhance(masked_image,
                                         clip_limit=clahe_clip,
                                         tile_grid=clahe_grid)
        detector_enhance = bool(enhance_contrast and multi_channel)

        # Calculate sherd area for density calculations (pixel count is unaffected by crop).
        # sherd_mask returns a 3-channel mask by default, so collapse to 2D before
        # counting — otherwise each pixel is counted once per channel (3x inflation).
        mask_2d = mask[:, :, 0] if mask.ndim == 3 else mask
        sherd_area_cm2 = np.sum(mask_2d > 0) / ((scan_dpi * 0.3937) ** 2)
        results['sherd_area_cm2'] = sherd_area_cm2

        # BLOB DETECTION - Better for round, circular inclusions and voids
        # Good for: quartz grains, rounded temper, spherical voids
        # Less good for: angular fragments, elongated inclusions, irregular shapes
        if use_blob:
            light_blobs, dark_blobs = sherd_blobs(
                masked_image, scan_dpi=scan_dpi,
                channels=channels, combine_mode=combine_mode, vote_min=vote_min,
                enhance_contrast=detector_enhance,
                clahe_clip=clahe_clip, clahe_grid=clahe_grid,
                void_intensity_max=void_intensity_max,
                inclusion_pop_min=inclusion_pop_min)

            # Size analysis for BLOB detection
            blob_stats = size_count_summary_single(light_blobs, dark_blobs, scan_dpi)
            # Prefix all blob stats with 'blob_'
            for key, value in blob_stats.items():
                results[f'blob_{key}'] = value
        else:
            light_blobs, dark_blobs = [], []

        # CONTOUR DETECTION - Better for irregular, angular, and complex shapes
        # Good for: angular rock fragments, irregular voids, elongated inclusions
        # Less good for: very small features, noisy backgrounds
        if use_contour:
            from .detection import contour_detection
            try:
                contour_results = contour_detection(
                    masked_image, scan_dpi=scan_dpi, debug_mode=False,
                    channels=channels, combine_mode=combine_mode, vote_min=vote_min,
                    enhance_contrast=detector_enhance,
                    clahe_clip=clahe_clip, clahe_grid=clahe_grid,
                    void_intensity_max=void_intensity_max,
                    inclusion_pop_min=inclusion_pop_min)
                contour_inclusions = contour_results.get('inclusions', [])
                contour_voids = contour_results.get('voids', [])

                # Convert contour data to areas (cm²)
                dpcm = scan_dpi * 0.3937
                contour_inclusion_areas = [cv2.contourArea(c) / (dpcm ** 2) for c in contour_inclusions]
                contour_void_areas = [cv2.contourArea(c) / (dpcm ** 2) for c in contour_voids]

            except Exception as e:
                print(f"Warning: Contour detection failed: {e}")
                contour_results = {}
                contour_inclusions, contour_voids = [], []
                contour_inclusion_areas, contour_void_areas = [], []
        else:
            contour_results = {}
            contour_inclusions, contour_voids = [], []
            contour_inclusion_areas, contour_void_areas = [], []
        
        # Size analysis for CONTOUR detection manually since it returns areas not keypoints
        if use_contour:
            contour_stats = {}

            # Inclusion statistics from contours
            if contour_inclusion_areas:
                contour_stats['inclusion_count'] = len(contour_inclusion_areas)
                contour_stats['inclusion_total_area_cm2'] = np.sum(contour_inclusion_areas)
                contour_stats['inclusion_mean_area_cm2'] = np.mean(contour_inclusion_areas)
                contour_stats['inclusion_std_area_cm2'] = np.std(contour_inclusion_areas)
                contour_stats['inclusion_min_area_cm2'] = np.min(contour_inclusion_areas)
                contour_stats['inclusion_max_area_cm2'] = np.max(contour_inclusion_areas)
                contour_stats['inclusion_median_area_cm2'] = np.median(contour_inclusion_areas)
                contour_stats['inclusion_25pct_cm2'] = np.percentile(contour_inclusion_areas, 25)
                contour_stats['inclusion_75pct_cm2'] = np.percentile(contour_inclusion_areas, 75)
                contour_stats['inclusion_skewness'] = float(skew(contour_inclusion_areas)) if len(contour_inclusion_areas) >= 3 else 0.0
                contour_stats['inclusion_cv'] = float(np.std(contour_inclusion_areas) / np.mean(contour_inclusion_areas)) if np.mean(contour_inclusion_areas) > 0 else 0.0
            else:
                contour_stats['inclusion_count'] = 0
                for key in ['inclusion_total_area_cm2', 'inclusion_mean_area_cm2', 'inclusion_std_area_cm2',
                           'inclusion_min_area_cm2', 'inclusion_max_area_cm2', 'inclusion_median_area_cm2',
                           'inclusion_25pct_cm2', 'inclusion_75pct_cm2', 'inclusion_skewness', 'inclusion_cv']:
                    contour_stats[key] = 0

            # Void statistics from contours
            if contour_void_areas:
                contour_stats['void_count'] = len(contour_void_areas)
                contour_stats['void_total_area_cm2'] = np.sum(contour_void_areas)
                contour_stats['void_mean_area_cm2'] = np.mean(contour_void_areas)
                contour_stats['void_std_area_cm2'] = np.std(contour_void_areas)
                contour_stats['void_min_area_cm2'] = np.min(contour_void_areas)
                contour_stats['void_max_area_cm2'] = np.max(contour_void_areas)
                contour_stats['void_median_area_cm2'] = np.median(contour_void_areas)
                contour_stats['void_25pct_cm2'] = np.percentile(contour_void_areas, 25)
                contour_stats['void_75pct_cm2'] = np.percentile(contour_void_areas, 75)
                contour_stats['void_skewness'] = float(skew(contour_void_areas)) if len(contour_void_areas) >= 3 else 0.0
                contour_stats['void_cv'] = float(np.std(contour_void_areas) / np.mean(contour_void_areas)) if np.mean(contour_void_areas) > 0 else 0.0
            else:
                contour_stats['void_count'] = 0
                for key in ['void_total_area_cm2', 'void_mean_area_cm2', 'void_std_area_cm2',
                           'void_min_area_cm2', 'void_max_area_cm2', 'void_median_area_cm2',
                           'void_25pct_cm2', 'void_75pct_cm2', 'void_skewness', 'void_cv']:
                    contour_stats[key] = 0

            # Prefix all contour stats with 'contour_'
            for key, value in contour_stats.items():
                results[f'contour_{key}'] = value

        # Density calculations
        if use_blob:
            if sherd_area_cm2 > 0:
                results['blob_inclusion_density_per_cm2'] = results['blob_inclusion_count'] / sherd_area_cm2
                results['blob_void_density_per_cm2'] = results['blob_void_count'] / sherd_area_cm2
                results['blob_inclusion_area_percentage'] = (results['blob_inclusion_total_area_cm2'] / sherd_area_cm2) * 100
                results['blob_void_area_percentage'] = (results['blob_void_total_area_cm2'] / sherd_area_cm2) * 100
            else:
                results['blob_inclusion_density_per_cm2'] = 0
                results['blob_void_density_per_cm2'] = 0
                results['blob_inclusion_area_percentage'] = 0
                results['blob_void_area_percentage'] = 0

        if use_contour:
            if sherd_area_cm2 > 0:
                results['contour_inclusion_density_per_cm2'] = results['contour_inclusion_count'] / sherd_area_cm2
                results['contour_void_density_per_cm2'] = results['contour_void_count'] / sherd_area_cm2
                results['contour_inclusion_area_percentage'] = (results['contour_inclusion_total_area_cm2'] / sherd_area_cm2) * 100
                results['contour_void_area_percentage'] = (results['contour_void_total_area_cm2'] / sherd_area_cm2) * 100
            else:
                results['contour_inclusion_density_per_cm2'] = 0
                results['contour_void_density_per_cm2'] = 0
                results['contour_inclusion_area_percentage'] = 0
                results['contour_void_area_percentage'] = 0
        
        # Orientation analysis — contour-based (reuses pre-detected contour inclusions/voids)
        if use_contour and analyze_inclusions and contour_inclusions:
            try:
                orientation_data = inclusion_orientation2(masked_image, scan_dpi,
                                                         contour_result=contour_results, sherd_contour=best_contour)
                if orientation_data and len(orientation_data) >= 3:
                    inc_angles, void_angles, sherd_angle = orientation_data
                    results['contour_inclusion_orientation_mean'] = np.mean(inc_angles) if inc_angles else 0
                    results['contour_inclusion_orientation_std'] = np.std(inc_angles) if inc_angles else 0
                    results['sherd_orientation'] = sherd_angle if sherd_angle is not None else 0
                else:
                    results['contour_inclusion_orientation_mean'] = 0
                    results['contour_inclusion_orientation_std'] = 0
                    results['sherd_orientation'] = 0
            except Exception as e:
                print(f"Warning: Contour orientation analysis failed: {e}")
                results['contour_inclusion_orientation_mean'] = 0
                results['contour_inclusion_orientation_std'] = 0
                results['sherd_orientation'] = 0
        elif use_contour:
            # Contour enabled but no inclusions to orient — zero-fill contour columns
            results['contour_inclusion_orientation_mean'] = 0
            results['contour_inclusion_orientation_std'] = 0
            results['sherd_orientation'] = 0
        else:
            # Contour disabled — only keep the unprefixed sherd_orientation
            results['sherd_orientation'] = 0
        
        # Blob inclusion color analysis
        if use_blob and analyze_inclusions and light_blobs:
            try:
                sq_list, _ = sacredsquare(masked_image, light_blobs)
                color_data = inclusion_colors(masked_image, sq_list)
                # inclusion_colors returns a list of [[L,a,b], [L,a,b], [L,a,b]] per inclusion
                # Extract the dominant (first) Lab triplet from each inclusion
                if color_data:
                    dominant_lab = np.array([inc[0] for inc in color_data], dtype=float)
                    results['blob_inclusion_color_l_mean'] = np.mean(dominant_lab[:, 0])
                    results['blob_inclusion_color_a_mean'] = np.mean(dominant_lab[:, 1])
                    results['blob_inclusion_color_b_mean'] = np.mean(dominant_lab[:, 2])
                    results['blob_inclusion_color_l_std'] = np.std(dominant_lab[:, 0])
                    results['blob_inclusion_color_a_std'] = np.std(dominant_lab[:, 1])
                    results['blob_inclusion_color_b_std'] = np.std(dominant_lab[:, 2])
                    if len(dominant_lab) > 1:
                        dists = [np.linalg.norm(dominant_lab[i] - dominant_lab[j])
                                 for i in range(len(dominant_lab)) for j in range(i+1, len(dominant_lab))]
                        results['blob_inclusion_color_diversity'] = np.mean(dists)
                    else:
                        results['blob_inclusion_color_diversity'] = 0
                else:
                    for key in ['blob_inclusion_color_l_mean', 'blob_inclusion_color_a_mean', 'blob_inclusion_color_b_mean',
                               'blob_inclusion_color_l_std', 'blob_inclusion_color_a_std', 'blob_inclusion_color_b_std',
                               'blob_inclusion_color_diversity']:
                        results[key] = 0
            except Exception as e:
                print(f"Warning: Inclusion color analysis failed: {e}")
                for key in ['blob_inclusion_color_l_mean', 'blob_inclusion_color_a_mean', 'blob_inclusion_color_b_mean',
                           'blob_inclusion_color_l_std', 'blob_inclusion_color_a_std', 'blob_inclusion_color_b_std',
                           'blob_inclusion_color_diversity']:
                    results[key] = 0
        elif use_blob:
            # Blob enabled but no inclusions or analyze_inclusions=False — zero-fill
            for key in ['blob_inclusion_color_l_mean', 'blob_inclusion_color_a_mean', 'blob_inclusion_color_b_mean',
                       'blob_inclusion_color_l_std', 'blob_inclusion_color_a_std', 'blob_inclusion_color_b_std',
                       'blob_inclusion_color_diversity']:
                results[key] = 0

        # Contour inclusion color analysis
        if use_contour and analyze_inclusions and contour_inclusions:
            try:
                contour_color_data = inclusion_colors_from_contours(masked_image, contour_inclusions)
                if contour_color_data:
                    dominant_lab = np.array([inc[0] for inc in contour_color_data], dtype=float)
                    results['contour_inclusion_color_l_mean'] = np.mean(dominant_lab[:, 0])
                    results['contour_inclusion_color_a_mean'] = np.mean(dominant_lab[:, 1])
                    results['contour_inclusion_color_b_mean'] = np.mean(dominant_lab[:, 2])
                    results['contour_inclusion_color_l_std'] = np.std(dominant_lab[:, 0])
                    results['contour_inclusion_color_a_std'] = np.std(dominant_lab[:, 1])
                    results['contour_inclusion_color_b_std'] = np.std(dominant_lab[:, 2])
                    if len(dominant_lab) > 1:
                        dists = [np.linalg.norm(dominant_lab[i] - dominant_lab[j])
                                 for i in range(len(dominant_lab)) for j in range(i+1, len(dominant_lab))]
                        results['contour_inclusion_color_diversity'] = np.mean(dists)
                    else:
                        results['contour_inclusion_color_diversity'] = 0
                else:
                    for key in ['contour_inclusion_color_l_mean', 'contour_inclusion_color_a_mean', 'contour_inclusion_color_b_mean',
                               'contour_inclusion_color_l_std', 'contour_inclusion_color_a_std', 'contour_inclusion_color_b_std',
                               'contour_inclusion_color_diversity']:
                        results[key] = 0
            except Exception as e:
                print(f"Warning: Contour inclusion color analysis failed: {e}")
                for key in ['contour_inclusion_color_l_mean', 'contour_inclusion_color_a_mean', 'contour_inclusion_color_b_mean',
                           'contour_inclusion_color_l_std', 'contour_inclusion_color_a_std', 'contour_inclusion_color_b_std',
                           'contour_inclusion_color_diversity']:
                    results[key] = 0
        elif use_contour:
            # Contour enabled but no inclusions or analyze_inclusions=False — zero-fill
            for key in ['contour_inclusion_color_l_mean', 'contour_inclusion_color_a_mean', 'contour_inclusion_color_b_mean',
                       'contour_inclusion_color_l_std', 'contour_inclusion_color_a_std', 'contour_inclusion_color_b_std',
                       'contour_inclusion_color_diversity']:
                results[key] = 0

        # Sherd color analysis (CIELAB)
        try:
            sherd_color = sherd_color_analysis(image)
            results['sherd_color_l_mean'] = sherd_color.get('mean_l', 0)
            results['sherd_color_a_mean'] = sherd_color.get('mean_a', 128)
            results['sherd_color_b_mean'] = sherd_color.get('mean_b', 128)
        except Exception as e:
            print(f"Warning: Sherd color analysis failed: {e}")
            results['sherd_color_l_mean'] = 0
            results['sherd_color_a_mean'] = 128
            results['sherd_color_b_mean'] = 128

        # Core-periphery color analysis (firing atmosphere) - optional due to computation cost
        if analyze_core_periphery:
            try:
                core_periph = extract_core_periphery_colors(masked_image, mask, scan_dpi)
                # Core color
                if core_periph['core_lab']:
                    results['core_color_l'] = core_periph['core_lab'][0]
                    results['core_color_a'] = core_periph['core_lab'][1]
                    results['core_color_b'] = core_periph['core_lab'][2]
                else:
                    results['core_color_l'] = 0
                    results['core_color_a'] = 128
                    results['core_color_b'] = 128
                # Inner margin color
                if core_periph['inner_margin_lab']:
                    results['inner_margin_color_l'] = core_periph['inner_margin_lab'][0]
                    results['inner_margin_color_a'] = core_periph['inner_margin_lab'][1]
                    results['inner_margin_color_b'] = core_periph['inner_margin_lab'][2]
                else:
                    results['inner_margin_color_l'] = 0
                    results['inner_margin_color_a'] = 128
                    results['inner_margin_color_b'] = 128
                # Outer margin color
                if core_periph['outer_margin_lab']:
                    results['outer_margin_color_l'] = core_periph['outer_margin_lab'][0]
                    results['outer_margin_color_a'] = core_periph['outer_margin_lab'][1]
                    results['outer_margin_color_b'] = core_periph['outer_margin_lab'][2]
                else:
                    results['outer_margin_color_l'] = 0
                    results['outer_margin_color_a'] = 128
                    results['outer_margin_color_b'] = 128
                # Per-zone atmosphere classifications
                results['core_atmosphere'] = core_periph['core_atmosphere']
                results['inner_margin_atmosphere'] = core_periph['inner_margin_atmosphere']
                results['outer_margin_atmosphere'] = core_periph['outer_margin_atmosphere']
                # Gradient and interpretation
                results['core_periphery_gradient'] = core_periph['color_gradient']
                results['firing_interpretation'] = core_periph['firing_interpretation']
                results['margin_symmetry'] = core_periph['margin_symmetry']
            except Exception as e:
                print(f"Warning: Core-periphery color analysis failed: {e}")
                for key in ['core_color_l', 'inner_margin_color_l', 'outer_margin_color_l']:
                    results[key] = 0
                for key in ['core_color_a', 'core_color_b', 'inner_margin_color_a',
                           'inner_margin_color_b', 'outer_margin_color_a', 'outer_margin_color_b']:
                    results[key] = 128
                results['core_atmosphere'] = 'analysis_failed'
                results['inner_margin_atmosphere'] = 'analysis_failed'
                results['outer_margin_atmosphere'] = 'analysis_failed'
                results['core_periphery_gradient'] = 0
                results['firing_interpretation'] = 'analysis_failed'
                results['margin_symmetry'] = 'analysis_failed'

        results['analysis_status'] = 'success'
        
    except Exception as e:
        print(f"Error in sherd analysis: {e}")
        results['analysis_status'] = f'error: {str(e)}'
        # Fill with zeros for failed analysis - only for requested methods
        methods = []
        if use_blob:
            methods.append('blob')
        if use_contour:
            methods.append('contour')
        for key in ['sherd_area_cm2'] + \
                   [f'{method}_{feature}_{stat}' for method in methods
                    for feature in ['inclusion', 'void']
                    for stat in ['count', 'total_area_cm2', 'mean_area_cm2', 'std_area_cm2',
                                'min_area_cm2', 'max_area_cm2', 'median_area_cm2']] + \
                   ([f'contour_inclusion_orientation_{stat}'
                    for stat in ['mean', 'std']] if use_contour else []) + \
                   ['sherd_orientation']:
            if key not in results:
                results[key] = 0
    
    return results


def full_analysis(folder_path, scan_dpi=1200, analyze_inclusions=True, analyze_voids=True,
                  analyze_core_periphery=True, use_blob=True, use_contour=True,
                  interleave_columns=False, file_formats=None, save_csv=True, output_filename=None,
                  enhance_contrast=True, clahe_clip=2.0, clahe_grid=(8, 8),
                  channels=('B', 'G', 'R'), combine_mode='union', vote_min=2,
                  void_intensity_max=60.0, inclusion_pop_min=25.0):
    """
    Comprehensive analysis of all ceramic sherds in a directory with both blob and contour detection.

    This function processes all images in a directory using both detection methods and provides
    complete size, orientation, color, and morphological analysis for archaeological research.

    Parameters
    ----------
    folder_path : str
        Path to folder containing ceramic images
    scan_dpi : int, optional
        Scan resolution in dots per inch (default: 1200)
    analyze_inclusions : bool, optional
        Whether to analyze inclusions (default: True)
    analyze_voids : bool, optional
        Whether to analyze voids (default: True)
    analyze_core_periphery : bool, optional
        Whether to perform core-periphery color analysis for firing atmosphere
        interpretation. This is computationally intensive. (default: True)
    use_blob : bool, optional
        Whether to use blob detection method (default: True)
    use_contour : bool, optional
        Whether to use contour detection method (default: True)
    interleave_columns : bool, optional
        Whether to reorder columns so blob/contour variants of the same metric
        are placed side-by-side. (default: False)
    file_formats : list, optional
        List of file extensions to process (default: ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'])
    save_csv : bool, optional
        Whether to automatically save results as CSV (default: True)
    output_filename : str, optional
        Custom filename for CSV output (default: auto-generated based on folder name)
    enhance_contrast : bool, optional
        If True (default), apply CLAHE before running blob/contour detection.
        With the default multi-channel ``channels`` setting, CLAHE is applied
        to each channel independently inside the detectors; with
        ``channels=('L',)`` it falls back to the legacy single-pass L\* CLAHE
        on the masked image.  Set to False to disable entirely.
    clahe_clip : float, optional
        CLAHE clip limit when ``enhance_contrast=True`` (default: 2.0).
    clahe_grid : tuple of int, optional
        CLAHE tile grid size when ``enhance_contrast=True`` (default: (8, 8)).
    channels : tuple of str, optional
        Image channels passed to both detectors.  Default ``('B', 'G', 'R')``
        runs detection on each BGR channel and combines the results so
        inclusions that only contrast in one channel are still picked up.
        Set ``channels=('L',)`` to recover the pre-multi-channel L\*-only
        behavior.  See ``analyze_single_sherd`` for the full description.
    combine_mode : {'union', 'vote'}, optional
        How to combine per-channel detections (default: ``'union'``).
        Pools detections without requiring cross-channel agreement,
        catching monochromatic features (e.g. red-brown grog only
        visible in B) that voting would drop.  Noise rejection is
        handled by ``inclusion_pop_min`` instead.  See
        ``analyze_single_sherd`` for the full rationale and the
        precision/recall data.
    vote_min : int, optional
        Minimum number of channels that must agree under ``combine_mode='vote'``
        (default: 2 of 3 BGR channels).  Ignored under the default
        ``combine_mode='union'``.
    void_intensity_max : float in 0..255, optional
        Brightness gate for void detection in both detectors (default: 60).
        See ``analyze_single_sherd`` for the full description; lower this
        for stricter voids, raise for low-contrast scans.
    inclusion_pop_min : float in 0..255, optional
        Minimum "pop" (how much an inclusion visually stands out
        against the surrounding paste) for a candidate to be kept,
        applied to both detectors (default: 25).  Computed as the
        max-across-BGR absolute difference between the candidate's
        core disc mean and a surrounding annulus mean on the raw
        (pre-CLAHE) image — see ``analyze_single_sherd`` for the
        formal definition and full rationale.  Primary noise rejection
        mechanism under ``combine_mode='union'``.  Set to 0 to
        disable, raise (30–35) for very uniform paste or tighter
        precision, lower (15–20) for subtle features in fine-grained
        fabrics.

    Returns
    -------
    pandas.DataFrame
        Comprehensive DataFrame containing:
        - Blob detection results (blob_inclusion_*, blob_void_*)
        - Contour detection results (contour_inclusion_*, contour_void_*)
        - Orientation analysis (inclusion_orientation_*, sherd_orientation)
        - Color analysis (inclusion_color_*, sherd_color_*)
        - Density and percentage calculations
        - Processing status and metadata

    Notes
    -----
    Output includes dual detection methods:

    **Blob Detection** - Better for round, circular features:
    - Good for: quartz grains, rounded temper, spherical voids
    - Metrics: blob_inclusion_count, blob_inclusion_total_area_cm2, etc.

    **Contour Detection** - Better for irregular, angular features:
    - Good for: angular rock fragments, irregular voids, elongated inclusions
    - Metrics: contour_inclusion_count, contour_inclusion_total_area_cm2, etc.

    All area measurements are in cm², densities in features per cm².
    """
    if file_formats is None:
        file_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
    
    # Find all image files in the directory
    folder_path = Path(folder_path)
    image_files = []
    
    for ext in file_formats:
        image_files.extend(list(folder_path.rglob(f'*.{ext}')))
        image_files.extend(list(folder_path.rglob(f'*.{ext.upper()}')))
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return pd.DataFrame()
    
    print(f"Starting full analysis of {len(image_files)} ceramic sherds...")
    print(f"Using scan DPI: {scan_dpi}")
    methods = [m for m, on in [('Blob', use_blob), ('Contour', use_contour)] if on]
    print(f"Detection methods: {' + '.join(methods) if methods else 'None'}")
    print(f"Analyzing inclusions: {analyze_inclusions}")
    print(f"Analyzing voids: {analyze_voids}")
    print("-" * 50)
    
    # Process each image
    results_list = []
    successful_count = 0
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {image_path.name} ({i+1}/{len(image_files)})")
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  Warning: Could not load image")
                continue
            
            # Analyze sherd using comprehensive function with selected detection methods
            result = analyze_single_sherd(
                image,
                scan_dpi=scan_dpi,
                analyze_inclusions=analyze_inclusions,
                analyze_voids=analyze_voids,
                analyze_core_periphery=analyze_core_periphery,
                use_blob=use_blob,
                use_contour=use_contour,
                enhance_contrast=enhance_contrast,
                clahe_clip=clahe_clip,
                clahe_grid=clahe_grid,
                channels=channels,
                combine_mode=combine_mode,
                vote_min=vote_min,
                void_intensity_max=void_intensity_max,
                inclusion_pop_min=inclusion_pop_min,
            )
            
            # Add filename and path information
            result['filename'] = image_path.name
            result['file_path'] = str(image_path)
            result['scan_dpi'] = scan_dpi
            
            if result['analysis_status'] == 'success':
                successful_count += 1
                # Print brief summary
                parts = []
                if use_blob:
                    parts.append(f"Blob: {result.get('blob_inclusion_count', 0)} inclusions")
                if use_contour:
                    parts.append(f"Contour: {result.get('contour_inclusion_count', 0)} inclusions")
                print(f"  Success - {', '.join(parts)}")
            else:
                print(f"  ❌ Failed: {result['analysis_status']}")
            
            results_list.append(result)
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            # Add error entry
            error_result = {
                'filename': image_path.name,
                'file_path': str(image_path),
                'scan_dpi': scan_dpi,
                'analysis_status': f'error: {str(e)}'
            }
            results_list.append(error_result)
    
    # Convert to DataFrame
    if not results_list:
        print("No results to process")
        return pd.DataFrame()
        
    df = pd.DataFrame(results_list)
    
    # Column ordering: always put metadata first; optionally interleave blob/contour
    primary_cols = ['filename', 'file_path', 'scan_dpi', 'analysis_status', 'sherd_area_cm2']
    if interleave_columns:
        df = _interleave_method_columns(df, primary_cols)
    else:
        df = _prioritize_columns(df, primary_cols)

    # Print summary
    print("-" * 50)
    print(f"Full Analysis Complete!")
    print(f"Total sherds processed: {len(df)}")
    print(f"Successful analyses: {successful_count}")
    print(f"Failed analyses: {len(df) - successful_count}")

    if successful_count > 0:
        success_df = df[df['analysis_status'] == 'success']
        if use_blob:
            print(f"Average inclusions (blob method): {success_df['blob_inclusion_count'].mean():.1f}")
        if use_contour:
            print(f"Average inclusions (contour method): {success_df['contour_inclusion_count'].mean():.1f}")
    
    # Save CSV if requested
    if save_csv:
        if output_filename is None:
            output_filename = f"amacfa_full_analysis_{folder_path.name}_{scan_dpi}dpi.csv"
        
        output_path = folder_path / output_filename
        df.to_csv(output_path, index=False)
        print(f"💾 Results saved to: {output_path}")
    
    return df


def size_count_summary(folder_path, fileformat='jpeg', scan_dpi=1200, use_blob=True, use_contour=True,
                       interleave_columns=False):
    """
    Analysis of inclusions and voids using blob and/or contour detection.

    Parameters
    ----------
    folder_path : str
        Path to folder containing ceramic images
    fileformat : str, optional
        Image file format to process (default: 'jpeg')
    scan_dpi : int, optional
        Scan resolution in dots per inch (default: 1200)
        Valid range: 150-2400 DPI
    use_blob : bool, optional
        Whether to use blob detection (default: True)
    use_contour : bool, optional
        Whether to use contour detection (default: True)
    interleave_columns : bool, optional
        Whether to reorder columns so blob/contour variants of the same metric
        are placed side-by-side. (default: False)

    Returns
    -------
    pandas.DataFrame
        Summary statistics for each ceramic's inclusions and voids.
        All area measurements are in cm². Column names are prefixed with
        'blob_' or 'contour_' to indicate detection method.
    """
    from .detection import sherd_mask, apply_mask, sherd_blobs, contour_detection

    if scan_dpi < 150 or scan_dpi > 2400:
        print(f"Warning: scan_dpi {scan_dpi} is outside recommended range (150-2400).")

    dpcm = scan_dpi * 0.3937
    path_strs = [str(path) for path in Path(folder_path).rglob(f'*.{fileformat}')]

    results_list = []

    for path in path_strs:
        im = cv2.imread(path)
        if im is None:
            print(f"Warning: Could not load image {path}")
            continue

        name = path.rsplit(sep='/')[-1]
        row = {'Name': name}

        # Mask the sherd (auto-cropped)
        try:
            mask, crop, _bc = sherd_mask(im, scan_dpi=scan_dpi)
            masked_im = apply_mask(im, mask, crop)
            # mask is 3-channel by default; collapse to 2D so we don't count
            # each sherd pixel three times.
            mask_2d = mask[:, :, 0] if mask.ndim == 3 else mask
            sherd_area_cm2 = np.sum(mask_2d > 0) / (dpcm ** 2)
            row['sherd_area_cm2'] = sherd_area_cm2
        except Exception as e:
            print(f"Warning: Could not mask {name}: {e}")
            continue

        # BLOB DETECTION
        if use_blob:
            try:
                blobs_light, blobs_dark = sherd_blobs(masked_im, scan_dpi=scan_dpi)

                # Inclusion (light blob) stats
                inc_areas = [np.pi * ((b.size / 2) / dpcm) ** 2 for b in blobs_light]
                row['blob_inclusion_count'] = len(inc_areas)
                if inc_areas:
                    row['blob_inclusion_max_area_cm2'] = np.max(inc_areas)
                    row['blob_inclusion_mean_area_cm2'] = np.mean(inc_areas)
                    row['blob_inclusion_min_area_cm2'] = np.min(inc_areas)
                    row['blob_inclusion_std_area_cm2'] = np.std(inc_areas)
                    row['blob_inclusion_median_area_cm2'] = np.median(inc_areas)
                    row['blob_inclusion_25pct_cm2'] = np.percentile(inc_areas, 25)
                    row['blob_inclusion_75pct_cm2'] = np.percentile(inc_areas, 75)
                    row['blob_inclusion_total_area_cm2'] = np.sum(inc_areas)
                    row['blob_inclusion_skewness'] = float(skew(inc_areas)) if len(inc_areas) >= 3 else 0.0
                    row['blob_inclusion_cv'] = float(np.std(inc_areas) / np.mean(inc_areas)) if np.mean(inc_areas) > 0 else 0.0
                else:
                    for k in ['max', 'mean', 'min', 'std', 'median', '25pct', '75pct', 'total']:
                        row[f'blob_inclusion_{k}_area_cm2'] = 0
                    row['blob_inclusion_skewness'] = 0
                    row['blob_inclusion_cv'] = 0

                # Void (dark blob) stats
                void_areas = [np.pi * ((b.size / 2) / dpcm) ** 2 for b in blobs_dark]
                row['blob_void_count'] = len(void_areas)
                if void_areas:
                    row['blob_void_max_area_cm2'] = np.max(void_areas)
                    row['blob_void_mean_area_cm2'] = np.mean(void_areas)
                    row['blob_void_min_area_cm2'] = np.min(void_areas)
                    row['blob_void_std_area_cm2'] = np.std(void_areas)
                    row['blob_void_median_area_cm2'] = np.median(void_areas)
                    row['blob_void_25pct_cm2'] = np.percentile(void_areas, 25)
                    row['blob_void_75pct_cm2'] = np.percentile(void_areas, 75)
                    row['blob_void_total_area_cm2'] = np.sum(void_areas)
                    row['blob_void_skewness'] = float(skew(void_areas)) if len(void_areas) >= 3 else 0.0
                    row['blob_void_cv'] = float(np.std(void_areas) / np.mean(void_areas)) if np.mean(void_areas) > 0 else 0.0
                else:
                    for k in ['max', 'mean', 'min', 'std', 'median', '25pct', '75pct', 'total']:
                        row[f'blob_void_{k}_area_cm2'] = 0
                    row['blob_void_skewness'] = 0
                    row['blob_void_cv'] = 0

                # Density
                if sherd_area_cm2 > 0:
                    row['blob_inclusion_density_per_cm2'] = row['blob_inclusion_count'] / sherd_area_cm2
                    row['blob_void_density_per_cm2'] = row['blob_void_count'] / sherd_area_cm2
                else:
                    row['blob_inclusion_density_per_cm2'] = 0
                    row['blob_void_density_per_cm2'] = 0

            except Exception as e:
                print(f"Warning: Blob detection failed for {name}: {e}")

        # CONTOUR DETECTION
        if use_contour:
            try:
                contour_results = contour_detection(masked_im, scan_dpi=scan_dpi, debug_mode=False)
                inc_contours = contour_results.get('inclusions', [])
                void_contours = contour_results.get('voids', [])

                # Inclusion stats from contours
                inc_areas = [cv2.contourArea(c) / (dpcm ** 2) for c in inc_contours]
                row['contour_inclusion_count'] = len(inc_areas)
                if inc_areas:
                    row['contour_inclusion_max_area_cm2'] = np.max(inc_areas)
                    row['contour_inclusion_mean_area_cm2'] = np.mean(inc_areas)
                    row['contour_inclusion_min_area_cm2'] = np.min(inc_areas)
                    row['contour_inclusion_std_area_cm2'] = np.std(inc_areas)
                    row['contour_inclusion_median_area_cm2'] = np.median(inc_areas)
                    row['contour_inclusion_25pct_cm2'] = np.percentile(inc_areas, 25)
                    row['contour_inclusion_75pct_cm2'] = np.percentile(inc_areas, 75)
                    row['contour_inclusion_total_area_cm2'] = np.sum(inc_areas)
                    row['contour_inclusion_skewness'] = float(skew(inc_areas)) if len(inc_areas) >= 3 else 0.0
                    row['contour_inclusion_cv'] = float(np.std(inc_areas) / np.mean(inc_areas)) if np.mean(inc_areas) > 0 else 0.0
                else:
                    for k in ['max', 'mean', 'min', 'std', 'median', '25pct', '75pct', 'total']:
                        row[f'contour_inclusion_{k}_area_cm2'] = 0
                    row['contour_inclusion_skewness'] = 0
                    row['contour_inclusion_cv'] = 0

                # Void stats from contours
                void_areas = [cv2.contourArea(c) / (dpcm ** 2) for c in void_contours]
                row['contour_void_count'] = len(void_areas)
                if void_areas:
                    row['contour_void_max_area_cm2'] = np.max(void_areas)
                    row['contour_void_mean_area_cm2'] = np.mean(void_areas)
                    row['contour_void_min_area_cm2'] = np.min(void_areas)
                    row['contour_void_std_area_cm2'] = np.std(void_areas)
                    row['contour_void_median_area_cm2'] = np.median(void_areas)
                    row['contour_void_25pct_cm2'] = np.percentile(void_areas, 25)
                    row['contour_void_75pct_cm2'] = np.percentile(void_areas, 75)
                    row['contour_void_total_area_cm2'] = np.sum(void_areas)
                    row['contour_void_skewness'] = float(skew(void_areas)) if len(void_areas) >= 3 else 0.0
                    row['contour_void_cv'] = float(np.std(void_areas) / np.mean(void_areas)) if np.mean(void_areas) > 0 else 0.0
                else:
                    for k in ['max', 'mean', 'min', 'std', 'median', '25pct', '75pct', 'total']:
                        row[f'contour_void_{k}_area_cm2'] = 0
                    row['contour_void_skewness'] = 0
                    row['contour_void_cv'] = 0

                # Density
                if sherd_area_cm2 > 0:
                    row['contour_inclusion_density_per_cm2'] = row['contour_inclusion_count'] / sherd_area_cm2
                    row['contour_void_density_per_cm2'] = row['contour_void_count'] / sherd_area_cm2
                else:
                    row['contour_inclusion_density_per_cm2'] = 0
                    row['contour_void_density_per_cm2'] = 0

            except Exception as e:
                print(f"Warning: Contour detection failed for {name}: {e}")

        results_list.append(row)

    data = pd.DataFrame(results_list)
    if not data.empty:
        primary_cols = ['Name', 'sherd_area_cm2']
        if interleave_columns:
            data = _interleave_method_columns(data, primary_cols)
        else:
            data = _prioritize_columns(data, primary_cols)
    return data


def void_counter(image, scan_dpi=1200):
    """
    Calculate the number and area of void spaces within a ceramic sherd.
    
    Parameters
    ----------
    image : numpy.ndarray
        Image of a scanned sherd
    scan_dpi : int, optional
        Scan resolution in dots per inch (default: 1200)
        Valid range: 150-2400 DPI
        
    Returns
    -------
    tuple
        (list of void areas in cm², number of voids found)
    """
    # Validate DPI input
    if scan_dpi < 150 or scan_dpi > 2400:
        print(f"Warning: scan_dpi {scan_dpi} is outside recommended range (150-2400). Results may be unreliable.")
    
    # Convert DPI to dots per centimeter
    dpcm = scan_dpi * 0.3937
    
    areas = []
    # Convert to L* channel (CIELAB lightness)
    bw = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 0]

    # Convert image to binary
    _, binary_img = cv2.threshold(bw, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    # Find all the contours in the thresholded image
    contours_inc , _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours_inc):
      # Calculate the area of each contour in pixels
        area_pixels = cv2.contourArea(c)

      # Ignore contours that are too small or too large
        # Convert pixel thresholds to be DPI-aware
        min_area_pixels = int((0.01 * dpcm) ** 2)  # ~0.01cm² minimum
        max_area_pixels = int((10.0 * dpcm) ** 2)   # ~10cm² maximum
        
        if area_pixels < min_area_pixels or area_pixels > max_area_pixels:
            continue
            
        # Convert area from pixels² to cm²
        area_cm2 = area_pixels / (dpcm**2)
        areas.append(area_cm2)
        
    return areas, len(areas)


def contour_counter(image, scan_dpi=1200):
    """
    Calculate the number and area of contours within a ceramic sherd.
    
    Parameters
    ----------
    image : numpy.ndarray
        Image of a scanned sherd
    scan_dpi : int, optional
        Scan resolution in dots per inch (default: 1200)
        Valid range: 150-2400 DPI
        
    Returns
    -------
    tuple
        (list of contour areas in cm², number of contours found)
    """
    # Validate DPI input
    if scan_dpi < 150 or scan_dpi > 2400:
        print(f"Warning: scan_dpi {scan_dpi} is outside recommended range (150-2400). Results may be unreliable.")
    
    # Convert DPI to dots per centimeter
    dpcm = scan_dpi * 0.3937
    
    areas = []
    # Convert to L* channel (CIELAB lightness)
    bw = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 0]

    # Convert image to binary
    _, binary_img = cv2.threshold(bw, 125, 255, cv2.THRESH_BINARY)

    # Find all the contours in the thresholded image
    contours_inc , _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours_inc):
      # Calculate the area of each contour in pixels
        area_pixels = cv2.contourArea(c)

      # Ignore contours that are too small or too large
        # Convert pixel thresholds to be DPI-aware
        min_area_pixels = int((0.01 * dpcm) ** 2)  # ~0.01cm² minimum
        max_area_pixels = int((10.0 * dpcm) ** 2)   # ~10cm² maximum
        
        if area_pixels < min_area_pixels or area_pixels > max_area_pixels:
            continue
            
        # Convert area from pixels² to cm²
        area_cm2 = area_pixels / (dpcm**2)
        areas.append(area_cm2)
        
    return areas, len(areas)

def sacredsquare(og_img, blobs):
    """
    Extract squares representing inclusions using blob detection results.
    
    Parameters
    ----------
    og_img : numpy.ndarray
        Original image from which blobs were detected
    blobs : list
        Blob KeyPoint objects from blob detection
        
    Returns
    -------
    tuple
        (sorted list of [(left_vertex, right_vertex), size], 
         image with squares drawn over blobs)
    """
    lst = []
    im = og_img.copy()
    
    for b in blobs:
        c = b.pt #center
        d = b.size #diameter
        l = math.sqrt(((((d)**2))/2))
        v1 = (int(c[0]-.5*l), int(c[1]-.5*l)) #figure out the vertices
        v2 = (int(c[0]+.5*l), int(c[1]+.5*l))
        lst.append([(v1,v2), d])
        
    for i in lst:
        cv2.rectangle(im, i[0][0], i[0][1], (0, 255, 255), 3)

    lst.sort(key = lambda x : x[1], reverse=True)
    return lst, im


def inclusion_colors(image, inclusion_list):
    """
    Extract color information for each inclusion using k-means clustering.

    Parameters
    ----------
    image : numpy.ndarray
        Masked scanned image for color analysis (BGR format)
    inclusion_list : list
        List of inclusions and their locations from sacredsquare

    Returns
    -------
    list
        List of CIELAB colors for dominant 3 colors of each inclusion
        [[L*, a*, b*], [L*, a*, b*], [L*, a*, b*]] ordered by frequency
        L* = lightness (0-100), a* = green-red, b* = blue-yellow
    """
    lab_lst = []

    for i in inclusion_list:
        # Extract inclusion region
        try:
            inc_img = image[i[0][0][1]:i[0][1][1], i[0][0][0]:i[0][1][0]]

            if inc_img.size == 0:
                lab_lst.append([[0, 128, 128], [0, 128, 128], [0, 128, 128]])
                continue

            Z = inc_img.reshape((-1, 3))
            Z = np.float32(Z)

            # K-means clustering to find dominant colors
            criteria = (cv2.TERM_CRITERIA_EPS, 10, 0.1)
            K = 3
            ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            center = np.uint8(center)

            # Sort centers by frequency (most common first)
            new_label = np.array([x[0] for x in label]).T
            Y = list(np.unique(new_label, return_counts=True)[1])
            Z_center = [x for _, x in sorted(zip(Y, center), key=lambda x: x[0], reverse=True)]

            # Convert BGR cluster centers to Lab
            lab_colors = []
            for idx in range(3):
                if idx < len(Z_center):
                    # Create a small swatch to convert color
                    swatch = np.zeros((1, 1, 3), np.uint8)
                    swatch[0, 0] = Z_center[idx]  # BGR values
                    swatch_lab = cv2.cvtColor(swatch, cv2.COLOR_BGR2LAB)
                    lab_colors.append(list(map(float, swatch_lab[0, 0])))
                else:
                    # Duplicate last color if fewer than 3 clusters
                    lab_colors.append(lab_colors[-1] if lab_colors else [0, 128, 128])

            lab_lst.append(lab_colors)

        except Exception as e:
            print(f"Warning: Could not process inclusion colors: {e}")
            lab_lst.append([[0, 128, 128], [0, 128, 128], [0, 128, 128]])

    return lab_lst

def inclusion_colors_from_contours(image, contours):
    """
    Extract color information for contour-detected inclusions using contour masks.

    Unlike inclusion_colors() which uses rectangular bounding boxes,
    this function masks each inclusion to its exact contour boundary,
    avoiding paste/matrix color contamination.

    Parameters
    ----------
    image : numpy.ndarray
        Masked scanned image (BGR format)
    contours : list
        List of contour arrays from contour_detection['inclusions']

    Returns
    -------
    list
        List of CIELAB colors for dominant 3 colors of each inclusion
        [[L*, a*, b*], [L*, a*, b*], [L*, a*, b*]] ordered by frequency
    """
    lab_lst = []
    for c in contours:
        try:
            x, y, w, h = cv2.boundingRect(c)
            if w == 0 or h == 0:
                lab_lst.append([[0, 128, 128]] * 3)
                continue

            # Crop to bounding box for efficiency
            roi = image[y:y+h, x:x+w]

            # Create contour mask in ROI coordinates
            mask = np.zeros((h, w), dtype=np.uint8)
            shifted = c - np.array([x, y])
            cv2.drawContours(mask, [shifted], -1, 255, -1)

            # Extract only pixels inside the contour
            pixels = roi[mask > 0]
            if len(pixels) == 0:
                lab_lst.append([[0, 128, 128]] * 3)
                continue

            Z = np.float32(pixels)
            criteria = (cv2.TERM_CRITERIA_EPS, 10, 0.1)
            K = min(3, len(pixels))  # can't have more clusters than pixels
            _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            center = np.uint8(center)

            # Sort centers by frequency (most common first)
            unique, counts = np.unique(label.flatten(), return_counts=True)
            order = np.argsort(-counts)
            sorted_centers = center[order]

            # Convert BGR → Lab
            lab_colors = []
            for idx in range(3):
                if idx < len(sorted_centers):
                    swatch = np.zeros((1, 1, 3), np.uint8)
                    swatch[0, 0] = sorted_centers[idx]
                    swatch_lab = cv2.cvtColor(swatch, cv2.COLOR_BGR2LAB)
                    lab_colors.append(list(map(float, swatch_lab[0, 0])))
                else:
                    lab_colors.append(lab_colors[-1] if lab_colors else [0, 128, 128])

            lab_lst.append(lab_colors)
        except Exception as e:
            print(f"Warning: Could not process contour inclusion colors: {e}")
            lab_lst.append([[0, 128, 128]] * 3)

    return lab_lst


def inclusion_orientation(image, scan_dpi, contour_result=None):
    """
    Estimate orientations of inclusions and voids.

    Parameters
    ----------
    image : numpy.ndarray
        Masked image of a scanned sherd (background zeroed).
    scan_dpi : int
        Scan resolution in dots per inch.
    contour_result : dict, optional
        Output dict from ``contour_detection()``.  When provided the
        already-filtered inclusion and void contours are used directly,
        avoiding a redundant re-detection pass.  When None the function
        falls back to independent threshold-based detection.

    Returns
    -------
    tuple
        ``(inclusion_angles, void_angles)`` — lists of integer angles in
        degrees derived from the minimum-area bounding rectangle of each
        detected feature.

    Notes
    -----
    Orientation will depend on how sherds were scanned — trends will be
    in modal angles, not true measures.  Use ``inclusion_orientation2``
    to correct for the sherd's own principal axis.
    """
    inclusion_angles = []
    void_angles = []

    if contour_result is not None:
        # Use the already-detected, shape-filtered contours from contour_detection()
        inc_contours  = contour_result['inclusions']
        void_contours = contour_result['voids']
    else:
        # Fallback: re-detect from the image directly
        bw = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 0]  # L* channel
        _, binary_inc  = cv2.threshold(bw, 125, 255, cv2.THRESH_BINARY)
        _, binary_void = cv2.threshold(bw, 0,   255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        raw_inc,  _ = cv2.findContours(binary_inc,  cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        raw_void, _ = cv2.findContours(binary_void, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        pixels_per_mm = scan_dpi / 25.4
        min_area = int((0.1  * pixels_per_mm) ** 2)
        max_area = int((25.0 * pixels_per_mm) ** 2)

        all_raw = list(raw_inc) + list(raw_void)
        if all_raw:
            min_sherd_size = cv2.contourArea(
                max(all_raw, key=cv2.contourArea)) * 0.8
        else:
            min_sherd_size = float('inf')

        inc_contours  = [c for c in raw_inc  if min_area < cv2.contourArea(c) < max_area
                         and cv2.contourArea(c) < min_sherd_size]
        void_contours = [c for c in raw_void if min_area < cv2.contourArea(c) < max_area
                         and cv2.contourArea(c) < min_sherd_size]

    def _angle_from_rect(contour):
        rect = cv2.minAreaRect(contour)
        w, h  = int(rect[1][0]), int(rect[1][1])
        angle = int(rect[2])
        return (90 - angle) if w < h else -angle

    inclusion_angles = [_angle_from_rect(c) for c in inc_contours]
    void_angles      = [_angle_from_rect(c) for c in void_contours]

    return inclusion_angles, void_angles


def _long_axis_angle(rect):
    """
    Return the orientation of the LONG axis of a minAreaRect in [0, 180).

    cv2.minAreaRect returns (center, (w, h), theta) where theta is the
    angle of the *width* vector from the horizontal x-axis.
    Convention (consistent across OpenCV versions):
      - If w >= h the width IS the long side  → long axis at theta
      - If w <  h the height is the long side → long axis at theta + 90°
    Result is normalised to [0°, 180°) so that 0° = horizontal, 90° = vertical.
    """
    w, h  = rect[1][0], rect[1][1]
    center = rect[0]
    theta = rect[2]
    # Determine the angle of the long axis
    angle = theta if w >= h else (theta + 90.0)
    angle = angle % 180.0
    # Convert angle to radians for vector calculation
    angle_rad = np.deg2rad(angle)
    # Unit vector along the long axis
    long_axis_vector = (np.cos(angle_rad), np.sin(angle_rad))
    # Return center, unit vector, and angle
    return center, long_axis_vector, angle


def inclusion_orientation2(image, scan_dpi, contour_result=None, sherd_contour=None):
    """
    Enhanced orientation analysis that corrects angles relative to the sherd's
    own principal axis.

    Parameters
    ----------
    image : numpy.ndarray
        Masked image of a scanned sherd (background zeroed).
    scan_dpi : int
        Scan resolution in dots per inch.
    contour_result : dict, optional
        Output dict from ``contour_detection()``.  When provided the
        already-filtered inclusion and void contours are used directly,
        avoiding a redundant re-detection pass.  When None the function
        falls back to independent threshold-based detection.
    sherd_contour : numpy.ndarray or None, optional
        The ``best_contour`` returned by ``sherd_mask()``.  When provided
        the sherd's principal axis is derived from this exact contour via
        ``cv2.minAreaRect`` — the same geometry that determined the mask and
        crop.  This is the authoritative sherd orientation because the mask
        bounding box is what orients the entire sherd in the pipeline.
        When None the function derives the sherd orientation by
        thresholding the masked image (fallback — less reliable on
        dark-matrix sherds).

    Returns
    -------
    tuple
        ``(inclusion_angles, void_angles, sherd_angle)`` — angle lists in
        degrees corrected for the sherd's principal axis orientation, plus
        the sherd angle itself.
    """
    inclusion_angles = []
    void_angles      = []

    # --- Sherd principal axis ---------------------------------------------------
    # Prefer the exact contour that sherd_mask() used (passed in as sherd_contour)
    # because its minAreaRect defines the crop/mask geometry — i.e., the same
    # bounding box that orients the whole sherd in the pipeline.
    # minAreaRect angle is purely rotational and coordinate-translation-independent,
    # so using a contour from image_cropped space is valid here.
    impangle = 0.0

    if sherd_contour is not None:
        rect     = cv2.minAreaRect(sherd_contour)
        _ , _,impangle = _long_axis_angle(rect)
    else:
        # Fallback: re-derive sherd outline by thresholding the masked image.
        # Threshold at 1 so the full sherd silhouette is captured (background = 0
        # exactly; any pixel > 0 belongs to the sherd).
        bw = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 0]  # L* channel
        _, binary_sherd = cv2.threshold(bw, 1, 255, cv2.THRESH_BINARY)
        contours_sherd, _ = cv2.findContours(binary_sherd, cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_NONE)
        if len(contours_sherd) > 0:
            _sc      = max(contours_sherd, key=cv2.contourArea)
            _ , _,impangle = _long_axis_angle(cv2.minAreaRect(_sc))

    if contour_result is not None:
        # Use the already-detected, shape-filtered contours from contour_detection()
        inc_contours  = contour_result['inclusions']
        void_contours = contour_result['voids']
    else:
        # Fallback: re-detect from the image directly
        bw = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 0]  # L* channel
        _, binary_inc  = cv2.threshold(bw, 125, 255, cv2.THRESH_BINARY)
        _, binary_void = cv2.threshold(bw, 0,   255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        raw_inc,  _ = cv2.findContours(binary_inc,  cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        raw_void, _ = cv2.findContours(binary_void, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        pixels_per_mm = scan_dpi / 25.4
        min_area = int((0.1  * pixels_per_mm) ** 2)
        max_area = int((25.0 * pixels_per_mm) ** 2)

        all_raw = list(raw_inc) + list(raw_void)
        if all_raw:
            min_sherd_size = cv2.contourArea(
                max(all_raw, key=cv2.contourArea)) * 0.8
        else:
            min_sherd_size = float('inf')

        inc_contours  = [c for c in raw_inc  if min_area < cv2.contourArea(c) < max_area
                         and cv2.contourArea(c) < min_sherd_size]
        void_contours = [c for c in raw_void if min_area < cv2.contourArea(c) < max_area
                         and cv2.contourArea(c) < min_sherd_size]

    def _corrected_angle(contour, imp):
        """
        Long-axis angle of an inclusion contour corrected for the sherd axis.
        Both angles use _long_axis_angle() for consistency.
        Result is in (-90°, 90°] — the magnitude of angular difference from
        the sherd's principal axis (0° = parallel, ±90° = orthogonal).
        """
        _,_,inc_angle = _long_axis_angle(cv2.minAreaRect(contour))
        # Difference mapped to (-90, 90] so that ±90° are treated identically
        # (orientation is symmetric: 0° and 180° are the same direction).
        return (inc_angle - imp + 90.0) % 180.0 - 90.0

    inclusion_angles = [_corrected_angle(c, impangle) for c in inc_contours]
    void_angles      = [_corrected_angle(c, impangle) for c in void_contours]

    return inclusion_angles, void_angles, impangle


def sherd_color_analysis(image, mask=None, crop=None):
    """
    Analyze color properties of a single ceramic sherd image using CIELAB.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array (BGR format).  Pass the original (un-cropped) image;
        the function will slice it using ``crop`` when provided.
    mask : numpy.ndarray, optional
        Mask to apply (already cropped to the sherd region when ``crop`` is
        given).  If None, a mask is generated automatically via ``sherd_mask``.
    crop : tuple or None, optional
        ``(y1, y2, x1, x2)`` crop rectangle as returned by ``sherd_mask``.
        When provided together with ``mask``, the image is sliced to this
        region so its dimensions match the (already-cropped) mask.

    Returns
    -------
    dict
        Dictionary containing CIELAB color values:
        - mean_l: L* lightness (0-100)
        - mean_a: a* green-red axis (-128 to +127)
        - mean_b: b* blue-yellow axis (-128 to +127)
    """
    if mask is None:
        mask, crop, _bc = sherd_mask(image)
        y1, y2, x1, x2 = crop[:4]
        image = _pad_crop(image[y1:y2, x1:x2], crop)
    elif crop is not None:
        # Caller supplied a mask that is already cropped; slice the image to match
        y1, y2, x1, x2 = crop[:4]
        image = _pad_crop(image[y1:y2, x1:x2], crop)

    # Ensure mask is in the correct format for cv2.mean
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    try:
        # Calculate average color using mask (image and mask are the same cropped size)
        avg_color = cv2.mean(image, mask=mask)

        # Create color swatch for conversion
        swatch = np.zeros((1, 1, 3), np.uint8)
        swatch[0, 0, 0] = int(avg_color[0])  # B
        swatch[0, 0, 1] = int(avg_color[1])  # G
        swatch[0, 0, 2] = int(avg_color[2])  # R

        # Convert to CIELAB
        swatch_lab = cv2.cvtColor(swatch, cv2.COLOR_BGR2LAB)
        lab_values = swatch_lab[0, 0]

        return {
            'mean_l': float(lab_values[0]),
            'mean_a': float(lab_values[1]),
            'mean_b': float(lab_values[2])
        }

    except Exception as e:
        print(f"Warning: Could not analyze color: {e}")
        return {'mean_l': 0, 'mean_a': 128, 'mean_b': 128}


def sherd_color_summary(folder_path, scan_dpi=1200, use_blob=True, use_contour=True,
                        analyze_core_periphery=True, interleave_columns=False):
    """
    Provide summary of color aspects of sherds in CIELAB colorspace.

    Parameters
    ----------
    folder_path : str
        Path to folder containing ceramic images
    scan_dpi : int, optional
        Scan resolution in dots per inch (default: 1200)
    use_blob : bool, optional
        Whether to analyze inclusion colors using blob detection (default: True)
    use_contour : bool, optional
        Whether to analyze inclusion colors using contour detection (default: True)
    analyze_core_periphery : bool, optional
        Whether to perform core-periphery color analysis for firing atmosphere
        interpretation. This is computationally intensive. (default: True)
    interleave_columns : bool, optional
        Whether to reorder columns so blob/contour variants of the same metric
        are placed side-by-side. (default: False)

    Returns
    -------
    pandas.DataFrame
        Summary statistics for each ceramic's color and inclusion colors in CIELAB.
        L* = lightness (0-100), a* = green-red, b* = blue-yellow.
        Columns are prefixed with 'blob_' or 'contour_' to indicate method.
    """
    path_strs = [str(path) for path in Path(folder_path).rglob('*.jpeg')]

    results_list = []

    for path in path_strs:
        name = path.rsplit(sep='/')[-1]
        im = cv2.imread(path)

        if im is None:
            print(f"Warning: Could not load image {name}")
            continue

        row = {'Name': name}

        try:
            mask, crop, _bc = sherd_mask(im, scan_dpi=scan_dpi)
            y1, y2, x1, x2 = crop[:4]
            im_crop = _pad_crop(im[y1:y2, x1:x2], crop)
            masked_im = apply_mask(im, mask, crop)

            # Sherd average color in CIELAB (im_crop and mask share the same dimensions)
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
            avg_color = cv2.mean(im_crop, mask=gray_mask)
            swatch = np.zeros((1, 1, 3), np.uint8)
            swatch[0, 0, 0] = int(avg_color[0])  # B
            swatch[0, 0, 1] = int(avg_color[1])  # G
            swatch[0, 0, 2] = int(avg_color[2])  # R
            swatch_lab = cv2.cvtColor(swatch, cv2.COLOR_BGR2LAB)
            lab_vals = swatch_lab[0][0]
            row['sherd_avg_color_l'] = float(lab_vals[0])
            row['sherd_avg_color_a'] = float(lab_vals[1])
            row['sherd_avg_color_b'] = float(lab_vals[2])

            # BLOB method inclusion colors
            if use_blob:
                try:
                    blobs_light, _ = sherd_blobs(masked_im, scan_dpi)
                    blob_lst, _ = sacredsquare(masked_im, blobs_light)
                    blob_lab = inclusion_colors(masked_im, blob_lst)
                    # Extract dominant color stats
                    if blob_lab:
                        dominant = np.array([inc[0] for inc in blob_lab], dtype=float)
                        row['blob_inclusion_color_l_mean'] = np.mean(dominant[:, 0])
                        row['blob_inclusion_color_a_mean'] = np.mean(dominant[:, 1])
                        row['blob_inclusion_color_b_mean'] = np.mean(dominant[:, 2])
                        row['blob_inclusion_color_l_std'] = np.std(dominant[:, 0])
                        row['blob_inclusion_color_a_std'] = np.std(dominant[:, 1])
                        row['blob_inclusion_color_b_std'] = np.std(dominant[:, 2])
                        if len(dominant) > 1:
                            dists = [np.linalg.norm(dominant[i] - dominant[j])
                                     for i in range(len(dominant)) for j in range(i+1, len(dominant))]
                            row['blob_inclusion_color_diversity'] = np.mean(dists)
                        else:
                            row['blob_inclusion_color_diversity'] = 0
                    else:
                        row['blob_inclusion_color_l_mean'] = 0
                        row['blob_inclusion_color_a_mean'] = 128
                        row['blob_inclusion_color_b_mean'] = 128
                        row['blob_inclusion_color_l_std'] = 0
                        row['blob_inclusion_color_a_std'] = 0
                        row['blob_inclusion_color_b_std'] = 0
                        row['blob_inclusion_color_diversity'] = 0
                except Exception as e:
                    print(f"Warning: Blob color analysis failed for {name}: {e}")
                    row['blob_inclusion_color_l_mean'] = 0
                    row['blob_inclusion_color_a_mean'] = 128
                    row['blob_inclusion_color_b_mean'] = 128
                    row['blob_inclusion_color_l_std'] = 0
                    row['blob_inclusion_color_a_std'] = 0
                    row['blob_inclusion_color_b_std'] = 0
                    row['blob_inclusion_color_diversity'] = 0

            # CONTOUR method inclusion colors
            if use_contour:
                try:
                    contour_results = contour_detection(masked_im, scan_dpi=scan_dpi)
                    inc_contours = contour_results.get('inclusions', [])
                    contour_lab = inclusion_colors_from_contours(masked_im, inc_contours)
                    # Extract dominant color stats
                    if contour_lab:
                        dominant = np.array([inc[0] for inc in contour_lab], dtype=float)
                        row['contour_inclusion_color_l_mean'] = np.mean(dominant[:, 0])
                        row['contour_inclusion_color_a_mean'] = np.mean(dominant[:, 1])
                        row['contour_inclusion_color_b_mean'] = np.mean(dominant[:, 2])
                        row['contour_inclusion_color_l_std'] = np.std(dominant[:, 0])
                        row['contour_inclusion_color_a_std'] = np.std(dominant[:, 1])
                        row['contour_inclusion_color_b_std'] = np.std(dominant[:, 2])
                        if len(dominant) > 1:
                            dists = [np.linalg.norm(dominant[i] - dominant[j])
                                     for i in range(len(dominant)) for j in range(i+1, len(dominant))]
                            row['contour_inclusion_color_diversity'] = np.mean(dists)
                        else:
                            row['contour_inclusion_color_diversity'] = 0
                    else:
                        row['contour_inclusion_color_l_mean'] = 0
                        row['contour_inclusion_color_a_mean'] = 128
                        row['contour_inclusion_color_b_mean'] = 128
                        row['contour_inclusion_color_l_std'] = 0
                        row['contour_inclusion_color_a_std'] = 0
                        row['contour_inclusion_color_b_std'] = 0
                        row['contour_inclusion_color_diversity'] = 0
                except Exception as e:
                    print(f"Warning: Contour color analysis failed for {name}: {e}")
                    row['contour_inclusion_color_l_mean'] = 0
                    row['contour_inclusion_color_a_mean'] = 128
                    row['contour_inclusion_color_b_mean'] = 128
                    row['contour_inclusion_color_l_std'] = 0
                    row['contour_inclusion_color_a_std'] = 0
                    row['contour_inclusion_color_b_std'] = 0
                    row['contour_inclusion_color_diversity'] = 0

            # Core-periphery color analysis (firing atmosphere) - optional due to computation cost
            if analyze_core_periphery:
                try:
                    core_periph = extract_core_periphery_colors(masked_im, mask, scan_dpi)
                    # Core color
                    if core_periph['core_lab']:
                        row['core_color_l'] = core_periph['core_lab'][0]
                        row['core_color_a'] = core_periph['core_lab'][1]
                        row['core_color_b'] = core_periph['core_lab'][2]
                    else:
                        row['core_color_l'] = 0
                        row['core_color_a'] = 128
                        row['core_color_b'] = 128
                    # Inner margin color
                    if core_periph['inner_margin_lab']:
                        row['inner_margin_color_l'] = core_periph['inner_margin_lab'][0]
                        row['inner_margin_color_a'] = core_periph['inner_margin_lab'][1]
                        row['inner_margin_color_b'] = core_periph['inner_margin_lab'][2]
                    else:
                        row['inner_margin_color_l'] = 0
                        row['inner_margin_color_a'] = 128
                        row['inner_margin_color_b'] = 128
                    # Outer margin color
                    if core_periph['outer_margin_lab']:
                        row['outer_margin_color_l'] = core_periph['outer_margin_lab'][0]
                        row['outer_margin_color_a'] = core_periph['outer_margin_lab'][1]
                        row['outer_margin_color_b'] = core_periph['outer_margin_lab'][2]
                    else:
                        row['outer_margin_color_l'] = 0
                        row['outer_margin_color_a'] = 128
                        row['outer_margin_color_b'] = 128
                    # Per-zone atmosphere classifications
                    row['core_atmosphere'] = core_periph['core_atmosphere']
                    row['inner_margin_atmosphere'] = core_periph['inner_margin_atmosphere']
                    row['outer_margin_atmosphere'] = core_periph['outer_margin_atmosphere']
                    # Gradient and interpretation
                    row['core_periphery_gradient'] = core_periph['color_gradient']
                    row['firing_interpretation'] = core_periph['firing_interpretation']
                    row['margin_symmetry'] = core_periph['margin_symmetry']
                except Exception as e:
                    print(f"Warning: Core-periphery color analysis failed for {name}: {e}")
                    for key in ['core_color_l', 'inner_margin_color_l', 'outer_margin_color_l']:
                        row[key] = 0
                    for key in ['core_color_a', 'core_color_b', 'inner_margin_color_a',
                               'inner_margin_color_b', 'outer_margin_color_a', 'outer_margin_color_b']:
                        row[key] = 128
                    row['core_atmosphere'] = 'analysis_failed'
                    row['inner_margin_atmosphere'] = 'analysis_failed'
                    row['outer_margin_atmosphere'] = 'analysis_failed'
                    row['core_periphery_gradient'] = 0
                    row['firing_interpretation'] = 'analysis_failed'
                    row['margin_symmetry'] = 'analysis_failed'

        except Exception as e:
            print(f"Warning: Could not process colors for {name}: {e}")
            row['sherd_avg_color_l'] = 0
            row['sherd_avg_color_a'] = 128
            row['sherd_avg_color_b'] = 128

        results_list.append(row)

    data = pd.DataFrame(results_list)
    if not data.empty:
        primary_cols = ['Name', 'sherd_avg_color_l', 'sherd_avg_color_a', 'sherd_avg_color_b']
        if interleave_columns:
            data = _interleave_method_columns(data, primary_cols)
        else:
            data = _prioritize_columns(data, primary_cols)
    return data


# --- Helper functions for core-periphery analysis (module level for performance) ---

def _extract_dominant_color_lab(pixels):
    """Extract dominant color using k-means clustering, return as CIELAB."""
    if len(pixels) < 10:
        return None
    try:
        pixel_data = pixels.reshape((-1, 3)).astype(np.float32)
        # Filter extreme values
        brightness = np.mean(pixel_data, axis=1)
        valid = (brightness > 30) & (brightness < 240)
        filtered = pixel_data[valid] if np.sum(valid) >= 10 else pixel_data
        # K-means for dominant color
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(filtered, 1, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        # Convert to CIELAB
        pixel = np.zeros((1, 1, 3), np.uint8)
        pixel[0, 0] = centers[0].astype(np.uint8)
        lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
        return [float(lab[0, 0, 0]), float(lab[0, 0, 1]), float(lab[0, 0, 2])]
    except Exception:
        return None


def _classify_zone_atmosphere(lab):
    """Classify a ceramic zone's firing atmosphere from its CIELAB color.

    Uses the full CIELAB color space — not just lightness — to determine
    the oxidation state.  The a* channel (red-green axis) is the primary
    indicator: oxidized iron (Fe2O3, hematite) shifts a* strongly positive
    (red/brown), while reduced iron (FeO) and preserved carbon remain
    chromatically neutral (gray/black).  Lightness then disambiguates
    chromatically neutral cases: a light + neutral fabric must be a
    low-iron clay fired oxidizing (kaolinitic/calcareous whitewares such
    as American Southwest Anasazi wares), because reduced firing on an
    iron-bearing clay would darken it.

    All values in OpenCV 8-bit encoding (L: 0-255, a/b: 128 = neutral).
    Internal thresholds operate on the real CIELAB scale (L*: 0-100,
    a*/b*: ±127); the conversions below match Photoshop/standard CIELAB.
    """
    L, a, b = lab[0], lab[1], lab[2]
    L_real = L * 100.0 / 255.0   # 8-bit L → standard L* (0..100)
    a_c = a - 128                # centered: + = red, - = green
    b_c = b - 128                # centered: + = yellow, - = blue

    # 1. CARBONACEOUS: very dark + completely neutral = unburnt organics
    #    L* < 29 (raw L < 75); chromaticity within ±8 of neutral
    if L < 75 and abs(a_c) < 8 and abs(b_c) < 8:
        return 'carbonaceous'

    # 2. OXIDIZED: meaningful red shift from iron oxidation
    #    a* > +8 (raw a > 136) — perceptible redness from Fe2O3/hematite
    if a_c > 8:
        return 'oxidized'

    # 3. INCOMPLETE_OXIDATION: slight warmth developing
    #    a* +2 to +8 — partial Fe oxidation, brown/buff tones
    if a_c > 2:
        return 'incomplete_oxidation'

    # 4. OXIDIZED_LOW_IRON: light + chromatically near-neutral
    #    L* >= 65 (raw L >= 166), |a*| <= 4, |b*| <= 10.  Low-iron clays
    #    (kaolinite, marl, calcareous) fired oxidizing develop no red
    #    shift because there is little Fe to oxidize; the result is
    #    white/cream/buff (Anasazi whitewares, calcareous Mediterranean
    #    wares).  Distinct from 'reduced' because reduction on an
    #    iron-bearing clay would darken the matrix — a light neutral
    #    fabric is petrologically inconsistent with reduction.
    if L_real >= 65 and abs(a_c) <= 4 and abs(b_c) <= 10:
        return 'oxidized_low_iron'

    # 5. REDUCED: dark + neutral chromaticity = reduced iron or
    #    preserved carbon on an iron-bearing clay
    return 'reduced'


def _define_core_periphery_regions(binary_mask, scan_dpi=1200):
    """
    Define core and periphery regions using distance transform percentiles.

    Core = innermost 20% of sherd (pixels at ≥ 80th percentile distance from edge)
    Periphery = near-edge band (pixels between 10th and 25th percentile distance)
    """
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    dist_values = dist_transform[binary_mask > 0]

    if len(dist_values) == 0:
        z = np.zeros_like(binary_mask)
        return z, z, dist_transform

    p80 = np.percentile(dist_values, 80)
    p25 = np.percentile(dist_values, 25)
    p10 = np.percentile(dist_values, 10)

    core_mask = (dist_transform >= p80).astype(np.uint8) * 255
    periphery_mask = ((dist_transform >= p10) & (dist_transform <= p25)).astype(np.uint8) * 255

    # Zero out background pixels that happen to be at distance 0
    core_mask[binary_mask == 0] = 0
    periphery_mask[binary_mask == 0] = 0

    return core_mask, periphery_mask, dist_transform


def _split_margins(binary_mask, periphery_mask):
    """Split periphery into inner vs outer margin bands using rotation-aware splitting.

    The split is perpendicular to the sherd's long axis (via minAreaRect),
    restricted to the middle 60% along the long axis.  This handles sherds
    scanned at arbitrary angles — unlike the old axis-aligned approach.

    The contour is always derived from *binary_mask* via ``findContours``
    to guarantee the center and angle are in the same coordinate system as
    the masks (avoids offset bugs when the mask has been auto-cropped).

    Parameters
    ----------
    binary_mask : numpy.ndarray
        Binary sherd mask (uint8, 0/255).
    periphery_mask : numpy.ndarray
        Binary periphery region mask.
    """
    h, w = periphery_mask.shape

    # --- Get sherd contour and orientation (always from binary_mask) ---
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        inner = periphery_mask.copy()
        outer = periphery_mask.copy()
        inner[:, w // 2:] = 0
        outer[:, :w // 2] = 0
        return inner, outer
    sherd_contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(sherd_contour)
    center = np.array(rect[0], dtype=np.float64)          # (cx, cy)
    _, _, angle_deg = _long_axis_angle(rect)                     # long-axis angle [0, 180)
    angle_rad = np.deg2rad(angle_deg)

    # Unit vectors: long axis and short (perpendicular) axis
    long_axis  = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    short_axis = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

    # --- Project periphery pixels onto both axes ---
    ys, xs = np.nonzero(periphery_mask)
    if len(ys) == 0:
        return np.zeros_like(periphery_mask), np.zeros_like(periphery_mask)

    offsets = np.column_stack([xs.astype(np.float64) - center[0],
                               ys.astype(np.float64) - center[1]])  # (N, 2)

    proj_long  = offsets @ long_axis    # projection onto long axis
    proj_short = offsets @ short_axis   # projection onto short axis

    # --- Restrict to middle 60% along long axis ---
    long_min, long_max = proj_long.min(), proj_long.max()
    long_range = long_max - long_min
    if long_range == 0:
        return np.zeros_like(periphery_mask), np.zeros_like(periphery_mask)

    mid_lo = long_min + 0.2 * long_range   # skip bottom 20%
    mid_hi = long_max - 0.2 * long_range   # skip top 20%
    in_middle = (proj_long >= mid_lo) & (proj_long <= mid_hi)

    # --- Split on short axis: negative = inner margin, positive = outer margin ---
    inner_mask = np.zeros_like(periphery_mask)
    outer_mask = np.zeros_like(periphery_mask)

    inner_sel = in_middle & (proj_short < 0)
    outer_sel = in_middle & (proj_short >= 0)

    inner_mask[ys[inner_sel], xs[inner_sel]] = 255
    outer_mask[ys[outer_sel], xs[outer_sel]] = 255

    return inner_mask, outer_mask


def extract_core_periphery_colors(masked_image, mask, scan_dpi=1200):
    """
    Extract ceramic paste colors from core vs margin regions using distance transform.

    Uses distance transform to define the structural core (innermost 20% by distance
    from edges) and analyzes color differences between core and margin regions.

    Archaeological Significance
    ---------------------------
    The a* channel (red-green axis) is the primary indicator of iron oxidation
    state in ceramics.  Oxidized iron (Fe2O3, hematite) produces red/brown
    colors (high a*), while reduced iron and preserved carbon remain
    chromatically neutral (gray/black).  L* alone cannot distinguish a dark
    oxidized ceramic from a reduced one.

    Each zone is independently classified as *oxidized*, *reduced*,
    *incomplete_oxidation*, or *carbonaceous* using the full CIELAB color,
    then a whole-sherd ``firing_interpretation`` is derived from the
    combination.

    Parameters
    ----------
    masked_image : numpy.ndarray
        Masked ceramic sherd image (BGR format)
    mask : numpy.ndarray
        Binary mask defining ceramic boundaries
    scan_dpi : int, optional
        Scan resolution (default: 1200)

    Returns
    -------
    dict
        - 'core_lab': [L*, a*, b*] for ceramic core
        - 'inner_margin_lab': [L*, a*, b*] for inner margin
        - 'outer_margin_lab': [L*, a*, b*] for outer margin
        - 'core_atmosphere': per-zone classification
        - 'inner_margin_atmosphere': per-zone classification
        - 'outer_margin_atmosphere': per-zone classification
        - 'color_gradient': Max Delta-E between regions
        - 'firing_interpretation': Archaeological assessment
        - 'margin_symmetry': symmetric/symmetric_transitional/asymmetric
        - 'core_pixels', 'inner_margin_pixels', 'outer_margin_pixels': Counts
    """
    # Initialize results
    result = {
        'core_lab': None,
        'inner_margin_lab': None,
        'outer_margin_lab': None,
        'core_atmosphere': 'insufficient_data',
        'inner_margin_atmosphere': 'insufficient_data',
        'outer_margin_atmosphere': 'insufficient_data',
        'color_gradient': 0.0,
        'firing_interpretation': 'insufficient_data',
        'margin_symmetry': 'insufficient_data',
        'core_pixels': 0,
        'inner_margin_pixels': 0,
        'outer_margin_pixels': 0,
    }

    # Prepare binary mask
    if len(mask.shape) > 2:
        binary_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        binary_mask = mask
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    # Check minimum area
    if np.sum(binary_mask > 0) < 1000:
        return result

    # Define regions
    core_mask, periphery_mask, _dt = _define_core_periphery_regions(binary_mask, scan_dpi)
    inner_mask, outer_mask = _split_margins(binary_mask, periphery_mask)

    # Check sufficient core pixels
    if np.sum(core_mask > 0) < 50:
        return result

    # Extract pixels (handle grayscale)
    img = masked_image if len(masked_image.shape) == 3 else cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)

    core_pixels = img[core_mask > 0]
    inner_pixels = img[inner_mask > 0]
    outer_pixels = img[outer_mask > 0]

    result['core_pixels'] = len(core_pixels)
    result['inner_margin_pixels'] = len(inner_pixels)
    result['outer_margin_pixels'] = len(outer_pixels)

    # Extract dominant colors
    result['core_lab'] = _extract_dominant_color_lab(core_pixels)
    result['inner_margin_lab'] = _extract_dominant_color_lab(inner_pixels)
    result['outer_margin_lab'] = _extract_dominant_color_lab(outer_pixels)

    # Classify each zone and determine firing interpretation
    if result['core_lab'] and result['inner_margin_lab'] and result['outer_margin_lab']:
        core = np.array(result['core_lab'])
        inner = np.array(result['inner_margin_lab'])
        outer = np.array(result['outer_margin_lab'])

        # Per-zone atmosphere classification
        core_atm = _classify_zone_atmosphere(core)
        inner_atm = _classify_zone_atmosphere(inner)
        outer_atm = _classify_zone_atmosphere(outer)
        result['core_atmosphere'] = core_atm
        result['inner_margin_atmosphere'] = inner_atm
        result['outer_margin_atmosphere'] = outer_atm

        # Delta-E calculations (supplementary continuous measure)
        de_core_inner = np.linalg.norm(core - inner)
        de_core_outer = np.linalg.norm(core - outer)
        de_inner_outer = np.linalg.norm(inner - outer)
        max_de = max(de_core_inner, de_core_outer, de_inner_outer)
        result['color_gradient'] = float(max_de)

        # --- Firing interpretation from zone comparison ---
        # 'oxidized_low_iron' groups with oxidized for sandwich/differential
        # logic: a low-iron oxidized fabric is chemically oxidized, just
        # without iron to redden it.
        oxidized_set = {'oxidized', 'incomplete_oxidation', 'oxidized_low_iron'}
        reduced_set = {'reduced', 'carbonaceous'}

        # ΔE stability floor: when the max color gradient across zones is
        # below this perceptual threshold, treat per-zone bin differences
        # as noise (a sherd whose three zones differ by <15 ΔE in 8-bit
        # CIELAB-Euclidean is visually uniform).  Without this, a uniform
        # grey/light sherd whose core a* lands at +2.0 (just under the
        # incomplete_oxidation cutoff of +2) reads as 'sandwich_oxidized'
        # even though all three zones are within JND of each other.
        UNIFORM_DE_FLOOR = 15.0
        zones_uniform_by_color = (max_de < UNIFORM_DE_FLOOR)

        all_same = (core_atm == inner_atm == outer_atm)
        if all_same or zones_uniform_by_color:
            # When the zones are colorimetrically uniform but bins
            # disagree (boundary-flicker case), re-classify from the
            # whole-sherd mean LAB so the interpretation reflects the
            # actual fabric rather than the noisier of two adjacent bins.
            if zones_uniform_by_color and not all_same:
                mean_lab = (core + inner + outer) / 3.0
                resolved_atm = _classify_zone_atmosphere(mean_lab)
            else:
                resolved_atm = core_atm
            if resolved_atm in oxidized_set and resolved_atm != 'incomplete_oxidation':
                result['firing_interpretation'] = 'fully_oxidized'
            elif resolved_atm == 'reduced':
                result['firing_interpretation'] = 'fully_reduced'
            elif resolved_atm == 'carbonaceous':
                result['firing_interpretation'] = 'carbonaceous_throughout'
            else:  # incomplete_oxidation
                result['firing_interpretation'] = 'incomplete_oxidation'
        else:
            core_reduced = core_atm in reduced_set
            both_margins_oxidized = (inner_atm in oxidized_set and outer_atm in oxidized_set)
            core_oxidized = core_atm in oxidized_set
            both_margins_reduced = (inner_atm in reduced_set and outer_atm in reduced_set)

            if core_reduced and both_margins_oxidized:
                result['firing_interpretation'] = 'sandwich_oxidized_margins'
            elif core_oxidized and both_margins_reduced:
                result['firing_interpretation'] = 'reduced_margins_oxidized_core'
            elif inner_atm != outer_atm:
                result['firing_interpretation'] = 'differential_margins'
            elif core_reduced and (inner_atm in oxidized_set or outer_atm in oxidized_set):
                result['firing_interpretation'] = 'partial_oxidation'
            else:
                result['firing_interpretation'] = 'mixed_atmosphere'

        # --- Margin symmetry ---
        if inner_atm == outer_atm:
            result['margin_symmetry'] = 'symmetric'
        elif inner_atm in oxidized_set and outer_atm in oxidized_set:
            result['margin_symmetry'] = 'symmetric_transitional'
        else:
            result['margin_symmetry'] = 'asymmetric'

    return result


def analyze_inclusion_angularity(contours, scan_dpi=1200):
    """
    Analyze geometric angularity and roundness of inclusion contours.

    This function uses polygon approximation and roundness metrics to classify
    inclusions into the six standard sedimentological roundness categories
    established by Muller (1964) and Powers (1953), as applied to ceramic
    petrography by Stienstra (1986).

    Parameters
    ----------
    contours : list
        List of contour objects from cv2.findContours()
    scan_dpi : int, optional
        Scan resolution for size-aware filtering (default: 1200)

    Returns
    -------
    dict
        Dictionary containing:
        - 'angularity_scores': list of angularity scores (0-1, higher = more angular)
        - 'vertex_counts': list of vertex counts for each inclusion
        - 'roundness_ratios': list of roundness ratios (0-1, higher = more round)
        - 'roundness_classifications': list of Muller/Powers roundness classes
          ('very_angular', 'angular', 'sub_angular', 'sub_rounded', 'rounded', 'well_rounded')
        - 'approx_polygons': list of approximated polygon contours
        - 'summary_stats': dict with aggregate statistics
        - 'pca_metrics': dict with metrics formatted for PCA analysis

    Notes
    -----
    Roundness classification follows the Powers (1953) / Muller (1964) scale
    adapted for automated circularity measurement:
    - Very angular:  circularity < 0.17
    - Angular:       0.17 <= circularity < 0.25
    - Sub-angular:   0.25 <= circularity < 0.35
    - Sub-rounded:   0.35 <= circularity < 0.49
    - Rounded:       0.49 <= circularity < 0.70
    - Well-rounded:  circularity >= 0.70
    """
    # DPI-aware size filtering
    dpcm = scan_dpi * 0.3937
    min_area_pixels = int((0.005 * dpcm) ** 2)  # 0.005cm² minimum (50μm diameter)
    max_area_pixels = int((4.0 * dpcm) ** 2)     # 4cm² maximum (reasonable inclusion limit)
    
    # Initialize result containers
    angularity_scores = []
    vertex_counts = []
    roundness_ratios = []
    roundness_classifications = []
    areas_cm2 = []
    approx_polygons = []
    
    for contour in contours:
        area_pixels = cv2.contourArea(contour)
        
        # Skip contours that are too small or too large
        if area_pixels < min_area_pixels or area_pixels > max_area_pixels:
            continue
            
        # Convert area to cm²
        area_cm2 = area_pixels / (dpcm ** 2)
        areas_cm2.append(area_cm2)
        
        # Calculate perimeter for polygon approximation
        perimeter = cv2.arcLength(contour, True)
        
        # Skip degenerate contours
        if perimeter < 1.0:
            continue
        
        # Polygon approximation - find simplified polygon that represents the shape
        # More conservative epsilon (2%) to preserve important vertices while removing noise
        epsilon = 0.02 * perimeter
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
        approx_polygons.append(approx_polygon)
        vertex_count = len(approx_polygon)
        vertex_counts.append(vertex_count)
        
        # Calculate multiple angularity metrics
        
        # 1. Vertex-based angularity score
        # More vertices generally = more complex/angular shape
        # Normalize by logarithmic scale since vertex count grows non-linearly with complexity
        vertex_score = min(1.0, math.log(max(3, vertex_count)) / math.log(12))  # 12 vertices = maximum complexity
        
        # 2. Perimeter-to-area ratio (complexity measure)
        # More angular shapes have higher perimeter relative to area
        theoretical_circle_perimeter = 2 * math.sqrt(math.pi * area_pixels)
        complexity_ratio = perimeter / theoretical_circle_perimeter
        complexity_score = min(1.0, (complexity_ratio - 1.0) / 2.0)  # Normalize: circle=0, complex shapes approach 1
        
        # 3. Convex hull ratio (solidity) - inverted for angularity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area_pixels / hull_area
            # Invert solidity: angular shapes have lower solidity, round shapes higher
            hull_angularity = 1.0 - solidity
        else:
            hull_angularity = 0.0
        
        # 4. Calculate roundness ratio (Feret diameter approach)
        # Fit minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circle_area = math.pi * radius * radius
        if circle_area > 0:
            roundness = area_pixels / circle_area
        else:
            roundness = 0.0
        roundness_ratios.append(roundness)
        
        # Combine metrics into final angularity score
        # Weight the different components based on archaeological significance
        angularity_score = (
            0.35 * vertex_score +           # Vertex complexity (most important)
            0.30 * complexity_score +       # Perimeter complexity  
            0.25 * hull_angularity +        # Shape irregularity
            0.10 * (1.0 - roundness)        # Non-roundness
        )
        
        angularity_scores.append(angularity_score)
        
        # Roundness classification using Powers (1953) / Muller (1964) scale
        # Boundaries based on Wadell roundness index adapted for circularity
        if roundness < 0.17:
            classification = 'very_angular'
        elif roundness < 0.25:
            classification = 'angular'
        elif roundness < 0.35:
            classification = 'sub_angular'
        elif roundness < 0.49:
            classification = 'sub_rounded'
        elif roundness < 0.70:
            classification = 'rounded'
        else:
            classification = 'well_rounded'

        roundness_classifications.append(classification)
    
    # Muller/Powers roundness class names in order from most angular to most rounded
    _ROUNDNESS_CLASSES = [
        'very_angular', 'angular', 'sub_angular',
        'sub_rounded', 'rounded', 'well_rounded',
    ]

    # Calculate summary statistics
    if len(angularity_scores) > 0:
        mean_angularity = np.mean(angularity_scores)
        std_angularity = np.std(angularity_scores)
        mean_vertices = np.mean(vertex_counts)
        mean_roundness = np.mean(roundness_ratios)

        # Count each roundness class
        total_count = len(roundness_classifications)
        class_counts = {c: roundness_classifications.count(c) for c in _ROUNDNESS_CLASSES}
        class_proportions = {
            c: class_counts[c] / total_count if total_count > 0 else 0.0
            for c in _ROUNDNESS_CLASSES
        }

        # Create summary statistics
        summary_stats = {
            'total_inclusions_analyzed': total_count,
            'mean_angularity': mean_angularity,
            'std_angularity': std_angularity,
            'mean_vertex_count': mean_vertices,
            'mean_roundness': mean_roundness,
        }
        summary_stats.update({f'{c}_count': class_counts[c] for c in _ROUNDNESS_CLASSES})
        summary_stats.update({f'{c}_proportion': class_proportions[c] for c in _ROUNDNESS_CLASSES})

        # Create PCA-ready metrics (standardized variables for multivariate analysis)
        pca_metrics = {
            'angularity_mean': mean_angularity,
            'angularity_std': std_angularity,
            'vertex_count_mean': mean_vertices,
            'roundness_mean': mean_roundness,
            'geometric_complexity': mean_angularity * mean_vertices / 10.0,
        }
        pca_metrics.update({f'{c}_ratio': class_proportions[c] for c in _ROUNDNESS_CLASSES})

    else:
        # No valid inclusions found
        summary_stats = {
            'total_inclusions_analyzed': 0,
            'mean_angularity': 0.0,
            'std_angularity': 0.0,
            'mean_vertex_count': 0.0,
            'mean_roundness': 0.0,
        }
        summary_stats.update({f'{c}_count': 0 for c in _ROUNDNESS_CLASSES})
        summary_stats.update({f'{c}_proportion': 0.0 for c in _ROUNDNESS_CLASSES})

        pca_metrics = {
            'angularity_mean': 0.0,
            'angularity_std': 0.0,
            'vertex_count_mean': 0.0,
            'roundness_mean': 0.0,
            'geometric_complexity': 0.0,
        }
        pca_metrics.update({f'{c}_ratio': 0.0 for c in _ROUNDNESS_CLASSES})

    return {
        'angularity_scores': angularity_scores,
        'vertex_counts': vertex_counts,
        'roundness_ratios': roundness_ratios,
        'roundness_classifications': roundness_classifications,
        'approx_polygons': approx_polygons,
        'areas_cm2': areas_cm2,
        'summary_stats': summary_stats,
        'pca_metrics': pca_metrics
    }


def analyze_orientation_for_pca(orientation_angles):
    """
    Convert circular orientation data to PCA-compatible metrics.
    
    Handles the statistical challenges of circular data by computing:
    1. Vector components (sine/cosine) - preserves directional information
    2. Circular statistical measures (mean direction, concentration)  
    3. Fabric strength indicators (preferred orientation vs randomness)
    
    Parameters
    ----------
    orientation_angles : list or array-like
        List of angles in degrees (0-360° or -180° to +180°)
        
    Returns
    -------
    dict
        Dictionary containing PCA-ready orientation metrics:
        - orientation_strength: How strongly oriented the fabric is (0=random, 1=perfectly aligned)
        - mean_orientation_x: X-component of mean orientation vector  
        - mean_orientation_y: Y-component of mean orientation vector
        - orientation_concentration: Circular concentration parameter (higher = more aligned)
        - orientation_uniformity: Measure of how evenly distributed angles are
        - dominant_orientation_deg: Main orientation direction in degrees
        - orientation_bimodality: Whether fabric shows two preferred orientations
        
    Notes
    -----
    This approach solves the "circular data problem" for PCA by:
    1. Converting angles to unit vectors, avoiding 0°/360° discontinuity
    2. Computing vector statistics that are PCA-compatible
    3. Providing archaeological interpretations (fabric strength, preferred orientations)
    
    For ceramic analysis:
    - High orientation_strength = strong fabric, deliberate manufacturing technique
    - Low orientation_strength = random fabric, hand-building or poor clay preparation
    - Bimodality = cross-hatched or woven fabric structure
    """
    
    if not orientation_angles or len(orientation_angles) == 0:
        return {
            'orientation_strength': 0.0,
            'mean_orientation_x': 0.0,
            'mean_orientation_y': 0.0, 
            'orientation_concentration': 0.0,
            'orientation_uniformity': 0.0,
            'dominant_orientation_deg': 0.0,
            'orientation_bimodality': 0.0
        }
    
    # Convert angles to numpy array and ensure they're in radians
    angles_deg = np.array(orientation_angles)
    angles_rad = np.deg2rad(angles_deg)
    
    # Method 1: Vector approach - convert to unit vectors
    # This avoids the circular discontinuity problem
    x_components = np.cos(angles_rad)
    y_components = np.sin(angles_rad)
    
    # Calculate mean vector components (PCA-ready)
    mean_x = np.mean(x_components)
    mean_y = np.mean(y_components)
    
    # Calculate resultant vector length (orientation strength)
    # Length = 1 means perfect alignment, length = 0 means random
    resultant_length = np.sqrt(mean_x**2 + mean_y**2)
    orientation_strength = resultant_length
    
    # Calculate mean orientation direction
    mean_orientation_rad = np.arctan2(mean_y, mean_x)
    dominant_orientation_deg = np.rad2deg(mean_orientation_rad) % 360
    
    # Method 2: Circular statistics
    # Calculate circular concentration (von Mises parameter estimate)
    if resultant_length < 0.53:
        # Low concentration estimate
        concentration = 2 * resultant_length + resultant_length**3 + (5 * resultant_length**5) / 6
    elif resultant_length < 0.85:
        # Medium concentration estimate
        concentration = -0.4 + 1.39 * resultant_length + 0.43 / (1 - resultant_length)
    else:
        # High concentration estimate
        concentration = 1 / (resultant_length**3 - 4 * resultant_length**2 + 3 * resultant_length)
    
    # Method 3: Uniformity test
    # Calculate how evenly distributed the angles are (Kuiper's test approximation)
    # Sort angles and calculate spacings
    sorted_angles = np.sort(angles_deg)
    n = len(sorted_angles)
    
    if n > 1:
        # Calculate spacings between consecutive angles
        spacings = np.diff(sorted_angles) 
        # Add the wraparound spacing
        spacings = np.append(spacings, 360 + sorted_angles[0] - sorted_angles[-1])
        
        # Calculate expected spacing for uniform distribution
        expected_spacing = 360 / n
        
        # Calculate uniformity measure (1 = perfectly uniform, 0 = completely clustered)
        spacing_variance = np.var(spacings)
        max_possible_variance = (360**2) / 12  # Variance for maximally non-uniform distribution
        orientation_uniformity = 1.0 - (spacing_variance / max_possible_variance)
        orientation_uniformity = max(0.0, min(1.0, orientation_uniformity))  # Clamp to [0,1]
    else:
        orientation_uniformity = 1.0
    
    # Method 4: Ceramic fabric alignment detection
    # Check for preferred orientations in ceramic inclusions (not textile patterns)
    # Ceramics may show alignment from manufacturing techniques like coiling or wheel throwing
    if n >= 4:  # Lowered threshold - need fewer points for basic pattern detection
        hist, bin_edges = np.histogram(angles_deg, bins=12)  # 30° bins for ceramic analysis
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Count significant peaks (local maxima above background)
        mean_count = np.mean(hist)
        std_count = np.std(hist)
        threshold = mean_count + 0.5 * std_count  # More sensitive peak detection
        
        peak_indices = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > threshold:
                peak_indices.append(i)
        
        # Calculate bimodality based on distribution patterns relevant to ceramic manufacturing
        if len(peak_indices) >= 2:
            # Multiple peaks indicate preferred orientations from manufacturing
            # Score based on how distinct the peaks are
            peak_heights = [hist[i] for i in peak_indices]
            peak_prominence = np.std(peak_heights) / (mean_count + 1)  # Avoid division by zero
            bimodality_score = min(1.0, peak_prominence * 0.5)  # Scale appropriately
        elif len(peak_indices) == 1:
            # Single strong peak = unimodal alignment
            bimodality_score = 0.1
        else:
            # No clear peaks = random orientation
            bimodality_score = 0.0
    else:
        bimodality_score = 0.0
    
    return {
        'orientation_strength': orientation_strength,
        'mean_orientation_x': mean_x,
        'mean_orientation_y': mean_y,
        'orientation_concentration': concentration,
        'orientation_uniformity': orientation_uniformity, 
        'dominant_orientation_deg': dominant_orientation_deg,
        'orientation_bimodality': bimodality_score
    }


def analyze_manufacturing_technique(orientation_metrics, size_metrics, geometric_metrics):
    """
    Identify likely ceramic manufacturing technique based on inclusion patterns.
    
    Based on archaeological research (Berg 2008, Roux & Courty 2005, EXARC 2021):
    - Coiling: wavy/spiral patterns, radial orientations, moderate alignment
    - Wheel throwing: strong horizontal alignment, high uniformity, low bimodality  
    - Slab construction: parallel to walls, moderate-high alignment, clustered joints
    - Pinching: random orientations, low alignment, clustering at stress points
    
    Parameters
    ----------
    orientation_metrics : dict
        Results from analyze_orientation_for_pca()
    size_metrics : dict
        Size distribution metrics
    geometric_metrics : dict 
        Geometric analysis results
        
    Returns
    -------
    dict
        Manufacturing technique analysis with confidence scores
    """
    
    # Extract key orientation metrics
    strength = orientation_metrics.get('orientation_strength', 0)
    uniformity = orientation_metrics.get('orientation_uniformity', 0) 
    bimodality = orientation_metrics.get('orientation_bimodality', 0)
    concentration = orientation_metrics.get('orientation_concentration', 0)
    
    # Extract geometric indicators if available
    angularity = geometric_metrics.get('geometric_angularity_mean', 0.5)
    
    # Initialize technique scores
    techniques = {
        'wheel_thrown': 0.0,
        'coiled': 0.0,
        'slab_built': 0.0, 
        'pinched': 0.0
    }
    
    # WHEEL THROWING signatures (gradual scoring):
    # - Strong horizontal alignment (high strength)
    # - Very uniform distribution (high uniformity)  
    # - Low bimodality (single preferred orientation)
    wheel_strength_score = max(0, min(1, (strength - 0.3) / 0.4))  # Scale 0.3-0.7 → 0-1
    wheel_uniformity_score = max(0, min(1, (uniformity - 0.5) / 0.3))  # Scale 0.5-0.8 → 0-1
    wheel_bimodal_score = max(0, min(1, (0.4 - bimodality) / 0.4))  # Scale 0.4-0 → 0-1
    techniques['wheel_thrown'] = (wheel_strength_score + wheel_uniformity_score + wheel_bimodal_score) / 3 * 0.9
    
    # COILING signatures (gradual scoring):
    # - Moderate alignment with spiral/wavy patterns
    # - Moderate bimodality (radial + spiral orientations)
    # - Some concentration but not extreme
    coil_strength_score = 1.0 - abs(strength - 0.5) * 2  # Peak at 0.5, decline toward 0/1
    coil_bimodal_score = min(1, bimodality * 3)  # Scale bimodality 0-0.33 → 0-1
    coil_concentration_score = min(1, concentration * 2)  # Scale concentration 0-0.5 → 0-1
    techniques['coiled'] = max(0, (coil_strength_score + coil_bimodal_score + coil_concentration_score) / 3 * 0.8)
        
    # SLAB CONSTRUCTION signatures (gradual scoring):
    # - Moderate to high alignment parallel to walls
    # - Low to moderate uniformity (joints create variation)
    # - Angular inclusions clustered at joints
    slab_strength_score = max(0, min(1, (strength - 0.2) / 0.5))  # Scale 0.2-0.7 → 0-1
    slab_uniformity_score = max(0, min(1, (0.7 - uniformity) / 0.4))  # Lower uniformity = higher score
    slab_angularity_score = max(0, min(1, (angularity - 0.4) / 0.4))  # Scale 0.4-0.8 → 0-1
    techniques['slab_built'] = (slab_strength_score + slab_uniformity_score + slab_angularity_score) / 3 * 0.7
    
    # PINCHING signatures (gradual scoring):
    # - Low overall alignment (random orientations)
    # - Low uniformity (clustering at stress points)
    # - Low bimodality (no preferred directions)
    pinch_strength_score = max(0, min(1, (0.5 - strength) / 0.5))  # Lower strength = higher score
    pinch_uniformity_score = max(0, min(1, (0.6 - uniformity) / 0.6))  # Lower uniformity = higher score
    pinch_bimodal_score = max(0, min(1, (0.3 - bimodality) / 0.3))  # Lower bimodality = higher score
    techniques['pinched'] = (pinch_strength_score + pinch_uniformity_score + pinch_bimodal_score) / 3 * 0.8
    
    # Light normalization to prevent any single technique from dominating
    # but keep reasonable confidence levels
    max_score = max(techniques.values())
    if max_score > 1.0:
        # Only normalize if scores are too high
        for technique in techniques:
            techniques[technique] = techniques[technique] / max_score
    
    # Determine most likely technique and calculate meaningful confidence
    primary_technique = max(techniques.keys(), key=lambda k: techniques[k])
    confidence = techniques[primary_technique]
    
    # Add uncertainty penalty if top techniques are very close (ambiguous results)
    sorted_scores = sorted(techniques.values(), reverse=True)
    if len(sorted_scores) >= 2 and sorted_scores[0] > 0:
        # If top 2 scores are very close, reduce confidence
        score_ratio = sorted_scores[1] / sorted_scores[0] if sorted_scores[0] > 0 else 0
        if score_ratio > 0.8:  # Very close scores = uncertain
            confidence = confidence * 0.7  # Reduce confidence for ambiguous cases
    
    # Archaeological interpretation
    interpretations = {
        'wheel_thrown': "Strong horizontal alignment suggests wheel throwing with centrifugal force organizing inclusions parallel to vessel walls.",
        'coiled': "Radial and spiral orientation patterns indicate coil construction with inclusions following clay manipulation paths.",
        'slab_built': "Parallel alignment with joint markers suggests slab construction with inclusions oriented along building planes.", 
        'pinched': "Random orientation with stress point clustering indicates pinch pot construction with localized clay deformation."
    }
    
    return {
        'technique_scores': techniques,
        'primary_technique': primary_technique,
        'confidence': confidence,
        'interpretation': interpretations.get(primary_technique, "Technique pattern unclear"),
        'archaeological_evidence': f"Based on inclusion orientation analysis used in archaeological ceramic studies (Berg 2008, EXARC 2021)",
        'strength_indicator': strength,
        'pattern_complexity': bimodality + (1 - uniformity) * 0.5
    }