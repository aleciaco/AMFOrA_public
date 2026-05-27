"""
Detection functions for AMACFA+ ceramic analysis.

This module contains functions for detecting sherds, inclusions, and voids
in ceramic scans with enhanced edge detection and blob detection capabilities.
"""

import cv2
import numpy as np
import os
from pathlib import Path

__all__ = [
    'setup_robust_blob_params', 'sherd_mask', 'apply_mask',
    'clahe_enhance', 'super_zorro_cv', 'sherd_blobs',
    'detect_multiple_sherds', 'split_multi_sherd_scan',
    'prepare_multi_sherd_directory', 'contour_detection',
]


def setup_robust_blob_params(image, scan_dpi, blob_type="light", size_params=None):
    """
    Create robust blob detector parameters with adaptive thresholding and optional size overrides.

    This function automatically calculates optimal detection parameters based on image characteristics,
    but allows user-specified size filtering to override the defaults.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image for parameter calculation
    scan_dpi : int
        Scan resolution in dots per inch (150-2400)
    blob_type : str
        Type of blobs to detect:
        - 'light': inclusions (bright features on darker background)
        - 'dark_inclusion': inclusions (dark minerals — ferruginous grains,
          magnetite, biotite, dark grog).  Uses the same adaptive dark
          thresholding as 'dark' but with inclusion-level size limits
          and strict shape filters: circularity >= 0.3, convexity >= 0.7,
          inertia ratio >= 0.35 (~3:1 max elongation).  These ensure only
          high-confidence compact mineral grains are captured, leaving
          irregular dark features to the void detector.
        - 'dark': voids (dark features/pores).  Uses upper-bound shape
          filters (maxCircularity = 0.85, maxConvexity = 0.85) to reject
          near-perfect circles and very smooth convex shapes that are
          almost certainly mineral grains rather than voids.
    size_params : dict, optional
        Size filtering parameters to override defaults. If None, uses:

        **Inclusions (blob_type='light'):**
        - min: 0.1mm (fine silt boundary, Wentworth scale)
        - max: 15mm (very coarse gravel)

        **Voids (blob_type='dark'):**
        - min: 0.25mm (macroscopic voids from organic burnout)
        - max: 15mm (larger voids are likely artifacts)

        For inclusions, provide:
        - 'min_inclusion_area_px': int, minimum area in pixels
        - 'max_inclusion_area_px': int, maximum area in pixels

        For voids, provide:
        - 'min_void_area_px': int, minimum area in pixels
        - 'max_void_area_px': int, maximum area in pixels

        Example for detecting inclusions up to 2cm diameter at 1200 DPI::

            dpcm = 1200 * 0.3937  # ~472 dots per cm
            max_area = int(np.pi * (2.0 / 2 * dpcm) ** 2)  # 2cm diameter
            size_params = {
                'min_inclusion_area_px': 50,
                'max_inclusion_area_px': max_area
            }

    Returns
    -------
    cv2.SimpleBlobDetector_Params
        Optimized parameters for blob detection with adaptive thresholding
    """
    params = cv2.SimpleBlobDetector_Params()
    
    # Calculate comprehensive image statistics for adaptive thresholding
    non_zero_pixels = image[image != 0]
    if len(non_zero_pixels) == 0:
        # Fallback for empty images
        auto_threshold = 127
        mean_brightness = 127
        std_brightness = 50
        median_brightness = 127
    else:
        auto_threshold, _ = cv2.threshold(non_zero_pixels, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        mean_brightness = np.mean(non_zero_pixels)
        std_brightness = np.std(non_zero_pixels)
        median_brightness = np.median(non_zero_pixels)
        
        # Additional adaptive measures for robust detection
        brightness_range = np.max(non_zero_pixels) - np.min(non_zero_pixels)
        contrast_factor = std_brightness / mean_brightness if mean_brightness > 0 else 0
    
    # Adaptive thresholding based on image characteristics and blob type
    if blob_type == "light":
        # For light blobs (inclusions) - adaptive to image contrast and brightness
        if contrast_factor > 0.3:  # High contrast image
            base_thresh = max(auto_threshold * 0.8, mean_brightness + 0.5 * std_brightness)
        elif contrast_factor < 0.15:  # Low contrast image
            base_thresh = max(median_brightness + std_brightness, mean_brightness + 0.8 * std_brightness)
        else:  # Medium contrast
            base_thresh = mean_brightness + 0.7 * std_brightness
            
        # Adaptive threshold range based on image characteristics
        adaptive_range = max(20, min(50, int(brightness_range * 0.2)))
        min_thresh = max(30, int(base_thresh - adaptive_range))
        max_thresh = min(255, int(base_thresh + adaptive_range * 1.5))
        step = max(5, min(15, int((max_thresh - min_thresh) / 8)))
        blob_color = 255
        
        # Area limits for inclusions: Adaptive defaults with user override capability
        dpcm = scan_dpi * 0.3937
        
        if size_params and 'min_inclusion_area_px' in size_params:
            # User has specified custom size filtering - use their values
            min_area = size_params['min_inclusion_area_px']
            max_area = size_params['max_inclusion_area_px']
        else:
            # Size filtering for inclusions
            # Min: 0.1mm (fine silt boundary, Wentworth scale adjusted for elbow in chart at 0.1mm)
            # Max: 20mm (elongated inclusions may appear larger to blob detector)
            min_diameter_cm = 0.01  # 0.1 mm
            max_diameter_cm = 2.0   # 20 mm

            min_area = int(np.pi * ((min_diameter_cm * dpcm / 2) ** 2))
            max_area = int(np.pi * ((max_diameter_cm * dpcm / 2) ** 2))

    elif blob_type == "dark_inclusion":
        # For dark inclusions (ferruginous grains, magnetite, biotite, dark grog)
        # Threshold logic identical to "dark" — we're still looking for dark features
        if contrast_factor > 0.3:  # High contrast image
            base_thresh = min(auto_threshold * 0.6, mean_brightness - 0.5 * std_brightness)
        elif contrast_factor < 0.15:  # Low contrast image
            base_thresh = min(median_brightness - std_brightness, mean_brightness - 0.8 * std_brightness)
        else:  # Medium contrast
            base_thresh = mean_brightness - 0.7 * std_brightness

        adaptive_range = max(15, min(40, int(brightness_range * 0.15)))
        min_thresh = max(0, int(base_thresh - adaptive_range * 1.5))
        max_thresh = min(200, int(base_thresh + adaptive_range))
        step = max(5, min(15, int((max_thresh - min_thresh) / 8)))
        blob_color = 0  # detect dark blobs

        # Area limits: use *inclusion* sizes (same as "light" branch)
        dpcm = scan_dpi * 0.3937

        if size_params and 'min_inclusion_area_px' in size_params:
            min_area = size_params['min_inclusion_area_px']
            max_area = size_params['max_inclusion_area_px']
        else:
            min_diameter_cm = 0.01  # 0.1 mm
            max_diameter_cm = 2.0   # 20 mm
            min_area = int(np.pi * ((min_diameter_cm * dpcm / 2) ** 2))
            max_area = int(np.pi * ((max_diameter_cm * dpcm / 2) ** 2))

    else:  # blob_type == "dark"
        # For dark blobs (voids) - adaptive to image characteristics
        if contrast_factor > 0.3:  # High contrast image
            base_thresh = min(auto_threshold * 0.6, mean_brightness - 0.5 * std_brightness)
        elif contrast_factor < 0.15:  # Low contrast image
            base_thresh = min(median_brightness - std_brightness, mean_brightness - 0.8 * std_brightness)
        else:  # Medium contrast
            base_thresh = mean_brightness - 0.7 * std_brightness

        # Adaptive threshold range for voids
        adaptive_range = max(15, min(40, int(brightness_range * 0.15)))
        min_thresh = max(0, int(base_thresh - adaptive_range * 1.5))
        max_thresh = min(200, int(base_thresh + adaptive_range))
        step = max(5, min(15, int((max_thresh - min_thresh) / 8)))
        blob_color = 0

        # Area limits for voids: Adaptive defaults with user override capability
        dpcm = scan_dpi * 0.3937

        if size_params and 'min_void_area_px' in size_params:
            # User has specified custom size filtering - use their values
            min_area = size_params['min_void_area_px']
            max_area = size_params['max_void_area_px']
        else:
            # Size filtering consistent with contour_detection
            # Min: 0.25mm (macroscopic voids from organic burnout)
            # Max: 15mm (voids larger than this are likely artifacts)
            min_diameter_cm = 0.025  # 0.25 mm
            max_diameter_cm = 1.5    # 15 mm

            min_area = int(np.pi * ((min_diameter_cm * dpcm / 2) ** 2))
            max_area = int(np.pi * ((max_diameter_cm * dpcm / 2) ** 2))
    
    # Set threshold parameters
    params.thresholdStep = step
    params.minThreshold = min_thresh
    params.maxThreshold = max_thresh
    
    # Area filtering (min_area and max_area are set above based on size_params or defaults)
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    
    # Color filtering
    params.filterByColor = True
    params.blobColor = blob_color
    
    # Adaptive distance filtering based on expected blob size and contrast
    base_distance = max(3, int(np.sqrt(min_area) * 1.2))
    # In low contrast images, increase distance to reduce false positives
    if 'contrast_factor' in locals() and contrast_factor < 0.15:
        base_distance = int(base_distance * 1.5)
    params.minDistBetweenBlobs = base_distance
    
    # Shape filtering to reject artifacts while preserving ceramic features
    #
    # EQUIVALENCE WITH CONTOUR DETECTION:
    # The inertia filter here is the blob-detector counterpart of contour_detection's
    # max_aspect_ratio filter.  The relationship is exact:
    #     minInertiaRatio = 1 / max_aspect_ratio
    # Both methods use the same default elongation limit:
    #     minInertiaRatio = 0.2  ↔  max_aspect_ratio = 5.0  (~5:1 maximum)
    # Elongated inclusions (biotite laths, feldspar needles, grog slivers) typically
    # fall in the 2:1–4:1 range and are captured by both detectors.
    # Wire-thin scan artifacts (dead-pixel rows, calibration lines) have ratios >> 5:1
    # and are rejected by both.
    #
    # VOID UPPER-BOUND FILTERS:
    # The void detector uses maxCircularity=0.85 and maxConvexity=0.85 to reject
    # dark features that are "too regular" (near-perfect circles or very smooth
    # convex shapes are mineral grains, not voids).  This complements the
    # dark_inclusion detector's lower-bound filters (minCircularity=0.2,
    # minConvexity=0.5).  The overlap band (0.20–0.85 circularity, 0.50–0.85
    # convexity) allows ambiguous features in both lists, matching
    # contour_detection's overlap behavior.
    if blob_type == "light":  # Light inclusions — moderately selective
        params.filterByCircularity = False  # Disabled: elongated inclusions have low circularity
        params.filterByConvexity = False    # Allow concave features (common in ceramics)
        params.filterByInertia = True       # Reject wire-thin scan artifacts (elongation filter)
        params.minInertiaRatio = 0.2        # Allow elongation up to ~5:1  (= 1 / max_aspect_ratio)
                                            # Matches contour_detection default max_aspect_ratio = 5.0
    elif blob_type == "dark_inclusion":  # Dark mineral grains — strict shape filters
        # Stricter than light inclusions — dark mineral grains (ferruginous,
        # magnetite, biotite) are typically compact and convex.  Stricter
        # filters avoid capturing irregular voids as false-positive inclusions.
        params.filterByCircularity = True
        params.minCircularity = 0.2         # Rejects very irregular outlines
        params.filterByConvexity = True
        params.minConvexity = 0.5           # Rejects concave/sinuous shapes (voids)
        params.filterByInertia = True
        params.minInertiaRatio = 0.35       # ~3:1 max aspect ratio (stricter than 0.2/5:1)
    else:  # Voids — permissive lower bounds, upper bounds reject mineral-grain-like features
        # Voids (pores, organic burnout, shrinkage cracks) are typically irregular
        # and concave.  Upper-bound filters reject dark features that are "too
        # perfect" — near-perfect circles or very smooth convex shapes are almost
        # certainly mineral grains, not voids.  This is the inverse of the
        # dark_inclusion detector's lower-bound filters.
        params.filterByCircularity = False
        params.minCircularity = 0.0         # No lower bound — voids can be very irregular
        params.maxCircularity = 0.85        # Reject near-perfect circles (likely mineral grains)
        params.filterByConvexity = False
        params.minConvexity = 0.0           # No lower bound — allow deep concavities
        params.maxConvexity = 0.85          # Reject very smooth/convex shapes (likely mineral grains)
        params.filterByInertia = False      # No elongation constraint — voids can be any shape
    
    return params


## potential to replace sherd_mask function with a more robust version that employs grab_cut algorithm without user interaction, using the current contour-based mask as an initial seed.  This could improve edge accuracy and handle cases where contours are incomplete or noisy.  However, it would add complexity and processing time, so it may be best as an optional alternative rather than a wholesale replacement of the existing sherd_mask logic.


def _optimal_canny_thresholds(image, sigma=0.33):
    """Compute Canny lower/upper thresholds from Otsu, image median, and gradient stats."""
    otsu_thresh, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    median_val = np.median(image)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_mean = np.mean(gradient_magnitude)
    base_thresh = min(otsu_thresh * 0.5, median_val, gradient_mean)
    lower_thresh = max(10, int((1.0 - sigma) * base_thresh))
    upper_thresh = max(30, int((1.0 + sigma) * base_thresh))
    lower_thresh = min(lower_thresh, 100)
    upper_thresh = min(upper_thresh, 255)
    upper_thresh = max(upper_thresh, lower_thresh * 2)
    return lower_thresh, upper_thresh


def _detect_background_statistics(image, border_size=0.05):
    """Sample image borders to estimate background median and std."""
    h, w = image.shape
    border_pixels = int(min(h, w) * border_size)
    top = image[:border_pixels, :]
    bottom = image[-border_pixels:, :]
    left = image[:, :border_pixels]
    right = image[:, -border_pixels:]
    all_border = np.concatenate([top.flatten(), bottom.flatten(),
                                 left.flatten(), right.flatten()])
    return np.median(all_border), np.std(all_border)


def _adaptive_morphology_kernel(scan_dpi, target_size_mm=0.5):
    """Build an elliptical structuring element sized for the scan resolution."""
    dpcm = scan_dpi * 0.3937
    target_size_cm = target_size_mm / 10.0
    kernel_size_pixels = int(target_size_cm * dpcm)
    kernel_size_pixels = max(3, kernel_size_pixels)
    if kernel_size_pixels % 2 == 0:
        kernel_size_pixels += 1
    kernel_size_pixels = min(kernel_size_pixels, 21)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (kernel_size_pixels, kernel_size_pixels))


def _grabcut_mask(image, scan_dpi, clahe_clip=2.0, clahe_grid=(8, 8),
                  loose_thresh_fraction=0.6, fg_erode_iter=2,
                  bg_dilate_iter=5, iterations=5, max_dim=1500):
    """
    Foreground extraction via CLAHE-enhanced V-channel + dual-Otsu trimap
    + ``cv2.grabCut``.

    Pipeline
    --------
    1. Convert BGR → HSV, take the V (value) channel.  V preserves
       luminance through JPEG chroma subsampling better than BGR→GRAY.
    2. Apply CLAHE (``clipLimit=clahe_clip``, ``tileGridSize=clahe_grid``)
       to V.  Adaptive histogram equalization enhances local contrast and
       suppresses large-scale shading (vignetting, lighting drift).
    3. Tight Otsu binarization on the CLAHE-enhanced V → ``tight_mask``
       (conservative sherd core).
    4. Loose threshold at ``loose_thresh_fraction × tight_Otsu`` →
       ``loose_mask`` (sherd plus halo).
    5. Build a three-zone trimap directly from the two masks:
         · ``GC_FGD``    = ``tight_mask`` eroded ``fg_erode_iter``× with
                          a 3×3 kernel (strictly inside the sherd, away
                          from halo).
         · ``GC_BGD``    = outside (``loose_mask`` dilated ``bg_dilate_iter``×
                          with a 5×5 kernel) — guaranteed clean background
                          samples well clear of the halo.
         · ``GC_PR_BGD`` = everything else (halo + uncertain boundary).
                          GrabCut classifies these via the learned colour
                          GMMs.
    6. ``cv2.grabCut(image, trimap, None, bgM, fgM, iterations,
                     cv2.GC_INIT_WITH_MASK)``.
    7. Output = pixels labelled ``GC_FGD`` or ``GC_PR_FGD`` → 255, else 0.
    8. Fill holes via ``findContours(RETR_EXTERNAL) + drawContours(FILLED)``
       so dark inclusions inside the sherd don't punch through the mask.
    9. Safety fallback: when the final output area is < 50% of the
       ``GC_FGD`` seed area, GrabCut has diverged — return ``tight_mask``
       so downstream code still has a usable mask.

    Returns
    -------
    numpy.ndarray
        Single-channel ``uint8`` mask in ``image`` coordinates.
    """
    # GrabCut is O(N²) on pixel count, so downsample very large scans for
    # the segmentation step and upsample the result mask afterwards.
    H_in, W_in = image.shape[:2]
    scale = 1.0
    if max(H_in, W_in) > max_dim:
        scale = max_dim / max(H_in, W_in)
        new_w = max(8, int(W_in * scale))
        new_h = max(8, int(H_in * scale))
        work_image = cv2.resize(image, (new_w, new_h),
                                interpolation=cv2.INTER_AREA)
    else:
        work_image = image

    hsv = cv2.cvtColor(work_image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip),
                            tileGridSize=tuple(clahe_grid))
    v_eq = clahe.apply(v)

    otsu_thresh, tight_mask = cv2.threshold(
        v_eq, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    _, loose_mask = cv2.threshold(
        v_eq, int(loose_thresh_fraction * otsu_thresh), 255, cv2.THRESH_BINARY
    )

    fg_core = cv2.erode(tight_mask, np.ones((3, 3), np.uint8),
                        iterations=fg_erode_iter)
    bg_far = cv2.dilate(loose_mask, np.ones((5, 5), np.uint8),
                        iterations=bg_dilate_iter)

    trimap = np.full(v.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    trimap[bg_far == 0] = cv2.GC_BGD
    trimap[fg_core > 0] = cv2.GC_FGD

    sure_fg_area = int(cv2.countNonZero(fg_core))
    if sure_fg_area == 0:
        # No core to seed the FG GMM — fall back to the tight mask
        # (resized back to input dimensions if we downsampled).
        if scale != 1.0:
            return cv2.resize(tight_mask, (W_in, H_in),
                              interpolation=cv2.INTER_NEAREST)
        return tight_mask

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    # GrabCut fits its foreground/background GMMs via K-means, whose
    # initialization is seeded from OpenCV's global RNG.  Lock the seed
    # so the mask (and every downstream detection) is bit-reproducible.
    cv2.setRNGSeed(0)
    try:
        cv2.grabCut(work_image, trimap, None, bg_model, fg_model,
                    iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        if scale != 1.0:
            return cv2.resize(tight_mask, (W_in, H_in),
                              interpolation=cv2.INTER_NEAREST)
        return tight_mask

    fg_mask = np.where(
        (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Fill holes (background-coloured inclusions inside the sherd).
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if contours:
        filled = np.zeros_like(fg_mask)
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
        fg_mask = filled

    # Safety fallback: when GrabCut collapses the output below half the
    # FG seed area, distrust the result and return the tight Otsu mask.
    if int(cv2.countNonZero(fg_mask)) / sure_fg_area < 0.5:
        fg_mask = tight_mask

    # Upsample mask back to the original input size if we downsampled.
    if scale != 1.0:
        fg_mask = cv2.resize(fg_mask, (W_in, H_in),
                             interpolation=cv2.INTER_NEAREST)

    return fg_mask


def _solidity(contour):
    """area / convex-hull area; 1.0 = perfectly convex, lower = ragged."""
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return float(area / hull_area)


def _smooth_contour(contour, eps_frac=0.001, eps_min=1.0):
    """Flatten pixel-level zigzag with approxPolyDP; preserves overall shape."""
    perim = cv2.arcLength(contour, True)
    eps = max(eps_min, eps_frac * perim)
    return cv2.approxPolyDP(contour, eps, True)


def _drop_nested(contours, areas):
    """Keep only outermost contours.

    Walks contours largest-first and rejects any whose centroid lies inside
    an already-kept larger contour.  A single inclusion with internal color
    gradient otherwise registers as 2+ detections (one parent contour + child
    contours from threshold-band boundaries inside it).
    """
    if len(contours) < 2:
        return list(contours), list(areas)
    order = sorted(range(len(contours)), key=lambda i: areas[i], reverse=True)
    kept_idx = []
    for i in order:
        M = cv2.moments(contours[i])
        if M['m00'] == 0:
            kept_idx.append(i)
            continue
        cx = float(M['m10'] / M['m00'])
        cy = float(M['m01'] / M['m00'])
        nested = any(
            cv2.pointPolygonTest(contours[j], (cx, cy), False) >= 0
            for j in kept_idx
        )
        if not nested:
            kept_idx.append(i)
    kept_idx.sort()  # preserve original input order
    return [contours[i] for i in kept_idx], [areas[i] for i in kept_idx]


def _select_best_contour(fg_mask, image_area, scan_dpi,
                         min_area_cm2=0.25, max_area_ratio=0.9,
                         solidity_floor=0.75):
    """
    Pick the single best sherd contour from ``fg_mask``.

    Filters out tiny specks (area < ``min_area_cm2``), the whole-frame
    fallback (area > ``max_area_ratio`` of image), and ragged dust-blob
    contours (solidity < ``solidity_floor``).  Returns the largest
    survivor smoothed with ``approxPolyDP``, or ``None`` when no contour
    passes the filters.
    """
    dpcm2 = (scan_dpi * 0.3937) ** 2
    min_area_px = min_area_cm2 * dpcm2
    max_area_px = max_area_ratio * image_area

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    survivors = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area_px or a > max_area_px:
            continue
        if _solidity(c) < solidity_floor:
            continue
        survivors.append((a, c))
    if not survivors:
        return None

    survivors.sort(key=lambda t: t[0], reverse=True)
    return _smooth_contour(survivors[0][1])


def _bbox_iou(b1, b2):
    """Intersection-over-union of two ``(x, y, w, h)`` bounding rects."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0


def _bbox_containment(inner, outer):
    """Fraction of ``inner`` bbox area that lies inside ``outer`` bbox."""
    x1, y1, w1, h1 = inner
    x2, y2, w2, h2 = outer
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    inner_area = w1 * h1
    return inter / inner_area if inner_area > 0 else 0


def _select_multiple_contours(fg_mask, image_area, scan_dpi,
                              n_sherds=None, min_area_cm2=0.25,
                              max_area_ratio=0.9, solidity_floor=0.75,
                              envelope_containment=0.8, envelope_min_children=2,
                              gap_ratio_threshold=0.4):
    """
    Pick sherd contours from an HSV foreground mask using shape filters,
    envelope-contour elimination, and a gap-based stopping rule.

    Strategy
    --------
    1. Extract external contours from ``fg_mask``.
    2. Drop contours below ``min_area_cm2`` (DPI-aware), above
       ``max_area_ratio`` of the image, or below ``solidity_floor``
       (ragged dust-blob contours).
    3. Sort survivors descending by area.
    4. Eliminate envelope contours: any survivor whose bbox contains
       ``envelope_min_children``+ other survivors at ``envelope_containment``
       fraction is a wrapper around a cluster (halos bridging) and dropped.
    5. Auto-count: walk consecutive area pairs and stop at the largest
       ratio drop-off only when that drop is ``< gap_ratio_threshold``
       (default 0.5 = "next contour at least 2× smaller").  Otherwise keep
       every survivor.
    6. If ``n_sherds`` is given, skip the gap rule and take top-N.
    7. Smooth each chosen contour with ``approxPolyDP``.

    Returns
    -------
    list of contours (smoothed), sorted descending by area.
    """
    dpcm2 = (scan_dpi * 0.3937) ** 2
    min_area_px = min_area_cm2 * dpcm2
    max_area_px = max_area_ratio * image_area

    raw_contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
    if not raw_contours:
        return []

    pool = []
    for c in raw_contours:
        a = cv2.contourArea(c)
        if a < min_area_px or a > max_area_px:
            continue
        if _solidity(c) < solidity_floor:
            continue
        pool.append((a, cv2.boundingRect(c), c))

    if not pool:
        return []

    pool.sort(key=lambda t: t[0], reverse=True)

    if len(pool) >= envelope_min_children + 1:
        non_envelope = []
        for i, (area_i, bbox_i, contour_i) in enumerate(pool):
            child_count = 0
            for j, (area_j, bbox_j, _) in enumerate(pool):
                if i == j or area_j >= area_i:
                    continue
                if _bbox_containment(bbox_j, bbox_i) >= envelope_containment:
                    child_count += 1
                    if child_count >= envelope_min_children:
                        break
            if child_count < envelope_min_children:
                non_envelope.append((area_i, bbox_i, contour_i))
        pool = non_envelope

    if not pool:
        return []

    if n_sherds is not None:
        chosen = [c for _, _, c in pool[:int(n_sherds)]]
    elif len(pool) == 1:
        chosen = [pool[0][2]]
    else:
        ratios = [pool[i][0] / pool[i - 1][0] for i in range(1, len(pool))]
        gap_idx = int(np.argmin(ratios))
        if ratios[gap_idx] < gap_ratio_threshold:
            keep_count = gap_idx + 1
        else:
            keep_count = len(pool)
        chosen = [c for _, _, c in pool[:keep_count]]

    return [_smooth_contour(c) for c in chosen]


def _contour_to_crop_and_mask(contour, orig_h, orig_w, border_crop,
                              crop_buffer, auto_crop):
    """
    Convert a sherd contour (in ``image_cropped`` coords) into a
    full-size binary mask, a cropped+padded ``mask_slice``, and the
    crop tuple in original-image coordinates.

    Returns
    -------
    tuple
        ``(mask_slice, crop)`` where ``crop`` is the same 8-element tuple
        ``sherd_mask`` returns: ``(y1, y2, x1, x2, pad_top, pad_bottom,
        pad_left, pad_right)``.
    """
    cropped_h = orig_h - 2 * border_crop
    cropped_w = orig_w - 2 * border_crop

    blackbox = np.zeros((cropped_h, cropped_w), np.uint8)
    if contour is not None:
        mask_cropped = cv2.drawContours(blackbox.copy(), [contour], -1, 255, cv2.FILLED, 1)
    else:
        mask_cropped = blackbox

    mask = np.zeros((orig_h, orig_w), np.uint8)
    mask[border_crop:orig_h - border_crop, border_crop:orig_w - border_crop] = mask_cropped

    if contour is not None and auto_crop:
        x_br, y_br, w_br, h_br = cv2.boundingRect(contour)
        side = max(w_br, h_br)
        x_center = x_br + w_br // 2
        y_center = y_br + h_br // 2
        x1_raw = x_center + border_crop - side // 2 - crop_buffer
        x2_raw = x_center + border_crop + side // 2 + crop_buffer
        y1_raw = y_center + border_crop - side // 2 - crop_buffer
        y2_raw = y_center + border_crop + side // 2 + crop_buffer
        x1 = max(0, x1_raw)
        x2 = min(orig_w, x2_raw)
        y1 = max(0, y1_raw)
        y2 = min(orig_h, y2_raw)
        pad_left = x1 - x1_raw
        pad_right = x2_raw - x2
        pad_top = y1 - y1_raw
        pad_bottom = y2_raw - y2
    else:
        y1, y2, x1, x2 = 0, orig_h, 0, orig_w
        pad_top = pad_bottom = pad_left = pad_right = 0

    crop = (y1, y2, x1, x2, pad_top, pad_bottom, pad_left, pad_right)

    mask_slice = mask[y1:y2, x1:x2]
    if pad_top or pad_bottom or pad_left or pad_right:
        mask_slice = np.pad(mask_slice,
                            ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='constant', constant_values=0)

    return mask_slice, crop


def sherd_mask(sherd_scan, gray=False, scan_dpi=1200, crop_buffer=125, auto_crop=True):
    """
    Enhanced sherd masking with optimal edge detection and adaptive parameters.

    By default the mask is automatically cropped to the tightest bounding box
    of the detected sherd contour plus ``crop_buffer`` pixels on every side.
    This removes irrelevant background pixels from all downstream computations,
    significantly reducing processing time for large scans.  Set
    ``auto_crop=False`` to skip cropping and retain the full original image
    dimensions (useful for stitching results back into a larger scan).

    Parameters
    ----------
    sherd_scan : numpy.ndarray
        A scanned image of the sherd for which you want a mask
    gray : bool, optional
        If True returns single channel masked grayscale images;
        if False creates color masks (default: False)
    scan_dpi : int, optional
        Scan resolution for adaptive parameter scaling (default: 1200)
        Valid range: 150-2400 DPI
    crop_buffer : int, optional
        Extra pixels to keep beyond the sherd bounding box on all four sides
        (default: 75).  Ignored when ``auto_crop=False``.
    auto_crop : bool, optional
        If True (default), crop the returned mask to the sherd bounding box
        plus ``crop_buffer``.  If False, return a full-size mask matching the
        original image dimensions.

    Returns
    -------
    tuple
        ``(mask, (y1, y2, x1, x2))`` where *mask* is the binary mask
        (grayscale uint8 or 3-channel) and the second element is the crop
        rectangle in the original image's pixel coordinates.  When
        ``auto_crop=True`` the mask is already cropped; apply the same crop
        to the source image with ``apply_mask(image, mask, crop)`` or
        directly as ``image[y1:y2, x1:x2]``.  When ``auto_crop=False`` the
        crop rectangle spans the full image ``(0, H, 0, W)`` and the mask is
        full-size.
    """
    # Input validation
    if scan_dpi < 150 or scan_dpi > 2400:
        print(f"Warning: scan_dpi {scan_dpi} outside recommended range (150-2400)")

    image = sherd_scan
    orig_h, orig_w = image.shape[:2]

    # Crop off outer 0.5cm border to remove the scanner-box outline that
    # blocks stray light on full-bed scans.  Tight per-sherd crops have no
    # scanner box and would lose their actual background margin, so the
    # crop is skipped when the image is too small to spare it.
    dpcm = scan_dpi * 0.3937
    desired_border = int(0.5 * dpcm)
    if min(orig_h, orig_w) >= 8 * desired_border:
        border_crop = desired_border
    else:
        border_crop = 0
    image_cropped = image[border_crop:orig_h - border_crop, border_crop:orig_w - border_crop]

    # CLAHE-enhanced V channel + dual Otsu thresholds → GrabCut trimap.
    fg_mask = _grabcut_mask(image_cropped, scan_dpi)
    image_area = image_cropped.shape[0] * image_cropped.shape[1]
    best_contour = _select_best_contour(fg_mask, image_area, scan_dpi)

    mask_slice, crop = _contour_to_crop_and_mask(
        best_contour, orig_h, orig_w, border_crop, crop_buffer, auto_crop
    )

    color_mask_slice = np.dstack((mask_slice, mask_slice, mask_slice))

    #return the mask (cropped when auto_crop=True), the crop rectangle, and the
    #best_contour (in image_cropped coordinates) so callers can use its geometry
    #(e.g. minAreaRect angle) without needing to re-derive the sherd boundary.
    if gray == True:
        return mask_slice, crop, best_contour
    else:
        return color_mask_slice, crop, best_contour


def apply_mask(image, mask, crop=None):
    """
    Apply a mask to an image (single image version of super_zorro_cv).

    Parameters
    ----------
    image : numpy.ndarray
        Original image to mask
    mask : numpy.ndarray
        Mask to apply (from sherd_mask function).  Must already be cropped to
        match the region described by ``crop`` if ``crop`` is provided.
    crop : tuple or None, optional
        ``(y1, y2, x1, x2, pad_top, pad_bottom, pad_left, pad_right)``
        crop rectangle returned by ``sherd_mask``.  The first four elements
        define the slice into the original image; the last four (optional)
        give zero-padding needed when the sherd is near the scan edge.
        When provided the image is sliced and padded to match the
        (already-cropped-and-padded) mask.  When None the image is used
        as-is and must already match the mask dimensions.

    Returns
    -------
    numpy.ndarray
        Masked image, cropped to the sherd bounding region when ``crop`` is
        provided.
    """
    if crop is not None:
        y1, y2, x1, x2 = crop[:4]
        image = image[y1:y2, x1:x2]
        if len(crop) == 8:
            pt, pb, pl, pr = crop[4:]
            if pt or pb or pl or pr:
                image = np.pad(image, ((pt, pb), (pl, pr), (0, 0)),
                               mode='constant', constant_values=0)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def clahe_enhance(masked_image, clip_limit=2.0, tile_grid=(8, 8)):
    """
    Apply CLAHE to the L* channel of a masked sherd image.

    Enhances local contrast between paste and inclusions/voids so the
    downstream blob and contour detectors see a wider, cleaner intra-sherd
    intensity range.  Operates in CIELAB to stay consistent with the rest
    of the analysis pipeline (sherd_blobs, contour_detection, and the
    color analysis all work in Lab).

    Parameters
    ----------
    masked_image : numpy.ndarray
        BGR image with the non-sherd background already set to zero
        (output of ``apply_mask``).
    clip_limit : float, optional
        CLAHE contrast clipping limit (default: 2.0).  Higher values give
        more aggressive enhancement; values above ~4 tend to amplify noise.
    tile_grid : tuple of int, optional
        CLAHE tile grid size (default: (8, 8)).  Smaller tiles give more
        local adaptation but can introduce boundary artifacts in
        low-texture regions.

    Returns
    -------
    numpy.ndarray
        BGR image with CLAHE-enhanced L*; background pixels (those that
        were zero on input) are re-zeroed so the mask remains intact.

    Notes
    -----
    CLAHE is applied to the full L* channel and then the original
    background (any pixel that was zero across all three input channels)
    is re-zeroed.  Tiles spanning the sherd boundary see a bimodal
    histogram (black background + sherd); the ``clip_limit`` of 2.0 keeps
    the resulting boundary artifacts well below the inclusion-detection
    thresholds.
    """
    if masked_image is None or masked_image.size == 0:
        return masked_image

    # Remember which pixels were background so we can re-zero them after
    # CLAHE inevitably bleeds some signal into boundary tiles.
    if masked_image.ndim == 3:
        bg = np.all(masked_image == 0, axis=2)
    else:
        bg = (masked_image == 0)

    lab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2Lab)
    l_channel = lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=tuple(tile_grid))
    lab[:, :, 0] = clahe.apply(l_channel)

    enhanced = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    enhanced[bg] = 0
    return enhanced


_VALID_CHANNELS = ('L', 'B', 'G', 'R')


def _extract_channel(image, channel, enhance_contrast=False,
                     clip_limit=2.0, tile_grid=(8, 8)):
    """Extract a single-channel uint8 image, optionally CLAHE-enhanced.

    Parameters
    ----------
    image : numpy.ndarray
        Masked BGR image (OpenCV's native channel order — not RGB).
        Background pixels are expected to be zero across all channels.
    channel : {'L', 'B', 'G', 'R'}
        'L' = CIELAB lightness (BGR→Lab, take L*); 'B'/'G'/'R' = the
        corresponding BGR channel directly from the input.
    enhance_contrast : bool
        If True, apply CLAHE on the extracted channel.  Background pixels
        (zero across all input channels) are re-zeroed afterward so the mask
        stays intact even though CLAHE bleeds signal into boundary tiles.
    clip_limit, tile_grid
        Forwarded to ``cv2.createCLAHE``.
    """
    if image is None or image.size == 0:
        return image
    if channel not in _VALID_CHANNELS:
        raise ValueError(
            f"Unknown channel {channel!r}; expected one of {_VALID_CHANNELS}")

    if image.ndim == 2:
        gray = image.copy()
    elif channel == 'L':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 0]
    else:
        # cv2 stores images in BGR order, so index 0=B, 1=G, 2=R.
        bgr_idx = {'B': 0, 'G': 1, 'R': 2}[channel]
        gray = image[:, :, bgr_idx].copy()

    if enhance_contrast:
        if image.ndim == 3:
            bg = np.all(image == 0, axis=2)
        else:
            bg = (image == 0)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                                tileGridSize=tuple(tile_grid))
        gray = clahe.apply(gray)
        gray[bg] = 0
    return gray


def _gate_contours_by_pop(contours, raw_bgr, pop_min):
    """Drop contours whose center-vs-ring intensity on raw BGR is below ``pop_min``.

    For each contour: derive an equivalent-circle radius from the contour
    area, sample the core disc and a surrounding annulus on each native
    BGR channel of the **unmodified input image**, take the maximum
    absolute (core_mean - ring_mean) across channels, and keep only
    contours whose max-pop is >= ``pop_min``.  Background pixels (== 0)
    are excluded from both regions.  Mirrors the inclusion gate in
    ``sherd_blobs`` so blob and contour pipelines share the same logic.
    """
    if not contours or raw_bgr.ndim != 3 or raw_bgr.shape[2] < 3:
        return contours
    h, w = raw_bgr.shape[:2]
    kept = []
    for c in contours:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        M = cv2.moments(c)
        if M['m00'] <= 0:
            continue
        cx = int(round(M['m10'] / M['m00']))
        cy = int(round(M['m01'] / M['m00']))
        r = max(2, int(round(np.sqrt(area / np.pi))))
        R = int(round(r * 2.0))
        y0, y1 = max(0, cy - R), min(h, cy + R + 1)
        x0, x1 = max(0, cx - R), min(w, cx + R + 1)
        yy = np.arange(y0 - cy, y1 - cy).reshape(-1, 1)
        xx = np.arange(x0 - cx, x1 - cx).reshape(1, -1)
        d2 = yy * yy + xx * xx
        core_mask = d2 <= (r * r)
        ring_mask = (d2 > r * r) & (d2 <= R * R)
        best = 0.0
        for ch in range(3):
            patch = raw_bgr[y0:y1, x0:x1, ch]
            if patch.size == 0:
                continue
            valid = patch > 0
            core_sel = core_mask & valid
            ring_sel = ring_mask & valid
            if core_sel.sum() < 3 or ring_sel.sum() < 3:
                continue
            delta = abs(float(patch[core_sel].mean())
                        - float(patch[ring_sel].mean()))
            if delta > best:
                best = delta
        if best >= pop_min:
            kept.append(c)
    return kept


def _combine_blob_lists(blob_lists_by_channel, combine_mode='union', vote_min=2,
                        distance_factor=0.5):
    """Pool blob keypoints across channels with NMS-based dedup and optional voting.

    Parameters
    ----------
    blob_lists_by_channel : dict[str, list[cv2.KeyPoint]]
        One list of detected keypoints per channel name.
    combine_mode : {'union', 'vote'}
        'union' returns the deduplicated pool (each spatial cluster contributes
        its largest blob).  'vote' additionally requires a cluster to contain
        contributions from at least ``vote_min`` distinct channels.
    vote_min : int
        Minimum number of distinct channels that must agree for a blob to pass
        in 'vote' mode.  Ignored for 'union'.
    distance_factor : float
        Two keypoints belong to the same spatial cluster when their centers
        are within ``distance_factor * max(size_a, size_b)`` pixels.
    """
    if not blob_lists_by_channel:
        return []
    channels = list(blob_lists_by_channel.keys())
    if len(channels) == 1:
        return list(blob_lists_by_channel[channels[0]])

    pooled = [(ch, b) for ch, bs in blob_lists_by_channel.items() for b in bs]
    pooled.sort(key=lambda cb: -cb[1].size)

    # Cluster merge radius is keyed off the SMALLER of the two blob sizes,
    # not the larger.  Rationale: two detections represent the same physical
    # feature only when their centers are within roughly the smaller blob's
    # own footprint — a spurious giant blob (e.g. when CLAHE on a partially
    # darker sherd region promotes the whole hemisphere into one massive
    # dark "blob") would otherwise sweep up every legitimate small detection
    # within half its radius into a single cluster, silently zeroing
    # detections across that area.  Using ``min`` keeps cross-channel
    # duplicates merged (real duplicates are nearly coincident regardless
    # of size) while preventing distant absorption.
    clusters = []  # list of [set(channels_seen), representative_blob]
    for ch, b in pooled:
        bx, by = b.pt
        matched = None
        for cluster in clusters:
            rb = cluster[1]
            rx, ry = rb.pt
            dist = float(np.hypot(bx - rx, by - ry))
            if dist < distance_factor * min(b.size, rb.size):
                matched = cluster
                break
        if matched is None:
            clusters.append([{ch}, b])
        else:
            matched[0].add(ch)

    if combine_mode == 'vote':
        return [rb for chs, rb in clusters if len(chs) >= vote_min]
    return [rb for _, rb in clusters]


def _combine_contour_lists(contour_lists_by_channel, image_shape,
                           combine_mode='union', vote_min=2):
    """Pool contour lists across channels with centroid-containment dedup and optional voting.

    Parameters
    ----------
    contour_lists_by_channel : dict[str, list[contour]]
        One list of contours per channel name.
    image_shape : tuple
        ``(h, w[, ...])`` of the source image; used to rasterize per-channel
        masks for the vote count.
    combine_mode : {'union', 'vote'}
        'union' returns the pooled outermost contours (``_drop_nested`` does
        the cross-channel dedup since a contour fully containing another's
        centroid is treated as the same physical feature).  'vote' additionally
        requires the contour's centroid to fall inside ≥ ``vote_min`` per-channel
        masks.
    vote_min : int
        Minimum number of channels that must agree.  Ignored for 'union'.
    """
    if not contour_lists_by_channel:
        return []
    channels = list(contour_lists_by_channel.keys())
    if len(channels) == 1:
        return list(contour_lists_by_channel[channels[0]])

    all_contours = [c for cs in contour_lists_by_channel.values() for c in cs]
    if not all_contours:
        return []
    areas = [cv2.contourArea(c) for c in all_contours]
    kept_contours, _ = _drop_nested(all_contours, areas)

    if combine_mode != 'vote':
        return kept_contours

    h, w = image_shape[:2]
    channel_masks = {}
    for ch, contours in contour_lists_by_channel.items():
        m = np.zeros((h, w), dtype=np.uint8)
        if contours:
            cv2.drawContours(m, contours, -1, 1, thickness=cv2.FILLED)
        channel_masks[ch] = m

    filtered = []
    for c in kept_contours:
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        votes = sum(1 for ch in channels if channel_masks[ch][cy, cx] > 0)
        if votes >= vote_min:
            filtered.append(c)
    return filtered


def super_zorro_cv(folder_read, folder_write, fileformat='jpeg', gray=False, scan_dpi=1200):
    """
    Enhanced batch sherd masking with optimal edge detection and adaptive parameters.
    
    Parameters
    ----------
    folder_read : str
        Path to folder containing images to process
    folder_write : str
        Path to folder where masked images will be saved
    fileformat : str, optional
        File format to process (default: 'jpeg')
    gray : bool, optional
        If True saves single channel masked grayscale images; 
        if False saves color masks (default: False)
    scan_dpi : int, optional
        Scan resolution for adaptive parameter scaling (default: 1200)
        Valid range: 150-2400 DPI
        
    Returns
    -------
    None
        Saves processed images to folder_write
    """
    os.makedirs(folder_write, exist_ok=True)
    
    pathstr = [str(path) for path in (Path(folder_read).rglob(f'*.{fileformat}'))]
    folder_len = len(folder_read) 
    
    for path in pathstr:
        #read image
        image = cv2.imread(path)
        if image is None:
            print(f"Warning: Could not load image {path}")
            continue
            
        #cvt to L* channel (CIELAB lightness)
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 0]
        
        # Enhanced edge detection with optimal thresholds
        # Use same robust approach as sherd_mask
        otsu_thresh, _ = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        median_val = np.median(im_gray)
        sobel_x = cv2.Sobel(im_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(im_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_mean = np.mean(gradient_magnitude)
        base_thresh = min(otsu_thresh * 0.5, median_val, gradient_mean)
        lower_thresh = max(10, int(0.67 * base_thresh))
        upper_thresh = max(30, int(1.33 * base_thresh))
        lower_thresh = min(lower_thresh, 100)
        upper_thresh = min(upper_thresh, 255)
        upper_thresh = max(upper_thresh, lower_thresh * 2)
        edges = cv2.Canny(im_gray, lower_thresh, upper_thresh)
        
        # DPI-aware morphological kernel
        dpcm = scan_dpi * 0.3937
        target_size_cm = 0.05  # 0.5mm target size
        kernel_size_pixels = int(target_size_cm * dpcm)
        kernel_size_pixels = max(3, kernel_size_pixels)
        if kernel_size_pixels % 2 == 0:
            kernel_size_pixels += 1
        kernel_size_pixels = min(kernel_size_pixels, 21)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_pixels, kernel_size_pixels))
        
        res = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel)
        res2 = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
        
        #find the contours of the almost fully binarized mask
        contours_canny, _ = cv2.findContours(res2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        importantcontour_canny = max(contours_canny, key = cv2.contourArea) if len(contours_canny) > 0 else None
        
        #run a blur on the grayscale image
        blur = cv2.GaussianBlur(im_gray,(5,5),0)
        #threshold the blurred image to get foreground background elements
        ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        #find the contours of the fore from the back
        contours_thresh, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        # this grabs the largest contour which in this case is the one we want for the whole sherd
        importantcontour_thresh = max(contours_thresh, key = cv2.contourArea) if len(contours_thresh) > 0 else None
        
        # Select best contour
        if importantcontour_canny is not None and importantcontour_thresh is not None:
            #Need the smaller of the two masks depending on the method because dusty scans will make the edges all come together
            if cv2.contourArea(importantcontour_canny) > 1.1*cv2.contourArea(importantcontour_thresh):
                importantcontour = importantcontour_thresh
            elif cv2.contourArea(importantcontour_canny) < cv2.contourArea(importantcontour_thresh):
                importantcontour = importantcontour_thresh
            else:
                importantcontour = importantcontour_canny
        elif importantcontour_thresh is not None:
            importantcontour = importantcontour_thresh
        elif importantcontour_canny is not None:
            importantcontour = importantcontour_canny
        else:
            print(f"Warning: No contours found for {path}")
            continue
            
        #multiplying the image by the 3Dmask to basically create a large 0,0,0 area for the background
        #create a mask that is all zeros the same shape as the og image;
        #take that big ole contour and try to fill it in with ones (this never fucking works)
        #(I'm putting this bastard in brackets now to pass it an array of arrays; hopefully results in something)
        blackbox = np.zeros(im_gray.shape, np.uint8)
        mask = cv2.drawContours(blackbox.copy(), [importantcontour], -1, 255,cv2.FILLED, 1)
        #Need to 'stack' the image to create a 3D array, because RGB images are 3D arrays
        color_mask = np.dstack((mask, mask, mask))
        
        masked_image = cv2.bitwise_and(color_mask, image)
        masked_image_gray = cv2.bitwise_and(mask, im_gray)
        
        #save the image in the specified folder
        if gray == False:
            cv2.imwrite(f'{folder_write}/{path[folder_len:]}', masked_image)
        else:
            cv2.imwrite(f'{folder_write}/{path[folder_len:]}', masked_image_gray)
    print('Done!')


def sherd_blobs(image, scan_dpi=1200, size_params=None, blob_params=None, blur_scale=1.0,
                channels=('B', 'G', 'R'), combine_mode='union', vote_min=2,
                enhance_contrast=True, clahe_clip=2.0, clahe_grid=(8, 8),
                void_intensity_max=60.0, inclusion_pop_min=25.0):
    """
    Enhanced blob detection with robust, adaptive parameters and customizable size filtering.

    Parameters
    ----------
    image : numpy.ndarray
        Image array of a scanned sherd (not file path)
    scan_dpi : int, optional
        Scan resolution in dots per inch (default: 1200)
        Valid range: 150-2400 DPI
    size_params : dict, optional
        Dictionary containing size filtering parameters:
        - min_inclusion_area_px: minimum inclusion area in pixels
        - max_inclusion_area_px: maximum inclusion area in pixels
        - min_void_area_px: minimum void area in pixels
        - max_void_area_px: maximum void area in pixels
    channels : tuple of str, optional
        Channels to run blob detection on.  Default ``('B', 'G', 'R')`` runs
        detection on each of OpenCV's native BGR channels (not RGB) and
        combines the results, so inclusions that only contrast strongly in
        one channel (e.g. iron-rich grains in R, organic dark cores in B)
        get picked up.  Valid entries also include ``'L'`` (CIELAB lightness)
        — pass ``channels=('L',)`` to recover the pre-multi-channel L\*-only
        behavior.  L\* is excluded from the default because it's a
        perceptually-weighted blend of B/G/R, so including it gives features
        visible in L\* an extra redundant vote in the combination step.
    combine_mode : {'union', 'vote'}, optional
        How to merge per-channel detections when ``len(channels) > 1``.
        Default ``'union'`` pools detections and removes spatial
        duplicates without requiring cross-channel agreement.  This
        catches monochromatic features that only contrast strongly in
        one channel — e.g. an iron-bearing mineral grain in sand
        temper may register as warm-toned against a cream matrix and
        thus pop in B (where the warm grain reads dark) while showing
        near-zero contrast in R (where both grain and matrix read
        bright).  The prior ``'vote'`` default with ``vote_min=2`` was
        dropping roughly half of these legitimate single-channel
        detections.  Noise rejection is instead handled by
        ``inclusion_pop_min`` (sampled on raw, pre-CLAHE BGR), which
        is a stronger discriminator than per-channel agreement: it
        directly measures whether a candidate carries real intensity
        contrast on the original pixels.  Use ``'vote'`` only if you
        have a specific reason to require cross-channel agreement
        (e.g. very noisy scans where the pop gate alone is
        insufficient).
    vote_min : int, optional
        Minimum number of channels that must agree for a feature to be kept
        when ``combine_mode='vote'`` (default: 2 of 3 BGR channels).
        Ignored under the default ``combine_mode='union'``.
    enhance_contrast : bool, optional
        Apply CLAHE to each requested channel before detection (default:
        True).  Set to False if you've already pre-applied contrast
        enhancement to the input image — otherwise the detector handles
        CLAHE per channel internally.
    clahe_clip, clahe_grid : float / tuple, optional
        Forwarded to ``cv2.createCLAHE`` when ``enhance_contrast=True``.
    void_intensity_max : float in 0..255, optional
        Maximum allowed mean pixel intensity inside a void keypoint's
        disc, sampled from the (pre-blur) channel (default: 60).  Mirrors
        the gate in ``contour_detection``: a real pore reads near-black
        inside, while a dark mineral inclusion is just darker paste and
        stays well above black.  Without this gate, dark mineral grains on
        light-grey fabrics show up in the void list because the dark-void
        blob detector's upper-bound shape filters alone can't separate
        them from grains.  Lower (e.g. 45) for stricter void detection;
        raise (e.g. 90) for low-contrast scans.
    inclusion_pop_min : float in 0..255, optional
        Minimum "pop" required for a candidate inclusion to be kept
        (default: 25).  "Pop" here is informal shorthand for **how
        much the feature visually stands out against its immediate
        surrounding paste** — concretely, the absolute difference
        between the candidate's core disc mean intensity and the mean
        intensity of a surrounding annulus, computed on the **raw
        (pre-CLAHE) BGR channels** of the input image and taken as the
        maximum across the three native channels:

            pop = max over BGR of |mean(core disc) - mean(annulus)|

        Higher values mean a clearer intensity discontinuity between
        the inclusion and the paste around it; a low pop value means
        the "blob" the detector found is actually flat against its
        surround — almost certainly CLAHE-amplified noise dressed up to
        look like a feature.  CLAHE on uniform paste amplifies
        sub-tile noise into pseudo-blobs that would otherwise survive
        detection; sampling center-vs-ring intensity on the original
        pixels reveals these locations have no real differential,
        while genuine inclusions pop strongly on at least one channel.
        This is the primary noise rejection mechanism under the
        default ``combine_mode='union'`` — it replaces the old cross-
        channel voting requirement, which incorrectly punished
        monochromatic features (e.g. an iron-bearing sand grain
        visible only in B against a warm matrix).
        Set to 0 to disable; raise (e.g. 30-35) for very uniform paste
        or to further tighten precision; lower (e.g. 15-20) when
        chasing subtle features in fine-grained fabrics (paired with
        ``combine_mode='vote'`` for noise control if needed).
    blob_params : dict, optional
        Dictionary to override any cv2.SimpleBlobDetector_Params attributes
        after the adaptive defaults are calculated by setup_robust_blob_params.
        Applies to all three internal detectors (light-inclusion,
        dark-inclusion, and dark-void).

        Shape filtering keys:
        - filterByCircularity (bool), minCircularity (float 0–1)
          Default: disabled.  Enable to restrict detection to compact grains.
          e.g. minCircularity=0.7 captures near-circular (quartz-like) grains
          and rejects elongated minerals (biotite laths, feldspar needles).
        - filterByConvexity (bool), minConvexity (float 0–1)
          Default: disabled.  Enable to reject grains with deep concavities.
        - filterByInertia (bool), minInertiaRatio (float 0–1)
          Default: True / 0.2 (allows up to ~5:1 aspect ratio).
          This is the primary shape filter for inclusions and directly
          mirrors the ``inclusion_max_aspect_ratio`` parameter in ``contour_detection``:

              minInertiaRatio = 1 / max_aspect_ratio
              0.2  ↔  max_aspect_ratio = 5.0  (the shared default)

          Decreasing minInertiaRatio accepts more elongated shapes:
          e.g. 0.1 → ~10:1 max, 0.05 → ~20:1 (very elongated laths).
          Increasing restricts to more equant grains:
          e.g. 0.5 → ~4:1 max, 0.9 → ~1.2:1 (near-circular only).
        - minDistBetweenBlobs (float, pixels)
          Default: adaptive (~1.2× sqrt of min area).
          Increase to avoid double-counting adjacent touching grains.

        Threshold keys (override the adaptive calculation):
        - minThreshold, maxThreshold (float 0–255)
        - thresholdStep (float)

        Note: blobColor and filterByColor are set internally to select
        light vs dark features and should NOT be overridden here.

        Example — restrict to near-circular grains (quartz, oolites)::

            inclusions, voids = amacfa_plus.sherd_blobs(
                masked_img, scan_dpi=SCAN_DPI,
                blob_params={'filterByCircularity': True, 'minCircularity': 0.7}
            )

        Example — accept highly elongated blobs (same as setting max_aspect_ratio=10 in contour_detection)::

            inclusions, _ = amacfa_plus.sherd_blobs(
                masked_img, scan_dpi=SCAN_DPI,
                blob_params={'filterByInertia': True, 'minInertiaRatio': 0.1}
            )

    Returns
    -------
    tuple
        (inclusion_blobs, void_blobs) — Two lists of ``cv2.KeyPoint``.
        *inclusion_blobs* contains both light and dark mineral inclusions;
        *void_blobs* contains all detected dark voids.

    Notes
    -----
    Internally three detectors run: light-inclusions, dark-inclusions, and
    dark-voids.  The dark-inclusion detector uses the same adaptive dark
    thresholding as the void detector but applies inclusion-level size limits
    and strict shape filters (circularity >= 0.2, convexity >= 0.5, inertia
    ratio >= 0.35) to capture only high-confidence dark mineral grains
    (ferruginous, magnetite, biotite, dark grog).  The void detector uses
    **upper-bound** shape filters (maxCircularity = 0.85, maxConvexity = 0.85)
    to reject features that are too regular — near-perfect circles or very
    smooth convex shapes are almost certainly mineral grains, not voids.
    Together the lower-bound (dark-inclusion) and upper-bound (void) filters
    form complementary shape discriminators, but on real masked sherds the
    blur smooths concavities and dark mineral grains can still pass the
    void detector's upper-bound shape gates.  A second gate — the
    ``void_intensity_max`` brightness filter — therefore drops any void
    keypoint whose disc isn't actually near-black, mirroring the gate in
    ``contour_detection``.  This makes the void/inclusion classification
    effectively mutually exclusive on common pottery samples.

    ``blob_params`` overrides are applied to all three detectors.  Note that
    ``blobColor`` and ``filterByColor`` are set internally per detector and
    should NOT be overridden.

    blob.size represents the diameter of the detected blob in pixels.
    To convert to real-world measurements, use: diameter_cm = blob.size / (scan_dpi * 0.3937)
    """
    # Validate DPI input
    if scan_dpi < 150 or scan_dpi > 2400:
        print(f"Warning: scan_dpi {scan_dpi} is outside recommended range (150-2400). Results may be unreliable.")

    # Input validation for numpy array
    if image is None or image.size == 0:
        print(f"Warning: Invalid image data provided")
        return [], []

    im = image.copy()

    if not channels:
        raise ValueError("`channels` must contain at least one entry")
    for ch in channels:
        if ch not in _VALID_CHANNELS:
            raise ValueError(
                f"Unknown channel {ch!r}; expected one of {_VALID_CHANNELS}")

    # DPI-scaled Gaussian blur to reduce noise before thresholding.
    # Base: 5×5 @ 600 DPI, 11×11 @ 1200 DPI, 21×21 @ 2400 DPI.
    # blur_scale is a user tuning knob — raise for noisier scans, lower
    # for crisper ones.  Default 1.0 is calibrated for the CLAHE +
    # BGR-vote pipeline; heavier blur smears shape detail and lets dark
    # mineral grains slip through the void detector's upper-bound shape
    # gates, inflating void counts while dropping inclusion counts.
    blur_k = int(round(scan_dpi / 600.0 * 5 * blur_scale))
    blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1  # must be odd
    blur_k = max(3, blur_k)  # minimum 3×3

    def _apply_overrides(params):
        """Apply user blob_params overrides to a detector param object."""
        if blob_params:
            for key, val in blob_params.items():
                if hasattr(params, key):
                    setattr(params, key, val)
        return params

    def _detect_one_channel(gray_blur):
        """Run the three blob detectors on a single blurred channel."""
        # 1. Light inclusions (bright features on darker background)
        light_params = _apply_overrides(
            setup_robust_blob_params(gray_blur, scan_dpi, "light", size_params))
        light_inc = list(
            cv2.SimpleBlobDetector_create(light_params).detect(gray_blur))

        # 2. Dark inclusions (dark minerals: ferruginous grains, magnetite, biotite, dark grog)
        dark_inc_params = _apply_overrides(
            setup_robust_blob_params(gray_blur, scan_dpi, "dark_inclusion", size_params))
        dark_inc = list(
            cv2.SimpleBlobDetector_create(dark_inc_params).detect(gray_blur))

        # 3. Dark voids (pores, organic burnout channels)
        dark_void_params = _apply_overrides(
            setup_robust_blob_params(gray_blur, scan_dpi, "dark", size_params))
        dark_void = list(
            cv2.SimpleBlobDetector_create(dark_void_params).detect(gray_blur))

        # Light + dark inclusions are pooled per channel.  Voids are kept
        # independent (dark_inclusion's strict shape filters and the void
        # detector's upper-bound shape filters form complementary
        # discriminators, so overlap is minimal).
        return light_inc + dark_inc, dark_void

    def _gate_inclusions_by_pop(inclusion_blobs, raw_bgr):
        """Drop inclusion keypoints with no real center-vs-ring contrast.

        Sampled on the **raw (pre-CLAHE) BGR channels** of the input image
        and taken as the max absolute (core_mean - annulus_mean) across
        the three channels.  CLAHE on uniform paste manufactures pseudo-
        blobs whose intensity is locally correlated across channels and
        therefore survives the BGR voting step, but on the original
        pixels those locations have almost no actual differential
        between center and surround.  Real inclusions pop on at least
        one channel.
        """
        if (inclusion_pop_min is None or inclusion_pop_min <= 0
                or not inclusion_blobs):
            return inclusion_blobs
        if raw_bgr.ndim != 3 or raw_bgr.shape[2] < 3:
            return inclusion_blobs
        h, w = raw_bgr.shape[:2]
        kept = []
        for kp in inclusion_blobs:
            x = int(round(kp.pt[0]))
            y = int(round(kp.pt[1]))
            r = max(2, int(round(kp.size / 2)))
            R = int(round(r * 2.0))
            y0, y1 = max(0, y - R), min(h, y + R + 1)
            x0, x1 = max(0, x - R), min(w, x + R + 1)
            yy = np.arange(y0 - y, y1 - y).reshape(-1, 1)
            xx = np.arange(x0 - x, x1 - x).reshape(1, -1)
            d2 = yy * yy + xx * xx
            core_mask = d2 <= (r * r)
            ring_mask = (d2 > r * r) & (d2 <= R * R)
            best = 0.0
            for c in range(3):
                patch = raw_bgr[y0:y1, x0:x1, c]
                if patch.size == 0:
                    continue
                # Exclude masked-out background (zero) from both regions.
                valid = patch > 0
                core_sel = core_mask & valid
                ring_sel = ring_mask & valid
                if core_sel.sum() < 3 or ring_sel.sum() < 3:
                    continue
                delta = abs(float(patch[core_sel].mean())
                            - float(patch[ring_sel].mean()))
                if delta > best:
                    best = delta
            if best >= inclusion_pop_min:
                kept.append(kp)
        return kept

    def _gate_voids_by_intensity(void_blobs, gray):
        """Drop void keypoints whose disc isn't actually dark.

        Mirrors the ``void_intensity_max`` gate in ``contour_detection``: a
        true pore reads near-black inside its bounds, while a dark mineral
        inclusion is just darker paste and stays well above black.  Mean is
        sampled on the unblurred channel within the keypoint's circular
        footprint (``kp.size / 2`` radius).
        """
        if void_intensity_max is None or not void_blobs:
            return void_blobs
        h, w = gray.shape[:2]
        kept = []
        for kp in void_blobs:
            x = int(round(kp.pt[0]))
            y = int(round(kp.pt[1]))
            r = max(1, int(round(kp.size / 2)))
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            yy = np.arange(y0 - y, y1 - y).reshape(-1, 1)
            xx = np.arange(x0 - x, x1 - x).reshape(1, -1)
            disc = (yy * yy + xx * xx) <= (r * r)
            vals = patch[disc]
            if vals.size == 0:
                continue
            if float(vals.mean()) < void_intensity_max:
                kept.append(kp)
        return kept

    inc_by_channel = {}
    void_by_channel = {}
    for ch in channels:
        gray = _extract_channel(im, ch,
                                enhance_contrast=enhance_contrast,
                                clip_limit=clahe_clip, tile_grid=clahe_grid)
        gray_blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        inc_ch, void_ch = _detect_one_channel(gray_blur)
        # Brightness gate uses the unblurred channel so the disc-mean
        # reflects the true interior darkness rather than the blurred halo.
        void_ch = _gate_voids_by_intensity(void_ch, gray)
        inc_by_channel[ch] = inc_ch
        void_by_channel[ch] = void_ch

    if len(channels) == 1:
        only = channels[0]
        return (_gate_inclusions_by_pop(inc_by_channel[only], im),
                void_by_channel[only])

    blobs_inclusions = _combine_blob_lists(inc_by_channel, combine_mode, vote_min)
    blobs_voids = _combine_blob_lists(void_by_channel, combine_mode, vote_min)
    blobs_inclusions = _gate_inclusions_by_pop(blobs_inclusions, im)
    return blobs_inclusions, blobs_voids


def detect_multiple_sherds(sherd_scan, scan_dpi=1200, crop_buffer=125,
                           auto_crop=True, n_sherds=None,
                           min_area_cm2=0.75, mask=None):
    """
    Detect one or many sherds in a single scan and return per-sherd masks/crops.

    This is the multi-sherd counterpart to ``sherd_mask``.  When the scanning
    plate carries several pieces it runs the same Canny + Otsu + adaptive
    threshold pipeline used by ``sherd_mask`` but, instead of keeping only the
    largest contour, retains every contour that survives an absolute-size
    filter, a bbox-IoU deduplication pass, and a gap-based stopping rule.

    Auto-count heuristic
    --------------------
    1. Pool contours from all three methods.
    2. Drop anything smaller than ``min_area_cm2`` (DPI-aware) or larger than
       90% of the image (filters out the whole-frame contour).
    3. Sort descending by area and deduplicate any pair whose bounding boxes
       overlap with IoU > 0.5 (keeps the larger of the two — prevents the
       outer Canny ring and the filled Otsu interior of the same sherd from
       being counted twice).
    4. Walk consecutive area ratios and stop at the largest drop-off
       (``area[i] / area[i-1]`` minimum).  Everything before the gap is a
       real sherd; everything after is noise.

    If ``n_sherds`` is supplied, the gap rule is skipped and the top-N largest
    survivors are returned instead.

    Parameters
    ----------
    sherd_scan : numpy.ndarray
        The scanned image.  Expected to be BGR (as returned by ``cv2.imread``).
    scan_dpi : int, optional
        Scan resolution for adaptive parameter scaling (default: 1200).
    crop_buffer : int, optional
        Extra pixels kept beyond each sherd's bounding box on all four sides
        when ``auto_crop=True`` (default: 125).
    auto_crop : bool, optional
        If True (default), each returned ``mask`` is cropped to its sherd's
        bounding box plus ``crop_buffer``.  If False, every returned mask is
        full-image-sized.
    n_sherds : int, optional
        Override the auto-count.  When set, returns the top-N contours by
        area regardless of the gap heuristic.  Default ``None`` = auto.
    min_area_cm2 : float, optional
        Absolute lower bound on sherd area (default: 0.25 cm²).  Contours
        below this are treated as noise.
    mask : numpy.ndarray, optional
        Pre-computed multi-blob mask.  When supplied, this function skips
        the edge pipeline and runs connected-components on ``mask`` instead.
        Useful for callers that already have a mask from a different source.
        Note: ``sherd_mask`` only ever produces a single-blob mask, so do
        **not** pass its output here.

    Returns
    -------
    list of dict
        One entry per detected sherd, sorted descending by area.  Each entry
        has the same keys ``sherd_mask`` would expose plus a few extras::

            {
                'mask':     mask_slice,       # cropped+padded binary mask (uint8)
                'color_mask': color_mask_slice,  # 3-channel version of `mask`
                'crop':     (y1, y2, x1, x2, pad_top, pad_bottom, pad_left, pad_right),
                'contour':  contour,          # in image_cropped coordinates
                'bbox':     (x, y, w, h),     # bbox in image_cropped coords
                'centroid': (cx, cy),         # centroid in image_cropped coords
                'area':     area_px,          # contour area in pixels
                'area_cm2': area_cm2,         # contour area in cm²
            }

        Returns an empty list if no sherd survives the filters.
    """
    if scan_dpi < 150 or scan_dpi > 2400:
        print(f"Warning: scan_dpi {scan_dpi} outside recommended range (150-2400)")

    image = sherd_scan
    orig_h, orig_w = image.shape[:2]
    dpcm = scan_dpi * 0.3937
    # Skip the scanner-box border crop on images too small to spare it
    # (e.g. tight per-sherd crops from split_multi_sherd_scan).
    desired_border = int(0.5 * dpcm)
    if min(orig_h, orig_w) >= 8 * desired_border:
        border_crop = desired_border
    else:
        border_crop = 0
    image_cropped = image[border_crop:orig_h - border_crop, border_crop:orig_w - border_crop]
    cropped_h, cropped_w = image_cropped.shape[:2]
    image_area = cropped_h * cropped_w

    if mask is not None:
        m = mask
        if len(m.shape) > 2:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        # The supplied mask is in original-image coords; pull out the
        # interior that corresponds to image_cropped.
        m_cropped = m[border_crop:orig_h - border_crop,
                      border_crop:orig_w - border_crop]
        fg_mask = (m_cropped > 0).astype(np.uint8) * 255
    else:
        # CLAHE-enhanced V + dual-Otsu trimap → GrabCut, single pass for
        # the whole image_cropped.  Inter-sherd gaps in the background
        # show through naturally because the loose Otsu mask leaves them
        # uncovered, so the trimap's GC_BGD reaches between sherds and
        # separate contours fall out of findContours downstream.
        fg_mask = _grabcut_mask(image_cropped, scan_dpi)

    contours = _select_multiple_contours(
        fg_mask, image_area, scan_dpi,
        n_sherds=n_sherds, min_area_cm2=min_area_cm2,
    )

    if not contours:
        return []

    results = []
    for contour in contours:
        mask_slice, crop = _contour_to_crop_and_mask(
            contour, orig_h, orig_w, border_crop, crop_buffer, auto_crop
        )
        color_mask_slice = np.dstack((mask_slice, mask_slice, mask_slice))
        area_px = cv2.contourArea(contour)
        x_br, y_br, w_br, h_br = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx = x_br + w_br / 2
            cy = y_br + h_br / 2

        results.append({
            'mask': mask_slice,
            'color_mask': color_mask_slice,
            'crop': crop,
            'contour': contour,
            'bbox': (x_br, y_br, w_br, h_br),
            'centroid': (cx, cy),
            'area': area_px,
            'area_cm2': area_px / (dpcm ** 2),
        })

    results.sort(key=lambda r: r['area'], reverse=True)
    return results


def split_multi_sherd_scan(image_path, output_dir, scan_dpi=1200,
                           crop_buffer=125, n_sherds=None, min_area_cm2=0.25,
                           write_manifest=True, manifest_path=None,
                           apply_mask_to_output=False):
    """
    Split a (possibly multi-sherd) scan into one cropped image per sherd and
    write them to ``output_dir`` so ``full_analysis`` can consume them.

    The output naming convention is::

        N == 1 : <stem>.<ext>           (no suffix; behaves like a normal single-sherd scan)
        N >= 2 : <stem>_1.<ext>, <stem>_2.<ext>, ...

    The shared ``<stem>`` is the original filename's stem, so downstream CSV
    rows (``filename`` column from ``full_analysis``) trace back to the source
    scan trivially.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the source scan.
    output_dir : str or pathlib.Path
        Directory to write cropped per-sherd images into.  Created if missing.
    scan_dpi : int, optional
        Scan resolution (default: 1200).
    crop_buffer : int, optional
        Pixels of padding around each sherd in the output crop (default: 125).
    n_sherds : int, optional
        Force a specific number of sherds.  Default ``None`` = auto-detect.
    min_area_cm2 : float, optional
        Minimum sherd area (default: 0.25 cm²).
    write_manifest : bool, optional
        If True (default), append a row per output file to ``manifest.csv``
        in ``output_dir`` mapping it back to its source.
    manifest_path : str or pathlib.Path, optional
        Override the default manifest location (``output_dir/manifest.csv``).
    apply_mask_to_output : bool, optional
        If True, multiply each output crop by its mask so the background is
        black.  Default ``False`` — write the raw crop so downstream
        ``sherd_mask`` can re-derive an accurate boundary.

    Returns
    -------
    list of pathlib.Path
        Paths of the written per-sherd images, in detection order
        (largest first).
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    sherds = detect_multiple_sherds(
        image, scan_dpi=scan_dpi, crop_buffer=crop_buffer,
        auto_crop=True, n_sherds=n_sherds, min_area_cm2=min_area_cm2,
    )

    if not sherds:
        print(f"  No sherds detected in {image_path.name}")
        return []

    stem = image_path.stem
    ext = image_path.suffix
    output_paths = []
    manifest_rows = []

    for i, sherd in enumerate(sherds, start=1):
        if len(sherds) == 1:
            out_name = f"{stem}{ext}"
        else:
            out_name = f"{stem}_{i}{ext}"
        out_path = output_dir / out_name

        y1, y2, x1, x2, pad_top, pad_bottom, pad_left, pad_right = sherd['crop']
        crop_img = image[y1:y2, x1:x2]
        if pad_top or pad_bottom or pad_left or pad_right:
            crop_img = np.pad(
                crop_img,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant', constant_values=0,
            )

        if apply_mask_to_output:
            crop_img = cv2.bitwise_and(crop_img, sherd['color_mask'])

        cv2.imwrite(str(out_path), crop_img)
        output_paths.append(out_path)

        manifest_rows.append({
            'output_file': out_name,
            'source_file': image_path.name,
            'source_path': str(image_path),
            'sherd_index': i,
            'sherd_count': len(sherds),
            'bbox_x': int(sherd['bbox'][0]),
            'bbox_y': int(sherd['bbox'][1]),
            'bbox_w': int(sherd['bbox'][2]),
            'bbox_h': int(sherd['bbox'][3]),
            'area_cm2': float(sherd['area_cm2']),
        })

    if write_manifest and manifest_rows:
        if manifest_path is None:
            manifest_path = output_dir / 'manifest.csv'
        else:
            manifest_path = Path(manifest_path)
        _append_manifest(manifest_path, manifest_rows)

    return output_paths


def _append_manifest(manifest_path, rows):
    """Append ``rows`` (list of dict) to ``manifest_path``, writing header if new."""
    import csv
    manifest_path = Path(manifest_path)
    fieldnames = ['output_file', 'source_file', 'source_path', 'sherd_index',
                  'sherd_count', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                  'area_cm2']
    write_header = not manifest_path.exists()
    with open(manifest_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def prepare_multi_sherd_directory(input_dir, output_dir, scan_dpi=1200,
                                  crop_buffer=125, n_sherds=None,
                                  min_area_cm2=0.25, file_formats=None,
                                  write_manifest=True,
                                  apply_mask_to_output=False):
    """
    Batch wrapper for ``split_multi_sherd_scan``.

    Iterates every image in ``input_dir`` (recursively), splits each one,
    and writes the per-sherd crops into ``output_dir`` with consistent
    ``<stem>[_N].<ext>`` naming.  A single combined ``manifest.csv`` is
    written into ``output_dir`` so every output file can be traced back to
    its source scan.

    Parameters
    ----------
    input_dir : str or pathlib.Path
        Directory of source scans (each scan may contain 1+ sherds).
    output_dir : str or pathlib.Path
        Directory to write per-sherd images into.
    scan_dpi, crop_buffer, n_sherds, min_area_cm2, apply_mask_to_output
        Forwarded to ``split_multi_sherd_scan`` and
        ``detect_multiple_sherds``.
    file_formats : list of str, optional
        Extensions to look for.  Default:
        ``['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']``.
    write_manifest : bool, optional
        Write a combined ``manifest.csv`` in ``output_dir`` (default True).

    Returns
    -------
    list of pathlib.Path
        All per-sherd image paths that were written.
    """
    if file_formats is None:
        file_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'manifest.csv' if write_manifest else None

    image_files = []
    for ext in file_formats:
        image_files.extend(list(input_dir.rglob(f'*.{ext}')))
        image_files.extend(list(input_dir.rglob(f'*.{ext.upper()}')))

    # rglob returns the same file twice on case-insensitive filesystems (macOS).
    image_files = sorted({p.resolve() for p in image_files})

    all_outputs = []
    for i, image_path in enumerate(image_files, start=1):
        print(f"Splitting {image_path.name} ({i}/{len(image_files)})")
        try:
            outs = split_multi_sherd_scan(
                image_path, output_dir,
                scan_dpi=scan_dpi, crop_buffer=crop_buffer,
                n_sherds=n_sherds, min_area_cm2=min_area_cm2,
                write_manifest=write_manifest,
                manifest_path=manifest_path,
                apply_mask_to_output=apply_mask_to_output,
            )
            all_outputs.extend(outs)
            print(f"  -> wrote {len(outs)} sherd image(s)")
        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}")

    print(f"Done. Wrote {len(all_outputs)} per-sherd images to {output_dir}")
    return all_outputs


def enhanced_contour_detection(image, scan_dpi=1200, size_params=None, shape_params=None,
                      morph_kernel_mm=2.5, debug_mode=False):
    """
    Contour-based detection using the exact cv2_test.py methodology for individual inclusions.

    This implements the approach from "Trying to find contours for individual inclusions":
    1. Threshold at 127 (not 125 from contour_counter)
    2. Find contours using RETR_TREE, CHAIN_APPROX_SIMPLE
    3. Sort by area (largest first)
    4. Filter by solidity (convex hull ratio > 0.7)

    Parameters
    ----------
    image : numpy.ndarray
        Image array of a scanned sherd (not file path)
    scan_dpi : int, optional
        Scan resolution in dots per inch (default: 1200)
        Valid range: 150-2400 DPI
    size_params : dict, optional
        Size filtering parameters to override defaults. If None, uses:

        **Inclusions:**
        - min: 0.1 mm (smallest grain size for v. fine sand, Wentworth scale after accounting for elbow in chart)
        - max: 15mm (very coarse gravel)

        **Voids:**
        - min: 0.25mm (macroscopic voids from organic burnout)
        - max: 15mm (larger voids are likely artifacts)

        To override, provide a dict with:
        - 'user_override': bool, must be True to enable custom sizes
        - 'min_inclusion_area_px': int, minimum inclusion area in pixels
        - 'max_inclusion_area_px': int, maximum inclusion area in pixels
        - 'min_void_area_px': int, minimum void area in pixels
        - 'max_void_area_px': int, maximum void area in pixels

        Example for detecting features up to 2cm diameter at 1200 DPI::

            dpcm = 1200 * 0.3937  # ~472 dots per cm
            max_area = int(np.pi * (2.0 / 2 * dpcm) ** 2)  # 2cm diameter
            size_params = {
                'user_override': True,
                'min_inclusion_area_px': 50,
                'max_inclusion_area_px': max_area,
                'min_void_area_px': 100,
                'max_void_area_px': max_area
            }
    shape_params : dict, optional
        Override the hardcoded shape-quality thresholds used to filter contours.
        If None, calibrated defaults are used.

        Keys:
        - inclusion_solidity_min (float, default 0.6)
          Ratio of contour area to convex hull area.
          Lower values (e.g. 0.3) accept more irregular, angular grains;
          higher values (e.g. 0.9) restrict to nearly-convex shapes only.
          Sub-angular to rounded grains typical of ceramic fabric score 0.6–0.95.
        - inclusion_compactness_min (float, default 0.25)
          4π · area / perimeter².  A perfect circle = 1.0.
          Lower values accept more irregular outlines (e.g. angular grog fragments);
          higher values (e.g. 0.5) restrict to rounder, more compact grains.
        - void_solidity_min (float, default 0.1)
          Solidity threshold for void contours.  More permissive than for inclusions
          because firing voids from organic burnout can be very irregular.
        - void_compactness_min (float, default 0.25)
          4π · area / perimeter² threshold for void contours.  Filters out
          wiggly, sinuous void shapes with irregular perimeters.
        - inclusion_max_aspect_ratio (float, default 3.0)
          **Primary shape filter.**  Maximum allowed ratio of the longer side
          to the shorter side of the minimum-area bounding rectangle (from
          ``cv2.minAreaRect``).  Contours exceeding this ratio are rejected as
          wire-thin scan artifacts (dead pixel rows, calibration lines, thin
          scratches).  This is the direct contour-detection counterpart of the
          blob detector's ``minInertiaRatio`` filter:

              inclusion_max_aspect_ratio = 1 / minInertiaRatio
              3.0  ↔  minInertiaRatio = 0.333  (the shared default for both detectors)

           Decreasing accepts fewer shapes (more equant only); increasing passes
           more elongated contours.  Most ceramic inclusions (biotite laths,
           elongated grog) fall in the 2:1–4:1 range and are safely captured
           by the 3:1 default.  Wire-thin artifacts typically exceed 10:1.
        - void_max_aspect_ratio (float, default 5.0)
            Maximum aspect ratio for void contours.  Voids can be more elongated

        Example — strict detection, convex grains only::

            cr = amacfa_plus.contour_detection(
                masked_img, scan_dpi=SCAN_DPI,
                shape_params={
                    'inclusion_solidity_min': 0.85,
                    'inclusion_compactness_min': 0.45,
                }
            )

        Example — permissive detection, captures angular / irregular grains::

            cr = amacfa_plus.contour_detection(
                masked_img, scan_dpi=SCAN_DPI,
                shape_params={
                    'inclusion_solidity_min': 0.3,
                    'inclusion_compactness_min': 0.1,
                    'void_solidity_min': 0.1,
                }
            )
    morph_kernel_mm : float, optional
        Diameter (in mm) of the elliptical structuring element used for the
        morphological tophat/blackhat transforms that isolate inclusions and
        voids from the paste background.  Default: 2.0 mm (~94 px at 1200 DPI).
        Features smaller than the kernel are fully highlighted; larger features
        are detected via their perimeter contrast.  Increase for coarser
        fabrics with very large inclusions; decrease for fine-grained pastes.
    debug_mode : bool, optional
        If True, prints a summary of candidate counts and filter decisions (default: False)

    Returns
    -------
    dict
        Dictionary containing:
        - 'inclusions': list of inclusion contours (cv2 contour arrays)
        - 'voids': list of void contours (cv2 contour arrays)
        - 'inclusion_areas': list of inclusion areas in cm²
        - 'void_areas': list of void areas in cm²
        - 'total_inclusions': count of inclusions
        - 'total_voids': count of voids
        - 'debug_info': dict with candidate counts, filter thresholds, and rejection breakdown
    """
    # Validate inputs
    if image is None or image.size == 0:
        print("Warning: Invalid image data provided")
        return {
            'inclusions': [], 'voids': [], 
            'inclusion_areas': [], 'void_areas': [],
            'total_inclusions': 0, 'total_voids': 0
        }
    
    if scan_dpi < 150 or scan_dpi > 2400:
        print(f"Warning: scan_dpi {scan_dpi} is outside recommended range (150-2400)")
    
    # Convert to L* (lightness) channel from CIELAB colour space
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 0]
    else:
        gray = image.copy()

    # Calculate DPI-aware parameters
    dpcm = scan_dpi * 0.3937  # dots per cm

    # Morphological structuring element for tophat/blackhat transforms
    kernel_px = max(3, int(morph_kernel_mm * 0.1 * dpcm))  # mm → cm → px
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))

    # INCLUSION size filtering: derive pixel thresholds from physical dimensions
    # Min: 0.1mm diameter (fine silt boundary, Wentworth scale)
    # Max: 15mm diameter (very coarse gravel temper)
    inc_min_diameter_cm = 0.01  # 0.1 mm
    inc_max_diameter_cm = 1.5    # 15 mm 
    min_area_threshold = int(np.pi * (inc_min_diameter_cm / 2 * dpcm) ** 2)
    max_area_threshold = int(np.pi * (inc_max_diameter_cm / 2 * dpcm) ** 2)

    # VOID size filtering: larger minimum since macroscopic voids from organic burnout
    # Min: 0.25mm diameter (macroscopic void threshold)
    # Max: 15mm diameter (voids larger than this are likely artifacts)
    void_min_diameter_cm = 0.025  # 0.25 mm
    void_max_diameter_cm = 1.5    # 15 mm 
    void_min_area_threshold = int(np.pi * (void_min_diameter_cm / 2 * dpcm) ** 2)
    void_max_area_threshold = int(np.pi * (void_max_diameter_cm / 2 * dpcm) ** 2)

    if size_params and size_params.get('user_override', False):
        # Use user-specified size limits if provided
        min_area_threshold = size_params.get('min_inclusion_area_px', min_area_threshold)
        max_area_threshold = size_params.get('max_inclusion_area_px', max_area_threshold)
        void_min_area_threshold = size_params.get('min_void_area_px', void_min_area_threshold)
        void_max_area_threshold = size_params.get('max_void_area_px', void_max_area_threshold)
    
    # FEATURE ISOLATION — tophat (bright) + blackhat (dark) morphological transforms.
    # Unlike global thresholding, these extract features based on *local* contrast
    # relative to the structuring element, preventing adjacent features of different
    # polarity from merging and improving detection in uneven paste backgrounds.
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)  # slight blur to reduce noise sensitivity
    tophat = cv2.morphologyEx(gray_blur, cv2.MORPH_TOPHAT, morph_kernel)    # bright features
    blackhat = cv2.morphologyEx(gray_blur, cv2.MORPH_BLACKHAT, morph_kernel)  # dark features


    # Combine – both bright and dark grains become bright
    combined = cv2.bitwise_or(tophat, blackhat)   # or use np.maximum(tophat, blackhat)

    # ----------------------------------------------------------------------
    # 4. Initial binarisation with Otsu
    # ----------------------------------------------------------------------
    _, binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    all_contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Otsu threshold on each transform output
    #_, th_light = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #_, th_dark = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours SEPARATELY, then concatenate — avoids merging adjacent
    # features of different polarity that would fuse in a bitwise_or.
    #contours_light, _ = cv2.findContours(th_light, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours_dark, _ = cv2.findContours(th_dark, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #all_contours = contours_light + contours_dark

    
    # INCLUSION candidates — direct size filter (no drop-largest trick needed;
    # any sherd-boundary contour exceeds max_area_threshold and is excluded).
    sel = [c for c in all_contours
           if min_area_threshold < cv2.contourArea(c) < max_area_threshold]
    
    # Shape-quality thresholds — calibrated defaults, overridable via shape_params.
    # inclusion_max_aspect_ratio is the primary elongation gate applied FIRST in the filter chain;
    # solidity and compactness are secondary convexity/regularity checks.
    # CROSS-METHOD CONSISTENCY: max_aspect_ratio = 1 / minInertiaRatio (blob detector)
    #   → max_aspect_ratio 5.0  ↔  minInertiaRatio 0.2  (both defaults identical)
    inclusion_max_aspect_ratio = 4.0    # primary filter: rejects contours with long/short > 4:1
                                       # matches blob default minInertiaRatio=0.333 exactly
    void_max_aspect_ratio     = 5.0    # voids can be more elongated than inclusions, but still filter out wire-thin artifacts
    inclusion_solidity_min    = 0.45    # secondary: area / convex-hull area
    inclusion_compactness_min = 0.25   # secondary: 4π·area / perimeter²
    void_solidity_min         = 0.1    # secondary (voids only, more permissive)
    if shape_params:
        # Primary filter first
        inclusion_max_aspect_ratio = shape_params.get('inclusion_max_aspect_ratio', inclusion_max_aspect_ratio)
        # Secondary filters
        inclusion_solidity_min    = shape_params.get('inclusion_solidity_min',    inclusion_solidity_min)
        inclusion_compactness_min = shape_params.get('inclusion_compactness_min', inclusion_compactness_min)
        void_solidity_min         = shape_params.get('void_solidity_min',         void_solidity_min)

    # Apply shape-quality filtering to inclusion candidates
    inclusion_contours = []
    inclusion_areas = []
    debug_info = {
        'morph_kernel_mm': morph_kernel_mm,
        'morph_kernel_px': kernel_px,
        'contours_from_top_and_blackhat': len(combined),
        'total_candidates': len(sel),
        'inclusion_accepted': 0,
        'inclusion_rejected_solidity': 0,
        'inclusion_rejected_compactness': 0,
        'void_accepted': 0,
        'void_rejected': 0,
        'solidity_threshold': inclusion_solidity_min,
        'compactness_threshold': inclusion_compactness_min,
        'void_solidity_threshold': void_solidity_min,
        'inclusion_max_aspect_ratio': inclusion_max_aspect_ratio,
        'void_max_aspect_ratio': void_max_aspect_ratio
    }

    for contour in sel:
        area_pixels = cv2.contourArea(contour)
        # Calculate solidity: area / convex hull area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area > 0:
            solidity = float(area_pixels) / hull_area

            # Calculate compactness to filter out wiggly, sinuous shapes
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                compactness = (4 * np.pi * area_pixels) / (perimeter ** 2)
            else:
                compactness = 0

            # PRIMARY FILTER: aspect ratio from minAreaRect.
            # Equivalent of blob detector's minInertiaRatio (max_aspect_ratio = 1/minInertiaRatio).
            # Applied before solidity/compactness — a clean-edged narrow rectangle is convex
            # (passes solidity) and has regular perimeter (passes compactness), so without this
            # filter wire-thin scan artifacts would be silently accepted by the other two checks.
            _, (rw, rh), _ = cv2.minAreaRect(contour)
            aspect_ratio = (max(rw, rh) / max(min(rw, rh), 1e-6))

            if (aspect_ratio <= inclusion_max_aspect_ratio           # primary: elongation gate
                    and solidity > inclusion_solidity_min  # secondary: convexity
                    and compactness > inclusion_compactness_min):  # secondary: perimeter regularity
                inclusion_contours.append(contour)
                area_cm2 = area_pixels / (dpcm ** 2)
                inclusion_areas.append(area_cm2)
                debug_info['inclusion_accepted'] += 1
            elif solidity <= inclusion_solidity_min:
                debug_info['inclusion_rejected_solidity'] += 1
            elif compactness <= inclusion_compactness_min:
                debug_info['inclusion_rejected_compactness'] += 1
            # (aspect ratio rejections counted implicitly in total_candidates - accepted)
    
     # VOID DETECTION - Use OTSU thresholding approach
    _, thresh_voids = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Find void contours
    contours_voids, _ = cv2.findContours(thresh_voids, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Same approach for voids: drop the largest contour (sherd boundary),
    # filter by DPI-derived size limits.
    size_sorted_contours_voids = sorted(contours_voids, key=cv2.contourArea, reverse=True)
    if len(size_sorted_contours_voids) > 1:
        sel_voids = [c for c in size_sorted_contours_voids[1:]
                     if void_min_area_threshold < cv2.contourArea(c) < void_max_area_threshold]
    else:
        sel_voids = []

    void_contours = []
    void_areas = []
    debug_info['void_candidates'] = len(sel_voids)
    
    for contour in sel_voids:
        area_pixels = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area > 0:
            solidity = float(area_pixels) / hull_area

            _, (rw, rh), _ = cv2.minAreaRect(contour)
            aspect_ratio = (max(rw, rh) / max(min(rw, rh), 1e-6))

            # Solidity for voids — permissive by default, overridable via shape_params
            # Aspect ratio shared with inclusions: organic-burnout voids can be elongated
            # but not wire-thin artifacts.
            if solidity > void_solidity_min and aspect_ratio <= void_max_aspect_ratio:
                void_contours.append(contour)
                area_cm2 = area_pixels / (dpcm ** 2)
                void_areas.append(area_cm2)
                debug_info['void_accepted'] += 1
            else:
                debug_info['void_rejected'] += 1
    
    if debug_mode:
        di = debug_info
        total_rej = di['inclusion_rejected_solidity'] + di['inclusion_rejected_compactness']
        print(f"[contour_detection debug]")
        print(f"  Morph kernel                       : {di['morph_kernel_mm']}mm ({di['morph_kernel_px']}px)")
        print(f"  Contours from tophat (bright)      : {di['contours_from_tophat']}")
        print(f"  Contours from blackhat (dark)      : {di['contours_from_blackhat']}")
        print(f"  Size-filtered inclusion candidates : {di['total_candidates']}")
        print(f"  Accepted inclusions                : {di['inclusion_accepted']}")
        print(f"  Rejected – solidity < {di['solidity_threshold']:.2f}          : {di['inclusion_rejected_solidity']}")
        print(f"  Rejected – compactness < {di['compactness_threshold']:.2f}       : {di['inclusion_rejected_compactness']}")
        print(f"  Size-filtered void candidates      : {di.get('void_candidates', '?')}")
        print(f"  Accepted voids                     : {di['void_accepted']}")
        print(f"  Rejected voids                     : {di['void_rejected']}")
        if di['total_candidates'] > 0:
            rate = total_rej / di['total_candidates'] * 100
            print(f"  Shape-filter rejection rate        : {rate:.0f}%")

    # GEOMETRIC ANGULARITY ANALYSIS - New Feature for Temper Analysis
    from .analysis import analyze_inclusion_angularity
    
    # Analyze geometric properties of inclusions for archaeological interpretation
    if len(inclusion_contours) > 0:
        geometric_analysis = analyze_inclusion_angularity(inclusion_contours, scan_dpi)
    else:
        geometric_analysis = analyze_inclusion_angularity([], scan_dpi)


    return {
        'inclusions': inclusion_contours,
        'voids': void_contours,
        'inclusion_areas': inclusion_areas,
        'void_areas': void_areas,
        'total_inclusions': len(inclusion_contours),
        'total_voids': len(void_contours),
        'debug_info': debug_info,
        'geometric_analysis': geometric_analysis,
    }

def contour_detection(image, scan_dpi=1200, size_params=None, shape_params=None,
                      debug_mode=False, blur_scale=1.0,
                      channels=('B', 'G', 'R'), combine_mode='vote', vote_min=2,
                      enhance_contrast=True, clahe_clip=2.0, clahe_grid=(8, 8),
                      void_intensity_max=60.0, inclusion_pop_min=20.0):
    """
    Contour-based detection using the exact cv2_test.py methodology for individual inclusions.

    This implements the approach from "Trying to find contours for individual inclusions":
    1. Threshold at 127 (not 125 from contour_counter)
    2. Find contours using RETR_TREE, CHAIN_APPROX_SIMPLE
    3. Sort by area (largest first)
    4. Filter by solidity (convex hull ratio > 0.7)

    Parameters
    ----------
    image : numpy.ndarray
        Image array of a scanned sherd (not file path)
    scan_dpi : int, optional
        Scan resolution in dots per inch (default: 1200)
        Valid range: 150-2400 DPI
    size_params : dict, optional
        Size filtering parameters to override defaults. If None, uses:

        **Inclusions:**
        - min: 0.1 mm (smallest grain size for v. fine sand, Wentworth scale after accounting for elbow in chart)
        - max: 15mm (very coarse gravel)

        **Voids:**
        - min: 0.25mm (macroscopic voids from organic burnout)
        - max: 15mm (larger voids are likely artifacts)

        To override, provide a dict with:
        - 'user_override': bool, must be True to enable custom sizes
        - 'min_inclusion_area_px': int, minimum inclusion area in pixels
        - 'max_inclusion_area_px': int, maximum inclusion area in pixels
        - 'min_void_area_px': int, minimum void area in pixels
        - 'max_void_area_px': int, maximum void area in pixels

        Example for detecting features up to 2cm diameter at 1200 DPI::

            dpcm = 1200 * 0.3937  # ~472 dots per cm
            max_area = int(np.pi * (2.0 / 2 * dpcm) ** 2)  # 2cm diameter
            size_params = {
                'user_override': True,
                'min_inclusion_area_px': 50,
                'max_inclusion_area_px': max_area,
                'min_void_area_px': 100,
                'max_void_area_px': max_area
            }
    shape_params : dict, optional
        Override the hardcoded shape-quality thresholds used to filter contours.
        If None, calibrated defaults are used.

        Keys:
        - inclusion_solidity_min (float, default 0.45)
          Ratio of contour area to convex hull area.
          Lower values (e.g. 0.3) accept more irregular, angular grains;
          higher values (e.g. 0.9) restrict to nearly-convex shapes only.
          The 0.45 default is permissive enough to capture angular ceramic
          temper (sub-angular to rounded grains in ceramic fabric score
          0.6–0.95 still pass with margin).
        - inclusion_compactness_min (float, default 0.125)
          4π · area / perimeter².  A perfect circle = 1.0.
          Lower values accept more irregular outlines (e.g. angular grog fragments);
          higher values (e.g. 0.5) restrict to rounder, more compact grains.
        - void_solidity_min (float, default 0.1)
          Solidity lower bound for void contours.  More permissive than for
          inclusions because firing voids from organic burnout can be very
          irregular.
        - void_compactness_min (float, default 0.06)
          4π · area / perimeter² lower bound for void contours.  Very permissive
          since organic-burnout voids can have highly irregular perimeters;
          tighter checks are handled by aspect ratio and the boundary-band gate.
        - void_solidity_max (float, default 1.01)
          Optional solidity *upper* bound for void contours.  In principle
          voids are concave (low solidity) and inclusions are convex, but the
          DPI-scaled blur + contour-simplification pipeline rounds out
          concavities so on real masked sherds nearly all dark contours end
          up with solidity ≥ 0.5 regardless of class.  The default leaves
          this gate effectively disabled; tighten it only if you know your
          scans preserve concavity well.
        - void_intensity_max (float in 0..255, default 60)
          **Primary inclusion-vs-void discriminator.**  Maximum allowed mean
          pixel intensity *inside* a void contour, measured on the channel
          being processed.  A real pore is a hole, so its interior reads
          near-black; a dark mineral inclusion is just darker paste with no
          near-black core.  Lower this for stricter void detection (e.g. 45
          to keep only deep blacks); raise it to count grey-toned cavities
          (e.g. 90 on low-contrast scans).
        - inclusion_max_aspect_ratio (float, default 4.0)
          **Primary shape filter.**  Maximum allowed ratio of the longer side
          to the shorter side of the minimum-area bounding rectangle (from
          ``cv2.minAreaRect``).  Contours exceeding this ratio are rejected as
          wire-thin scan artifacts (dead pixel rows, calibration lines, thin
          scratches).  This is the direct contour-detection counterpart of the
          blob detector's ``minInertiaRatio`` filter:

              inclusion_max_aspect_ratio = 1 / minInertiaRatio
              5.0  ↔  minInertiaRatio = 0.2  (the shared default for both detectors)

           Decreasing accepts fewer shapes (more equant only); increasing passes
           more elongated contours.  Most ceramic inclusions (biotite laths,
           elongated grog) fall in the 2:1–4:1 range and are safely captured
           by the 4:1 default.  Wire-thin artifacts typically exceed 10:1.
        - void_max_aspect_ratio (float, default 5.0)
            Maximum aspect ratio for void contours.  Voids can be more elongated
            but still filter out wire-thin artifacts.
        - edge_band_px (int, default max(5, 1.5% of shorter image dimension))
            Width of the band inside the sherd mask boundary that is treated
            as "edge."  Any candidate contour with a vertex inside this band
            is rejected as a CLAHE tile-boundary artifact or sherd-outline
            leak.  Scales with image size because CLAHE artifact band width
            tracks tile size (= image_dim / 8) rather than scan DPI; e.g.
            ~15 px on a 1000×1000 crop, ~85 px on a 5669×5669 scan.
            Set to 0 to disable.

        Example — strict detection, convex grains only::

            cr = amacfa_plus.contour_detection(
                masked_img, scan_dpi=SCAN_DPI,
                shape_params={
                    'inclusion_solidity_min': 0.85,
                    'inclusion_compactness_min': 0.45,
                }
            )

        Example — permissive detection, captures angular / irregular grains::

            cr = amacfa_plus.contour_detection(
                masked_img, scan_dpi=SCAN_DPI,
                shape_params={
                    'inclusion_solidity_min': 0.3,
                    'inclusion_compactness_min': 0.1,
                    'void_solidity_min': 0.1,
                }
            )
    debug_mode : bool, optional
        If True, prints a summary of candidate counts and filter decisions (default: False)
    channels : tuple of str, optional
        Channels to run contour detection on.  Default ``('B', 'G', 'R')``
        runs detection on each BGR channel and combines the results so
        inclusions that only contrast strongly in one channel get picked up.
        Valid entries also include ``'L'`` (CIELAB lightness) — pass
        ``channels=('L',)`` to recover the pre-multi-channel behavior.  See
        ``sherd_blobs`` for why L\* is excluded by default.
    combine_mode : {'union', 'vote'}, optional
        How to merge per-channel detections when ``len(channels) > 1``.
        Default ``'vote'`` requires a contour's centroid to fall inside the
        rasterized contour mask of at least ``vote_min`` channels — the
        calibrated sweet spot.  ``'union'`` pools detections and removes
        spatial duplicates via centroid containment without the agreement
        requirement.
    vote_min : int, optional
        Minimum number of channels that must agree for a contour to be kept
        when ``combine_mode='vote'`` (default: 2 of 3 BGR channels).
    enhance_contrast : bool, optional
        Apply CLAHE to each requested channel before detection (default:
        True).  Set to False if you've already pre-applied contrast
        enhancement to the input image — otherwise the detector handles
        CLAHE per channel internally.
    clahe_clip, clahe_grid : float / tuple, optional
        Forwarded to ``cv2.createCLAHE`` when ``enhance_contrast=True``.

    Returns
    -------
    dict
        Dictionary containing:
        - 'inclusions': list of inclusion contours (cv2 contour arrays)
        - 'voids': list of void contours (cv2 contour arrays)
        - 'inclusion_areas': list of inclusion areas in cm²
        - 'void_areas': list of void areas in cm²
        - 'total_inclusions': count of inclusions
        - 'total_voids': count of voids
        - 'debug_info': dict with candidate counts, filter thresholds, and rejection breakdown.
          When multi-channel mode is active, also contains a ``per_channel`` key
          mapping each channel to its individual debug_info.
    """
    # Validate inputs
    if image is None or image.size == 0:
        print("Warning: Invalid image data provided")
        return {
            'inclusions': [], 'voids': [],
            'inclusion_areas': [], 'void_areas': [],
            'total_inclusions': 0, 'total_voids': 0
        }

    if not channels:
        raise ValueError("`channels` must contain at least one entry")
    for ch in channels:
        if ch not in _VALID_CHANNELS:
            raise ValueError(
                f"Unknown channel {ch!r}; expected one of {_VALID_CHANNELS}")

    if scan_dpi < 150 or scan_dpi > 2400:
        print(f"Warning: scan_dpi {scan_dpi} is outside recommended range (150-2400)")

    # Calculate DPI-aware parameters
    dpcm = scan_dpi * 0.3937  # dots per cm

    # DPI-scaled Gaussian blur to reduce noise before thresholding.
    # Base: 5×5 @ 600 DPI, 11×11 @ 1200 DPI, 21×21 @ 2400 DPI.
    # blur_scale is a user tuning knob — raise for noisier scans, lower
    # for crisper ones.  Default 1.0 is calibrated for the CLAHE +
    # BGR-vote pipeline; heavier blur smears shape detail and lets dark
    # mineral grains slip through the void detector's upper-bound shape
    # gates, inflating void counts while dropping inclusion counts.
    blur_k = int(round(scan_dpi / 600.0 * 5 * blur_scale))
    blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1  # must be odd
    blur_k = max(3, blur_k)  # minimum 3×3

    # INCLUSION size filtering: thresholds in actual contour area (cm²), converted
    # to pixels via dpcm².  Using raw area instead of circular-equivalent diameter
    # so the filter works correctly for elongated shapes.
    # Min: 0.0001 cm² (≈ 0.1mm diameter circle, Wentworth v.fine sand boundary)
    # Max: 1.5 cm² (largest plausible inclusion — well below any sherd boundary)
    inc_min_area_cm2 = 0.0001   # ~0.1 mm equiv. diameter
    inc_max_area_cm2 = 1.5      # absolute area cap
    min_area_threshold = int(inc_min_area_cm2 * dpcm ** 2)
    max_area_threshold = int(inc_max_area_cm2 * dpcm ** 2)

    # VOID size filtering: larger minimum since macroscopic voids from organic burnout
    # Min: 0.0005 cm² (≈ 0.25mm diameter circle, macroscopic void threshold)
    # Max: 1.5 cm² (voids larger than this are likely artifacts)
    void_min_area_cm2 = 0.0005  # ~0.25 mm equiv. diameter
    void_max_area_cm2 = 1.5
    void_min_area_threshold = int(void_min_area_cm2 * dpcm ** 2)
    void_max_area_threshold = int(void_max_area_cm2 * dpcm ** 2)

    if size_params and size_params.get('user_override', False):
        # Use user-specified size limits if provided
        min_area_threshold = size_params.get('min_inclusion_area_px', min_area_threshold)
        max_area_threshold = size_params.get('max_inclusion_area_px', max_area_threshold)
        void_min_area_threshold = size_params.get('min_void_area_px', void_min_area_threshold)
        void_max_area_threshold = size_params.get('max_void_area_px', void_max_area_threshold)

    # Sherd-mask footprint — channel-independent because masked background is
    # zero across every input channel, so any channel's `gray > 0` agrees.
    if image.ndim == 3:
        sherd_pixels = np.any(image > 0, axis=2)
    else:
        sherd_pixels = (image > 0)
    sherd_area_px = int(np.count_nonzero(sherd_pixels))

    # Relative max-area cap: no legitimate inclusion or void should occupy
    # more than ~30% of the sherd.  Without this, small sherds (~1.5 cm²) hit
    # the absolute 1.5 cm² cap and the sherd boundary contour traced inside
    # the mask by findContours sails through as a single huge "void" (smooth
    # boundary = high solidity, high compactness, low aspect ratio).
    if sherd_area_px > 0:
        relative_max_px = int(0.30 * sherd_area_px)
        max_area_threshold = min(max_area_threshold, relative_max_px)
        void_max_area_threshold = min(void_max_area_threshold, relative_max_px)

    # Shape-quality thresholds — calibrated defaults, overridable via shape_params.
    # max_aspect_ratio is the primary elongation gate applied FIRST in the filter chain;
    # solidity and compactness are secondary convexity/regularity checks.
    # CROSS-METHOD CONSISTENCY: max_aspect_ratio = 1 / minInertiaRatio (blob detector)
    #   → max_aspect_ratio 5.0  ↔  minInertiaRatio 0.2  (the shared default)
    inclusion_max_aspect_ratio = 4.0    # primary filter: rejects contours with long/short > 4:1
    void_max_aspect_ratio      = 5.0
    inclusion_solidity_min     = 0.45   # secondary: area / convex-hull area
    inclusion_compactness_min  = 0.125  # secondary: 4π·area / perimeter²
    void_solidity_min          = 0.1    # secondary (voids only, more permissive)
    void_compactness_min       = 0.06   # secondary: 4π·area / perimeter² (voids)
    # Void solidity upper bound is left effectively unrestricted by default.
    # In principle voids are concave (low solidity) and inclusions are convex
    # (high solidity), but the DPI-scaled Gaussian blur plus
    # ``CHAIN_APPROX_SIMPLE`` contour simplification rounds out concavities,
    # so on real masked sherds almost every dark contour ends up with
    # solidity ≥ 0.5 regardless of class.  The real discriminator is below.
    void_solidity_max          = 1.01

    # Brightness-based discriminator.  A void is a *hole* in the ceramic, so
    # the pixels inside its contour read near-black; a dark mineral inclusion
    # is just paste with a darker hue, with no near-black core.  Threshold
    # comes from the top-level ``void_intensity_max`` kwarg; ``shape_params``
    # may also carry it for backward compatibility but the kwarg wins.
    # Boundary-band rejection: contours within this many pixels of the mask
    # edge are almost always CLAHE tile-boundary artifacts (bimodal histogram
    # at the mask edge creates a contrast jump), not real paste features.
    # Width scales with image size because CLAHE tile size = image_dim / 8,
    # so artifact bands on larger images are proportionally thicker.  1.5%
    # of the shorter dimension stays well under one tile width (12.5%).
    edge_band_px = max(5, int(min(image.shape[:2]) * 0.015))
    if shape_params:
        # Primary filter first
        inclusion_max_aspect_ratio          = shape_params.get('inclusion_max_aspect_ratio',          inclusion_max_aspect_ratio)
        void_max_aspect_ratio               = shape_params.get('void_max_aspect_ratio',               void_max_aspect_ratio)
        # Secondary filters
        inclusion_solidity_min    = shape_params.get('inclusion_solidity_min',    inclusion_solidity_min)
        inclusion_compactness_min = shape_params.get('inclusion_compactness_min', inclusion_compactness_min)
        void_solidity_min         = shape_params.get('void_solidity_min',         void_solidity_min)
        void_compactness_min      = shape_params.get('void_compactness_min',      void_compactness_min)
        void_solidity_max         = shape_params.get('void_solidity_max',     void_solidity_max)
        # void_intensity_max is also accepted here for backward compatibility,
        # but the top-level kwarg takes precedence when explicitly supplied.
        if 'void_intensity_max' in shape_params:
            void_intensity_max    = shape_params['void_intensity_max']
        edge_band_px              = shape_params.get('edge_band_px',          edge_band_px)

    # Build the interior mask used to reject boundary-touching contours.
    interior_mask = sherd_pixels.astype(np.uint8) * 255
    if edge_band_px > 0:
        interior_mask = cv2.erode(interior_mask, np.ones((3, 3), np.uint8),
                                  iterations=int(edge_band_px))

    def _touches_boundary(contour):
        """True if any contour vertex lies in the eroded mask boundary band."""
        h, w = interior_mask.shape
        pts = contour.reshape(-1, 2)
        step = max(1, len(pts) // 20)
        sampled = pts[::step]
        xs = np.clip(sampled[:, 0], 0, w - 1)
        ys = np.clip(sampled[:, 1], 0, h - 1)
        return bool(np.any(interior_mask[ys, xs] == 0))

    def _run_pipeline_for_channel(gray):
        """Threshold + size + shape + boundary + nested filtering on a single channel.

        Returns (inclusion_contours, inclusion_areas, void_contours, void_areas, debug_info).
        """
        # Adaptive thresholding statistics — exclude masked-out pixels.
        nonzero = gray[gray > 0]
        if nonzero.size > 0:
            mean_brightness = float(np.mean(nonzero))
            std_brightness = float(np.std(nonzero))
        else:
            mean_brightness, std_brightness = 0.0, 0.0

        gray_blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

        # INCLUSION DETECTION — dark + light thresholds on the same blurred channel.
        dark_thresh = max(30, int(mean_brightness - std_brightness))
        _, th_dark = cv2.threshold(gray_blur, dark_thresh, 255, cv2.THRESH_BINARY_INV)

        light_thresh = min(220, int(mean_brightness + std_brightness))
        _, th_light = cv2.threshold(gray_blur, light_thresh, 255, cv2.THRESH_BINARY)

        th1 = cv2.bitwise_or(th_dark, th_light)

        # Find inclusion contours (RETR_TREE, CHAIN_APPROX_SIMPLE as in cv2_test.py line 1598)
        contours_inc, _ = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Size-filter inclusion candidates.  max_area_threshold (1.5 cm²) excludes the
        # sherd boundary and any hierarchy duplicates from RETR_TREE.
        sel = [c for c in contours_inc
               if min_area_threshold < cv2.contourArea(c) < max_area_threshold]

        inclusion_contours = []
        inclusion_areas = []
        debug_info = {
            'total_candidates': len(sel),
            'inclusion_accepted': 0,
            'inclusion_rejected_solidity': 0,
            'inclusion_rejected_compactness': 0,
            'inclusion_rejected_boundary': 0,
            'void_accepted': 0,
            'void_rejected': 0,
            'void_rejected_solidity': 0,
            'void_rejected_compactness': 0,
            'void_rejected_boundary': 0,
            'void_rejected_intensity': 0,
            'solidity_threshold': inclusion_solidity_min,
            'compactness_threshold': inclusion_compactness_min,
            'void_solidity_threshold': void_solidity_min,
            'void_compactness_threshold': void_compactness_min,
            'void_solidity_max': void_solidity_max,
            'void_intensity_max': void_intensity_max,
            'inclusion_max_aspect_ratio': inclusion_max_aspect_ratio,
            'void_max_aspect_ratio': void_max_aspect_ratio,
            'edge_band_px': edge_band_px,
        }

        for contour in sel:
            # Boundary-band gate — applied before shape checks so a clean-edged
            # CLAHE artifact contour can't sneak through on solidity/compactness.
            if _touches_boundary(contour):
                debug_info['inclusion_rejected_boundary'] += 1
                continue

            area_pixels = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)

            if hull_area > 0:
                solidity = float(area_pixels) / hull_area

                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    compactness = (4 * np.pi * area_pixels) / (perimeter ** 2)
                else:
                    compactness = 0

                # PRIMARY FILTER: aspect ratio from minAreaRect.
                # Equivalent of blob detector's minInertiaRatio (max_aspect_ratio = 1/minInertiaRatio).
                # Applied before solidity/compactness — a clean-edged narrow rectangle is convex
                # (passes solidity) and has regular perimeter (passes compactness), so without this
                # filter wire-thin scan artifacts would be silently accepted by the other two checks.
                _, (rw, rh), _ = cv2.minAreaRect(contour)
                aspect_ratio = (max(rw, rh) / max(min(rw, rh), 1e-6))

                if (aspect_ratio <= inclusion_max_aspect_ratio           # primary: elongation gate
                        and solidity > inclusion_solidity_min  # secondary: convexity
                        and compactness > inclusion_compactness_min):  # secondary: perimeter regularity
                    inclusion_contours.append(contour)
                    area_cm2 = area_pixels / (dpcm ** 2)
                    inclusion_areas.append(area_cm2)
                    debug_info['inclusion_accepted'] += 1
                elif solidity <= inclusion_solidity_min:
                    debug_info['inclusion_rejected_solidity'] += 1
                elif compactness <= inclusion_compactness_min:
                    debug_info['inclusion_rejected_compactness'] += 1
                # (aspect ratio rejections counted implicitly in total_candidates - accepted)

        # VOID DETECTION — void contours are extracted from the dark-threshold
        # mask (th_dark).  The OTSU threshold below is dead code retained from
        # earlier iterations of this function.
        _, _thresh_voids_unused = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        contours_voids, _ = cv2.findContours(th_dark, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Size-filter void candidates (same logic — max threshold excludes sherd boundary)
        sel_voids = [c for c in contours_voids
                     if void_min_area_threshold < cv2.contourArea(c) < void_max_area_threshold]

        void_contours = []
        void_areas = []
        debug_info['void_candidates'] = len(sel_voids)

        for contour in sel_voids:
            # Boundary-band gate first — any "void" hugging the mask edge is the
            # CLAHE boundary artifact or the sherd outline, not a real pore.
            if _touches_boundary(contour):
                debug_info['void_rejected_boundary'] += 1
                continue

            area_pixels = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)

            if hull_area > 0:
                solidity = float(area_pixels) / hull_area

                perimeter = cv2.arcLength(contour, True)
                compactness = (4 * np.pi * area_pixels) / (perimeter ** 2) if perimeter > 0 else 0

                _, (rw, rh), _ = cv2.minAreaRect(contour)
                aspect_ratio = (max(rw, rh) / max(min(rw, rh), 1e-6))

                # Brightness gate is the primary inclusion-vs-void
                # discriminator.  Compute the mean pixel intensity inside the
                # contour from the (pre-blur) channel; a true pore reads
                # near-black there, while a dark mineral inclusion is just
                # darker paste and stays well above black.  Use a per-contour
                # bbox + filled mask so this stays O(contour_area), not
                # O(image_area).
                x, y, w, h = cv2.boundingRect(contour)
                roi = gray[y:y + h, x:x + w]
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                shifted = contour - np.array([[x, y]])
                cv2.drawContours(roi_mask, [shifted], -1, 255, cv2.FILLED)
                roi_vals = roi[roi_mask > 0]
                inside_mean = float(roi_vals.mean()) if roi_vals.size > 0 else 0.0

                shape_ok = (aspect_ratio <= void_max_aspect_ratio
                            and void_solidity_min < solidity < void_solidity_max
                            and compactness > void_compactness_min)
                # ``void_intensity_max=None`` disables the brightness gate
                # (matches the convention in ``sherd_blobs``).
                bright_ok = (void_intensity_max is None
                             or inside_mean < void_intensity_max)

                if shape_ok and bright_ok:
                    void_contours.append(contour)
                    area_cm2 = area_pixels / (dpcm ** 2)
                    void_areas.append(area_cm2)
                    debug_info['void_accepted'] += 1
                elif not bright_ok:
                    debug_info['void_rejected_intensity'] += 1
                elif solidity <= void_solidity_min or solidity >= void_solidity_max:
                    debug_info['void_rejected_solidity'] += 1
                elif compactness <= void_compactness_min:
                    debug_info['void_rejected_compactness'] += 1
                else:
                    debug_info['void_rejected'] += 1

        # Drop nested contours: a single inclusion with internal color gradient
        # otherwise registers as parent + child contours.  Same for voids.
        pre_nested_inc = len(inclusion_contours)
        inclusion_contours, inclusion_areas = _drop_nested(inclusion_contours, inclusion_areas)
        debug_info['inclusion_rejected_nested'] = pre_nested_inc - len(inclusion_contours)
        pre_nested_void = len(void_contours)
        void_contours, void_areas = _drop_nested(void_contours, void_areas)
        debug_info['void_rejected_nested'] = pre_nested_void - len(void_contours)

        return inclusion_contours, inclusion_areas, void_contours, void_areas, debug_info

    # Run the per-channel pipeline on every requested channel.
    inc_by_channel = {}
    void_by_channel = {}
    debug_per_channel = {}
    for ch in channels:
        gray = _extract_channel(image, ch,
                                enhance_contrast=enhance_contrast,
                                clip_limit=clahe_clip, tile_grid=clahe_grid)
        inc_c, inc_a, void_c, void_a, dbg = _run_pipeline_for_channel(gray)
        inc_by_channel[ch] = inc_c
        void_by_channel[ch] = void_c
        debug_per_channel[ch] = dbg

    if len(channels) == 1:
        only = channels[0]
        inclusion_contours = inc_by_channel[only]
        void_contours = void_by_channel[only]
        debug_info = debug_per_channel[only]
    else:
        inclusion_contours = _combine_contour_lists(
            inc_by_channel, image.shape, combine_mode, vote_min)
        void_contours = _combine_contour_lists(
            void_by_channel, image.shape, combine_mode, vote_min)
        # Aggregate debug_info: threshold/configuration values come from the
        # first channel (identical across all); integer counters are summed.
        threshold_keys = (
            'solidity_threshold', 'compactness_threshold',
            'void_solidity_threshold', 'void_compactness_threshold',
            'void_solidity_max', 'void_intensity_max',
            'inclusion_max_aspect_ratio', 'void_max_aspect_ratio',
            'edge_band_px',
        )
        first_dbg = debug_per_channel[channels[0]]
        debug_info = {k: first_dbg[k] for k in threshold_keys if k in first_dbg}
        counter_keys = [k for k in first_dbg if k not in threshold_keys]
        for k in counter_keys:
            debug_info[k] = sum(debug_per_channel[ch].get(k, 0) for ch in channels)
        debug_info['per_channel'] = debug_per_channel
        debug_info['combine_mode'] = combine_mode
        debug_info['vote_min'] = vote_min if combine_mode == 'vote' else None
        debug_info['channels'] = list(channels)

    # Drop low-pop inclusions: CLAHE-amplified noise on uniform paste
    # produces contours that lack real center-vs-ring intensity differential
    # on the raw BGR channels.  Mirrors ``sherd_blobs._gate_inclusions_by_pop``.
    if inclusion_pop_min is not None and inclusion_pop_min > 0 and inclusion_contours:
        pre_pop = len(inclusion_contours)
        inclusion_contours = _gate_contours_by_pop(
            inclusion_contours, image, inclusion_pop_min)
        debug_info['inclusion_rejected_low_pop'] = pre_pop - len(inclusion_contours)

    # Recompute areas from the final (possibly cross-channel-combined) contours.
    inclusion_areas = [cv2.contourArea(c) / (dpcm ** 2) for c in inclusion_contours]
    void_areas = [cv2.contourArea(c) / (dpcm ** 2) for c in void_contours]

    if debug_mode:
        di = debug_info
        total_rej = (di.get('inclusion_rejected_solidity', 0)
                     + di.get('inclusion_rejected_compactness', 0)
                     + di.get('inclusion_rejected_boundary', 0)
                     + di.get('inclusion_rejected_nested', 0))
        print(f"[contour_detection debug]")
        if len(channels) > 1:
            print(f"  Channels                           : {', '.join(channels)}")
            mode_suffix = f" (vote_min={vote_min})" if combine_mode == 'vote' else ""
            print(f"  Combine mode                       : {combine_mode}{mode_suffix}")
        print(f"  Size-filtered inclusion candidates : {di.get('total_candidates', 0)}")
        print(f"  Accepted inclusions                : {len(inclusion_contours)}")
        print(f"  Rejected – boundary band ({di.get('edge_band_px', '?')} px)   : {di.get('inclusion_rejected_boundary', 0)}")
        print(f"  Rejected – solidity < {di.get('solidity_threshold', 0):.2f}          : {di.get('inclusion_rejected_solidity', 0)}")
        print(f"  Rejected – compactness < {di.get('compactness_threshold', 0):.2f}       : {di.get('inclusion_rejected_compactness', 0)}")
        print(f"  Rejected – nested in larger contour : {di.get('inclusion_rejected_nested', 0)}")
        print(f"  Size-filtered void candidates      : {di.get('void_candidates', '?')}")
        print(f"  Accepted voids                     : {len(void_contours)}")
        print(f"  Rejected voids – boundary band     : {di.get('void_rejected_boundary', 0)}")
        print(f"  Rejected voids – solidity < {di.get('void_solidity_threshold', 0):.2f}   : {di.get('void_rejected_solidity', 0)}")
        print(f"  Rejected voids – compactness < {di.get('void_compactness_threshold', 0):.2f}: {di.get('void_rejected_compactness', 0)}")
        print(f"  Rejected voids – aspect ratio      : {di.get('void_rejected', 0)}")
        print(f"  Rejected voids – nested in larger  : {di.get('void_rejected_nested', 0)}")
        if di.get('total_candidates', 0) > 0:
            rate = total_rej / di['total_candidates'] * 100
            print(f"  Total inclusion rejection rate     : {rate:.0f}%")

    # GEOMETRIC ANGULARITY ANALYSIS - New Feature for Temper Analysis
    from .analysis import analyze_inclusion_angularity

    # Analyze geometric properties of inclusions for archaeological interpretation
    if len(inclusion_contours) > 0:
        geometric_analysis = analyze_inclusion_angularity(inclusion_contours, scan_dpi)
    else:
        geometric_analysis = analyze_inclusion_angularity([], scan_dpi)


    return {
        'inclusions': inclusion_contours,
        'voids': void_contours,
        'inclusion_areas': inclusion_areas,
        'void_areas': void_areas,
        'total_inclusions': len(inclusion_contours),
        'total_voids': len(void_contours),
        'debug_info': debug_info,
        'geometric_analysis': geometric_analysis,
    }