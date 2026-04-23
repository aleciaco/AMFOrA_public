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
    'super_zorro_cv', 'sherd_blobs', 'detect_multiple_sherds',
    'contour_detection',
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
    def optimal_canny_thresholds(image, sigma=0.33):
        """Calculate optimal Canny edge detection thresholds using automatic methods"""
        # Method 1: Otsu-based automatic threshold
        otsu_thresh, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Method 2: Median-based automatic threshold  
        median_val = np.median(image)
        
        # Method 3: Statistical approach - use image gradients
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_mean = np.mean(gradient_magnitude)
        
        # Combine methods for robust threshold selection
        base_thresh = min(otsu_thresh * 0.5, median_val, gradient_mean)
        
        # Calculate Canny thresholds with optimal ratio
        lower_thresh = max(10, int((1.0 - sigma) * base_thresh))
        upper_thresh = max(30, int((1.0 + sigma) * base_thresh))
        
        # Ensure reasonable bounds
        lower_thresh = min(lower_thresh, 100)
        upper_thresh = min(upper_thresh, 255)
        upper_thresh = max(upper_thresh, lower_thresh * 2)
        
        return lower_thresh, upper_thresh

    def detect_background_statistics(image, border_size=0.05):
        """Detect background color/intensity by sampling image borders."""
        h, w = image.shape
        border_pixels = int(min(h, w) * border_size)

        # Sample all four borders
        top_border = image[:border_pixels, :]
        bottom_border = image[-border_pixels:, :]
        left_border = image[:, :border_pixels]
        right_border = image[:, -border_pixels:]

        # Combine border pixels
        all_border_pixels = np.concatenate([
            top_border.flatten(),
            bottom_border.flatten(),
            left_border.flatten(),
            right_border.flatten()
        ])

        # Calculate robust statistics
        bg_mean = np.median(all_border_pixels)  # More robust than mean
        bg_std = np.std(all_border_pixels)

        return bg_mean, bg_std

    def adaptive_morphology_kernel(scan_dpi, target_size_mm=0.5):
        """Create morphological kernel sized appropriately for scan resolution"""
        # Convert mm to pixels
        dpcm = scan_dpi * 0.3937
        target_size_cm = target_size_mm / 10.0
        kernel_size_pixels = int(target_size_cm * dpcm)
        
        # Ensure odd kernel size and reasonable bounds
        kernel_size_pixels = max(3, kernel_size_pixels)
        if kernel_size_pixels % 2 == 0:
            kernel_size_pixels += 1
        
        # Limit maximum kernel size to prevent over-smoothing
        kernel_size_pixels = min(kernel_size_pixels, 21)
        
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                        (kernel_size_pixels, kernel_size_pixels))

    # Input validation
    if scan_dpi < 150 or scan_dpi > 2400:
        print(f"Warning: scan_dpi {scan_dpi} outside recommended range (150-2400)")

    #read image
    image = sherd_scan
    orig_h, orig_w = image.shape[:2]

    # Crop off outer 0.5cm border to remove box outline used to block light during scanning
    dpcm = scan_dpi * 0.3937
    border_crop = int(0.5 * dpcm)  # 0.5cm in pixels

    # Ensure we don't crop more than available
    border_crop = min(border_crop, min(orig_h, orig_w) // 4)

    # Crop the image
    image_cropped = image[border_crop:orig_h - border_crop, border_crop:orig_w - border_crop]

    #cvt to grayscale
    im_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

    # Enhanced edge detection with optimal thresholds
    lower_thresh, upper_thresh = optimal_canny_thresholds(im_gray)
    edges = cv2.Canny(im_gray, lower_thresh, upper_thresh)
    
    # DPI-aware morphological operations
    kernel = adaptive_morphology_kernel(scan_dpi, target_size_mm=0.5)
    
    # Improved morphological sequence
    # Close gaps in edges
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # Clean up noise
    edges_clean = cv2.morphologyEx(edges_closed, cv2.MORPH_OPEN, kernel)

    #find the contours of the almost fully binarized mask
    contours_canny, _ = cv2.findContours(edges_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Enhanced threshold-based approach with background detection
    bg_mean, bg_std = detect_background_statistics(im_gray)
    
    #run a blur on the grayscale image
    blur = cv2.GaussianBlur(im_gray,(5,5),0)
    
    # Original OTSU thresholding
    ret,thresh_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    
    # Adaptive threshold based on background statistics
    adaptive_thresh_val = max(0, min(255, int(bg_mean + 2 * bg_std)))
    _, thresh_adaptive = cv2.threshold(blur, adaptive_thresh_val, 255, cv2.THRESH_BINARY)
    
    #find the contours from both thresholding methods
    contours_otsu, _ = cv2.findContours(thresh_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_adaptive, _ = cv2.findContours(thresh_adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Select best contours based on multiple criteria
    all_contour_methods = [
        (contours_canny, "canny"),
        (contours_otsu, "otsu"), 
        (contours_adaptive, "adaptive")
    ]
    
    best_contour = None
    best_area = 0
    
    for contours, method in all_contour_methods:
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Prefer contours that are reasonable size (10-90% of image)
            image_area = im_gray.shape[0] * im_gray.shape[1]
            area_ratio = area / image_area

            if 0.1 <= area_ratio <= 0.9 and area > best_area:
                best_contour = largest_contour
                best_area = area
    
    # Fallback if no good contour found - use original logic
    if best_contour is None:
        if len(contours_otsu) > 0 and len(contours_canny) > 0:
            importantcontour_canny = max(contours_canny, key = cv2.contourArea)
            importantcontour_thresh = max(contours_otsu, key = cv2.contourArea)
            
            # Use original selection logic as fallback
            if cv2.contourArea(importantcontour_canny) > 1.1*cv2.contourArea(importantcontour_thresh):
                best_contour = importantcontour_thresh
            elif cv2.contourArea(importantcontour_canny) < cv2.contourArea(importantcontour_thresh):
                best_contour = importantcontour_thresh
            else:
                best_contour = importantcontour_canny
        elif len(contours_otsu) > 0:
            best_contour = max(contours_otsu, key = cv2.contourArea)
        elif len(contours_canny) > 0:
            best_contour = max(contours_canny, key = cv2.contourArea)

    #create a mask that is all zeros the same shape as the cropped image;
    #take that big ole contour and try to fill it in with ones
    blackbox = np.zeros(im_gray.shape, np.uint8)
    mask_cropped = cv2.drawContours(blackbox.copy(), [best_contour], -1, 255, cv2.FILLED, 1)

    # Pad the mask back to original image size (border region stays 0/black)
    mask = np.zeros((orig_h, orig_w), np.uint8)
    mask[border_crop:orig_h - border_crop, border_crop:orig_w - border_crop] = mask_cropped

    # Compute crop bounds from the sherd contour bounding rect.
    # best_contour is in image_cropped coordinates; translate back to original.
    if best_contour is not None and auto_crop:
        x_br, y_br, w_br, h_br = cv2.boundingRect(best_contour)
        # Calculate square side length
        side = max(w_br, h_br)
        # Center of the bounding rectangle
        x_center = x_br + w_br // 2
        y_center = y_br + h_br // 2
        # Ideal (possibly out-of-bounds) square crop coordinates
        x1_raw = x_center + border_crop - side // 2 - crop_buffer
        x2_raw = x_center + border_crop + side // 2 + crop_buffer
        y1_raw = y_center + border_crop - side // 2 - crop_buffer
        y2_raw = y_center + border_crop + side // 2 + crop_buffer
        # Clip to image bounds
        x1 = max(0, x1_raw)
        x2 = min(orig_w, x2_raw)
        y1 = max(0, y1_raw)
        y2 = min(orig_h, y2_raw)
        # Padding needed to restore the intended square shape
        pad_left   = x1 - x1_raw
        pad_right  = x2_raw - x2
        pad_top    = y1 - y1_raw
        pad_bottom = y2_raw - y2
    else:
        # auto_crop=False or no contour found: return full-size mask
        y1, y2, x1, x2 = 0, orig_h, 0, orig_w
        pad_top = pad_bottom = pad_left = pad_right = 0

    crop = (y1, y2, x1, x2, pad_top, pad_bottom, pad_left, pad_right)

    # Slice mask to crop region, then pad if edge-clipped
    mask_slice = mask[y1:y2, x1:x2]
    if pad_top or pad_bottom or pad_left or pad_right:
        mask_slice = np.pad(mask_slice,
                            ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='constant', constant_values=0)

    #Need to 'stack' the image to create a 3D array, because RGB images are 3D arrays
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


def sherd_blobs(image, scan_dpi=1200, size_params=None, blob_params=None):
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
    form complementary discriminators.  Some dark features may appear in both
    lists — this matches the behavior of ``contour_detection``, which also
    allows overlap between inclusion and void classifications.

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

    image_gray = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)[:, :, 0]  # L* channel

    # DPI-scaled Gaussian blur to reduce noise before thresholding.
    # Calibrated: 5×5 @ 600 DPI, 11×11 @ 1200 DPI, 21×21 @ 2400 DPI.
    # Scales linearly with DPI and forced odd for OpenCV.
    blur_k = int(round(scan_dpi / 600.0 * 5))
    blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1  # must be odd
    blur_k = max(3, blur_k)  # minimum 3×3

    gray_blur = cv2.GaussianBlur(image_gray, (blur_k, blur_k), 0)

    def _apply_overrides(params):
        """Apply user blob_params overrides to a detector param object."""
        if blob_params:
            for key, val in blob_params.items():
                if hasattr(params, key):
                    setattr(params, key, val)
        return params

    # 1. Light inclusions (bright features on darker background)
    light_params = _apply_overrides(
        setup_robust_blob_params(gray_blur, scan_dpi, "light", size_params))
    blobs_light_inc = list(
        cv2.SimpleBlobDetector_create(light_params).detect(gray_blur))

    # 2. Dark inclusions (dark minerals: ferruginous grains, magnetite, biotite, dark grog)
    dark_inc_params = _apply_overrides(
        setup_robust_blob_params(gray_blur, scan_dpi, "dark_inclusion", size_params))
    blobs_dark_inc = list(
        cv2.SimpleBlobDetector_create(dark_inc_params).detect(gray_blur))

    # 3. Dark voids (pores, organic burnout channels)
    dark_void_params = _apply_overrides(
        setup_robust_blob_params(gray_blur, scan_dpi, "dark", size_params))
    blobs_dark_void = list(
        cv2.SimpleBlobDetector_create(dark_void_params).detect(gray_blur))

    # Combine light + dark inclusions
    blobs_inclusions = blobs_light_inc + blobs_dark_inc

    # Dark voids are independent — no deduplication needed.  The strict
    # shape filters on dark_inclusion (circularity, convexity, inertia)
    # naturally separate compact mineral grains from irregular voids,
    # so overlap between the two lists is minimal.
    blobs_voids = blobs_dark_void

    return blobs_inclusions, blobs_voids


def detect_multiple_sherds(image, mask):
    """
    Detect if there are multiple sherds in a single scan and separate them.
    
    Parameters
    ----------
    image : numpy.ndarray
        Original image
    mask : numpy.ndarray
        Binary mask of ceramic areas
        
    Returns
    -------
    list of dict
        List of sherd regions, each containing:
        - 'submask': Binary mask for this sherd
        - 'bbox': Bounding box (x, y, width, height)
        - 'centroid': Center point of sherd
        - 'area': Area of sherd region
        If only one sherd detected, returns single-item list
    """
    
    # Find connected components in the mask  
    # Ensure mask is single-channel for OpenCV
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Ensure binary mask
    mask_binary = (mask > 0).astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    
    # Filter out background (label 0) and very small components
    min_sherd_area = np.sum(mask) * 0.05  # Minimum 5% of total mask area
    
    sherd_regions = []
    
    for label in range(1, num_labels):  # Skip background (label 0)
        component_area = stats[label, cv2.CC_STAT_AREA]
        
        if component_area > min_sherd_area:
            # Create submask for this component
            submask = (labels == label).astype(np.uint8) * 255
            
            # Get bounding box
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP] 
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            
            sherd_regions.append({
                'submask': submask,
                'bbox': (x, y, width, height),
                'centroid': centroids[label],
                'area': component_area
            })
    
    # Sort by area (largest first) for consistent ordering
    sherd_regions.sort(key=lambda x: x['area'], reverse=True)
    
    # If no valid sherds found, return the original mask as single sherd
    if not sherd_regions:
        sherd_regions.append({
            'submask': mask,
            'bbox': (0, 0, mask.shape[1], mask.shape[0]),
            'centroid': (mask.shape[1]//2, mask.shape[0]//2),
            'area': np.sum(mask)
        })
    
    return sherd_regions


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

def contour_detection(image, scan_dpi=1200, size_params=None, shape_params=None, debug_mode=False):
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
            but still filter out wire-thin artifacts.

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
    
    # Convert to L* (lightness) channel from CIELAB colour space.
    # L* is perceptually uniform and consistent with the Lab colour analyses
    # used throughout the pipeline, unlike cv2.cvtColor(BGR2GRAY) which is a
    # weighted RGB sum biased toward green.
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        gray = lab[:, :, 0]  # L* channel (0-255 in OpenCV's 8-bit Lab)
    else:
        gray = image.copy()

    # Calculate DPI-aware parameters
    dpcm = scan_dpi * 0.3937  # dots per cm

    # DPI-scaled Gaussian blur to reduce noise before thresholding.
    # Calibrated: 5×5 @ 600 DPI, 11×11 @ 1200 DPI, 21×21 @ 2400 DPI.
    # Scales linearly with DPI and forced odd for OpenCV.
    blur_k = int(round(scan_dpi / 600.0 * 5))
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

    # INCLUSION DETECTION - Handle both dark and light inclusions
    # Get image statistics for adaptive thresholding
    mean_brightness = np.mean(gray[gray > 0])  # Exclude black pixels
    std_brightness = np.std(gray[gray > 0])

    # Apply DPI-scaled Gaussian blur before thresholding
    gray_blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # Create two thresholded images: one for dark inclusions, one for light
    # Dark inclusions (lower threshold for dark features)
    dark_thresh = max(30, int(mean_brightness - std_brightness))
    ret_dark, th_dark = cv2.threshold(gray_blur, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    
    # Light inclusions (higher threshold for bright features)  
    light_thresh = min(220, int(mean_brightness + std_brightness))
    ret_light, th_light = cv2.threshold(gray_blur, light_thresh, 255, cv2.THRESH_BINARY)
    
    # Combine both thresholded images
    th1 = cv2.bitwise_or(th_dark, th_light)
    
    # Optional morphological cleaning (as shown in notebook)
    # cleaned_inc_cont = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, 
    #                                    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)))
    
    # Find inclusion contours (RETR_TREE, CHAIN_APPROX_SIMPLE as in cv2_test.py line 1598)
    contours_inc, _ = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Size-filter inclusion candidates.  max_area_threshold (1.5 cm²) excludes the
    # sherd boundary and any hierarchy duplicates from RETR_TREE.
    sel = [c for c in contours_inc
           if min_area_threshold < cv2.contourArea(c) < max_area_threshold]
    
    # Shape-quality thresholds — calibrated defaults, overridable via shape_params.
    # max_aspect_ratio is the primary elongation gate applied FIRST in the filter chain;
    # solidity and compactness are secondary convexity/regularity checks.
    # CROSS-METHOD CONSISTENCY: max_aspect_ratio = 1 / minInertiaRatio (blob detector)
    #   → max_aspect_ratio 5.0  ↔  minInertiaRatio 0.2  (both defaults identical)
    inclusion_max_aspect_ratio = 4.0    # primary filter: rejects contours with long/short > 4:1
                                       # matches blob default minInertiaRatio=0.2 exactly
    void_max_aspect_ratio     = 5.0
    inclusion_solidity_min    = 0.45    # secondary: area / convex-hull area
    inclusion_compactness_min = 0.125   # secondary: 4π·area / perimeter²
    void_solidity_min         = 0.1    # secondary (voids only, more permissive)
    void_compactness_min      = 0.06   # secondary: 4π·area / perimeter² (voids)
    if shape_params:
        # Primary filter first
        inclusion_max_aspect_ratio          = shape_params.get('inclusion_max_aspect_ratio',          inclusion_max_aspect_ratio)
        void_max_aspect_ratio               = shape_params.get('void_max_aspect_ratio',               void_max_aspect_ratio)
        # Secondary filters
        inclusion_solidity_min    = shape_params.get('inclusion_solidity_min',    inclusion_solidity_min)
        inclusion_compactness_min = shape_params.get('inclusion_compactness_min', inclusion_compactness_min)
        void_solidity_min         = shape_params.get('void_solidity_min',         void_solidity_min)
        void_compactness_min      = shape_params.get('void_compactness_min',      void_compactness_min)

    # Apply shape-quality filtering to inclusion candidates
    inclusion_contours = []
    inclusion_areas = []
    debug_info = {
        'total_candidates': len(sel),
        'inclusion_accepted': 0,
        'inclusion_rejected_solidity': 0,
        'inclusion_rejected_compactness': 0,
        'void_accepted': 0,
        'void_rejected': 0,
        'void_rejected_solidity': 0,
        'void_rejected_compactness': 0,
        'solidity_threshold': inclusion_solidity_min,
        'compactness_threshold': inclusion_compactness_min,
        'void_solidity_threshold': void_solidity_min,
        'void_compactness_threshold': void_compactness_min,
        'inclusion_max_aspect_ratio': inclusion_max_aspect_ratio,
        'void_max_aspect_ratio': void_max_aspect_ratio,
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

    # VOID DETECTION - Use OTSU thresholding on same blurred L* channel
    _, thresh_voids = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # maybe change back to 0, 255? 125 seems to give better void detection but may need tuning




    # Find void contours
    contours_voids, _ = cv2.findContours(th_dark, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Size-filter void candidates (same logic — max threshold excludes sherd boundary)
    sel_voids = [c for c in contours_voids
                 if void_min_area_threshold < cv2.contourArea(c) < void_max_area_threshold]

    void_contours = []
    void_areas = []
    debug_info['void_candidates'] = len(sel_voids)

    for contour in sel_voids:
        area_pixels = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area > 0:
            solidity = float(area_pixels) / hull_area

            perimeter = cv2.arcLength(contour, True)
            compactness = (4 * np.pi * area_pixels) / (perimeter ** 2) if perimeter > 0 else 0

            _, (rw, rh), _ = cv2.minAreaRect(contour)
            aspect_ratio = (max(rw, rh) / max(min(rw, rh), 1e-6))

            if (aspect_ratio <= void_max_aspect_ratio
                    and solidity > void_solidity_min
                    and compactness > void_compactness_min):
                void_contours.append(contour)
                area_cm2 = area_pixels / (dpcm ** 2)
                void_areas.append(area_cm2)
                debug_info['void_accepted'] += 1
            elif solidity <= void_solidity_min:
                debug_info['void_rejected_solidity'] += 1
            elif compactness <= void_compactness_min:
                debug_info['void_rejected_compactness'] += 1
            else:
                debug_info['void_rejected'] += 1

    if debug_mode:
        di = debug_info
        total_rej = di['inclusion_rejected_solidity'] + di['inclusion_rejected_compactness']
        print(f"[contour_detection debug]")
        print(f"  Size-filtered inclusion candidates : {di['total_candidates']}")
        print(f"  Accepted inclusions                : {di['inclusion_accepted']}")
        print(f"  Rejected – solidity < {di['solidity_threshold']:.2f}          : {di['inclusion_rejected_solidity']}")
        print(f"  Rejected – compactness < {di['compactness_threshold']:.2f}       : {di['inclusion_rejected_compactness']}")
        print(f"  Size-filtered void candidates      : {di.get('void_candidates', '?')}")
        print(f"  Accepted voids                     : {di['void_accepted']}")
        print(f"  Rejected voids – solidity < {di['void_solidity_threshold']:.2f}   : {di['void_rejected_solidity']}")
        print(f"  Rejected voids – compactness < {di['void_compactness_threshold']:.2f}: {di['void_rejected_compactness']}")
        print(f"  Rejected voids – aspect ratio      : {di['void_rejected']}")
        if di['total_candidates'] > 0:
            rate = total_rej / di['total_candidates'] * 100
            print(f"  Shape-filter rejection rate (inc)  : {rate:.0f}%")

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