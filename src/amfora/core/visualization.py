"""
Visualization functions for AMACFA+ ceramic analysis.

This module contains functions for visualizing inclusions, creating color
palettes, and interactive exploration of ceramic fabric features.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

__all__ = ['inclusion_viewer']


def _contours_to_sq_list(contours):
    """
    Convert a list of cv2 contours into the [(v1, v2), size] format used by
    sacredsquare, sorted largest-first by area.

    Each contour's bounding rect supplies v1/v2 and the equivalent circular
    diameter is stored as the size value (matching sacredsquare's convention).
    """
    items = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        v1 = (x, y)
        v2 = (x + w, y + h)
        area = cv2.contourArea(c)
        diameter = 2 * np.sqrt(area / np.pi) if area > 0 else 0
        items.append([(v1, v2), diameter])
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def inclusion_viewer(inclusions, img_color, method='blob'):
    """
    Interactive viewer for individual inclusions with color analysis.

    Parameters
    ----------
    inclusions : list
        When method='blob': output list from sacredsquare
            [(left_vertex, right_vertex), size]
        When method='contour': list of contour arrays from contour_detection
            (contour_results['inclusions'])
    img_color : numpy.ndarray
        Color image of the sherd (masked)
    method : str, optional
        Detection method that produced the inclusion list.
        'blob' (default) -- sacredsquare bounding boxes
        'contour' -- raw contour arrays from contour_detection

    Returns
    -------
    None
        Displays interactive matplotlib visualization
    """
    if method == 'contour':
        # Sort contours largest-first by area
        inclusions = sorted(inclusions, key=cv2.contourArea, reverse=True)
        if len(inclusions) == 0:
            print("No inclusions found to display")
            return
        num = int(input(prompt=f"which inclusion do you want to see up close and personal 1-{len(inclusions)}?")) - 1
        if num < 0 or num >= len(inclusions):
            print("Invalid inclusion number")
            return
    else:
        sq_lst = inclusions
        if len(sq_lst) == 0:
            print("No inclusions found to display")
            return
        num = int(input(prompt=f"which inclusion do you want to see up close and personal 1-{len(sq_lst)}?")) - 1
        if num < 0 or num >= len(sq_lst):
            print("Invalid inclusion number")
            return

    h_img, w_img = img_color.shape[:2]

    try:
        if method == 'contour':
            contour = inclusions[num]
            x, y, w, h = cv2.boundingRect(contour)

            # Extract pixels inside contour only
            roi = img_color[y:y+h, x:x+w]
            cmask = np.zeros((h, w), dtype=np.uint8)
            shifted = contour - np.array([x, y])
            cv2.drawContours(cmask, [shifted], -1, 255, -1)
            pixels = roi[cmask > 0]

            if len(pixels) == 0:
                print("No pixels found inside contour")
                return

            Z = np.float32(pixels)
            criteria = (cv2.TERM_CRITERIA_EPS, 10, 0.1)
            K = min(3, len(pixels))
            _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)

            # Sort centers by frequency
            unique, counts = np.unique(label.flatten(), return_counts=True)
            order = np.argsort(-counts)
            Z_center = list(center[order])

            # Highlight: draw contour instead of rectangle
            highlight = img_color.copy()
            cv2.drawContours(highlight, [contour], -1, (0, 255, 255), 3)

            # Close-up: crop around bounding box with padding
            pad = 250
            highlight_close = img_color.copy()
            cv2.drawContours(highlight_close, [contour], -1, (0, 255, 255), 3)
            highlight_close = highlight_close[max(0, y-pad):min(h_img, y+h+pad),
                                              max(0, x-pad):min(w_img, x+w+pad)]

            # Cropped inclusion view: show ROI with contour overlay
            inc_display = roi.copy()
            cv2.drawContours(inc_display, [shifted], -1, (0, 255, 255), 2)

            n_total = len(inclusions)
        else:
            inc_img = img_color[sq_lst[num][0][0][1]:sq_lst[num][0][1][1],
                                sq_lst[num][0][0][0]:sq_lst[num][0][1][0]]

            Z = inc_img.reshape((-1, 3))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS, 10, .1)
            K = 3
            ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            center = np.uint8(center)

            # Sort the centers by how many times they occur
            X = center
            new_label = np.array([x[0] for x in label]).T
            Y = list(np.unique(new_label, return_counts=True)[1])
            Z_center = [x for _, x in sorted(zip(Y, X))]

            # Pull original image and highlight interesting inclusion
            highlight = cv2.rectangle(img_color.copy(), sq_lst[num][0][0], sq_lst[num][0][1], (0, 255, 255), 7)

            # Close-up
            x1, y1 = sq_lst[num][0][0]
            x2, y2 = sq_lst[num][0][1]
            pad = 250
            highlight_close = cv2.rectangle(img_color.copy(), sq_lst[num][0][0], sq_lst[num][0][1], (0, 255, 255), 7)
            highlight_close = highlight_close[max(0, y1 - pad):min(h_img, y2 + pad),
                                              max(0, x1 - pad):min(w_img, x2 + pad)]

            inc_display = inc_img
            n_total = len(sq_lst)

        # Create swatches and convert to Lab for display
        swatch = np.zeros((250, 250, 3), np.uint8)
        swatch1 = swatch.copy()
        swatch2 = swatch.copy()

        def bgr_to_lab(bgr):
            pixel = np.zeros((1, 1, 3), np.uint8)
            pixel[0, 0] = bgr
            lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
            return list(map(int, lab[0, 0]))

        lab_colors = []
        if len(Z_center) >= 3:
            swatch[:, :, 2], swatch[:, :, 1], swatch[:, :, 0] = Z_center[2]
            swatch1[:, :, 2], swatch1[:, :, 1], swatch1[:, :, 0] = Z_center[1]
            swatch2[:, :, 2], swatch2[:, :, 1], swatch2[:, :, 0] = Z_center[0]
            lab_colors = [bgr_to_lab(Z_center[2]), bgr_to_lab(Z_center[1]), bgr_to_lab(Z_center[0])]
        elif len(Z_center) >= 2:
            swatch[:, :, 2], swatch[:, :, 1], swatch[:, :, 0] = Z_center[1]
            swatch1[:, :, 2], swatch1[:, :, 1], swatch1[:, :, 0] = Z_center[0]
            swatch2[:, :, 2], swatch2[:, :, 1], swatch2[:, :, 0] = Z_center[0]
            lab_colors = [bgr_to_lab(Z_center[1]), bgr_to_lab(Z_center[0]), bgr_to_lab(Z_center[0])]
        else:
            swatch[:, :, 2], swatch[:, :, 1], swatch[:, :, 0] = Z_center[0]
            swatch1[:, :, 2], swatch1[:, :, 1], swatch1[:, :, 0] = Z_center[0]
            swatch2[:, :, 2], swatch2[:, :, 1], swatch2[:, :, 0] = Z_center[0]
            lab_colors = [bgr_to_lab(Z_center[0]), bgr_to_lab(Z_center[0]), bgr_to_lab(Z_center[0])]

        # Plot it all
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 15))
        ax[0, 0].imshow(highlight[:, :, ::-1])
        ax[0, 0].set_title('Sherd Analysis', fontsize=15)
        ax[0, 0].axis('off')
        ax[0, 0].set_aspect('equal', 'box')
        ax[0, 1].imshow(inc_display[:, :, ::-1])
        ax[0, 2].imshow(highlight_close[:, :, ::-1])
        ax[0, 2].set_title(f'inclusion #{num+1}/{n_total}', fontsize=15)
        ax[1, 0].imshow(swatch)
        ax[1, 0].axis('on')
        ax[1, 0].get_xaxis().set_visible(False)
        ax[1, 0].get_yaxis().set_visible(False)
        ax[1, 0].set_title(f'dominant color L*a*b*={lab_colors[0]}', fontsize=15)
        ax[1, 1].imshow(swatch1)
        ax[1, 1].get_xaxis().set_visible(False)
        ax[1, 1].get_yaxis().set_visible(False)
        ax[1, 1].set_title(f'secondary color L*a*b*={lab_colors[1]}', fontsize=15)
        ax[1, 2].imshow(swatch2)
        ax[1, 2].get_xaxis().set_visible(False)
        ax[1, 2].get_yaxis().set_visible(False)
        ax[1, 2].set_title(f'tertiary color L*a*b*={lab_colors[2]}', fontsize=15)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error displaying inclusion: {e}")
