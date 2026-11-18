from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Edge2DResult:
    edge: np.ndarray
    distance_field: np.ndarray


def _to_gray_u8_rgb(image):
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # Match S3PO_recons/tools/get_edge_local.py exactly. That script receives
    # RGB from its test harness, but converts with COLOR_BGR2GRAY.
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)


def morphological_thinning(edges):
    _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    img = (binary > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(img, ((1, 1), (1, 1)), mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            neighbor_count = sum(neighbors)
            transitions = np.zeros_like(img)
            pairs = list(zip(neighbors, neighbors[1:] + neighbors[:1]))
            for a, b in pairs:
                transitions += ((a == 0) & (b == 1)).astype(np.uint8)
            if step == 0:
                preserve_1 = p2 * p4 * p6
                preserve_2 = p4 * p6 * p8
            else:
                preserve_1 = p2 * p4 * p8
                preserve_2 = p2 * p6 * p8
            remove = (
                (img == 1)
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
                & (preserve_1 == 0)
                & (preserve_2 == 0)
            )
            if np.any(remove):
                img[remove] = 0
                changed = True
    return (img * 255).astype(np.uint8)


def fourier_low_pass_filter(image, radius=35, alpha=1.0):
    if image.ndim > 2:
        image = _to_gray_u8_rgb(image)
    image_float = np.float32(image)
    fourier_image = cv2.dft(image_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    fourier_shifted = np.fft.fftshift(fourier_image)
    rows, cols = image.shape[:2]
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (center_col, center_row), int(radius), (1, 1, 1), -1)
    filtered_image = fourier_shifted * mask
    magnitude = cv2.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])
    magnitude += 1e-9
    magnitude_equalized = np.power(magnitude, float(alpha))
    phase = cv2.phase(filtered_image[:, :, 0], filtered_image[:, :, 1])
    filtered_image[:, :, 0] = magnitude_equalized * np.cos(phase)
    filtered_image[:, :, 1] = magnitude_equalized * np.sin(phase)
    filtered_shifted = np.fft.ifftshift(filtered_image)
    inverse_image = cv2.idft(filtered_shifted, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    inverse_image = cv2.normalize(inverse_image, None, 0, 255, cv2.NORM_MINMAX)
    return inverse_image.astype(np.uint8)


def remove_short_edges_fast(edge_image, min_length):
    edge_image = np.where(np.asarray(edge_image) > 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_image, connectivity=8)
    output = np.zeros_like(edge_image)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= int(max(1, round(min_length))):
            output[labels == label] = 255
    return output


def edge_detect(input_image, index):
    blurred_image = fourier_low_pass_filter(input_image)
    ksize = max(1, 5 - int(index) * 2)
    if ksize % 2 == 0:
        ksize += 1
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=ksize)
    edges = cv2.addWeighted(np.uint8(np.absolute(sobel_x)), 0.5, np.uint8(np.absolute(sobel_y)), 0.5, 0)
    edges = cv2.convertScaleAbs(edges)
    edges = morphological_thinning(edges)
    return remove_short_edges_fast(edges, min_length=50.0 / (1 + int(index)))


def build_image_pyramid(image, num_levels=3):
    pyramid = [image]
    current = image
    for _ in range(1, int(num_levels)):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def extract_edge_rgb(image, num_levels=3, border_width=0):
    """Extract a binary contour image from RGB input using the provided get_edge.py logic."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    pyramid = build_image_pyramid(arr, num_levels=num_levels)
    original_size = pyramid[0].shape[:2]
    final_edge_mask = None
    for index, img in enumerate(pyramid):
        edges = edge_detect(img, index)
        edges_resized = cv2.resize(edges, (original_size[1], original_size[0]))
        final_edge_mask = edges_resized if final_edge_mask is None else cv2.bitwise_and(final_edge_mask, edges_resized)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    final_edge_mask = cv2.morphologyEx(final_edge_mask, cv2.MORPH_OPEN, kernel)
    final_edge_mask = morphological_thinning(final_edge_mask)
    if border_width > 0:
        b = int(border_width)
        final_edge_mask[:b, :] = 0
        final_edge_mask[-b:, :] = 0
        final_edge_mask[:, :b] = 0
        final_edge_mask[:, -b:] = 0
    return np.where(final_edge_mask > 0, 1, 0).astype(np.uint8)


def build_distance_field(edge):
    edge = np.where(np.asarray(edge) > 0, 1, 0).astype(np.uint8)
    non_edge = 1 - edge
    return cv2.distanceTransform(non_edge, cv2.DIST_L2, 0, dstType=cv2.CV_32F)


def extract_edge_and_distance_rgb(image, num_levels=3, border_width=0):
    edge = extract_edge_rgb(image, num_levels=num_levels, border_width=border_width)
    return Edge2DResult(edge=edge, distance_field=build_distance_field(edge))
