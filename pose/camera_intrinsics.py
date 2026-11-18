from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageTransform:
    """Pixel-coordinate transform from a source image to a processed frame."""

    src_size: tuple[int, int]
    dst_size: tuple[int, int]
    scale_x: float
    scale_y: float
    crop_x: float = 0.0
    crop_y: float = 0.0


def as_intrinsics_matrix(intrinsics) -> np.ndarray:
    """Return a 3x3 float32 K matrix from either K or (fx, fy, cx, cy)."""
    arr = np.asarray(intrinsics, dtype=np.float32)
    if arr.shape == (3, 3):
        return arr.copy()
    flat = arr.reshape(-1)
    if flat.shape[0] < 4:
        raise ValueError(f"Expected K(3x3) or [fx, fy, cx, cy], got shape {arr.shape}")
    fx, fy, cx, cy = flat[:4]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def scale_intrinsics(K, src_size, dst_size) -> np.ndarray:
    """Scale camera intrinsics for image resize.

    Sizes are (width, height). This assumes no crop or padding.
    """
    K = as_intrinsics_matrix(K)
    src_w, src_h = map(float, src_size)
    dst_w, dst_h = map(float, dst_size)
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source size: {src_size}")
    out = K.copy()
    sx = dst_w / src_w
    sy = dst_h / src_h
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] *= sx
    out[1, 2] *= sy
    return out.astype(np.float32)


def transform_intrinsics(K, transform: ImageTransform) -> np.ndarray:
    """Apply resize and crop offsets to K.

    The transform maps original pixels to processed-frame pixels:
    u' = scale_x * u - crop_x, v' = scale_y * v - crop_y.
    """
    K = as_intrinsics_matrix(K)
    out = K.copy()
    out[0, 0] *= float(transform.scale_x)
    out[1, 1] *= float(transform.scale_y)
    out[0, 2] = out[0, 2] * float(transform.scale_x) - float(transform.crop_x)
    out[1, 2] = out[1, 2] * float(transform.scale_y) - float(transform.crop_y)
    return out.astype(np.float32)


def resize_transform(src_size, dst_size) -> ImageTransform:
    """Build an ImageTransform for direct resize from src_size to dst_size."""
    src_w, src_h = map(float, src_size)
    dst_w, dst_h = map(float, dst_size)
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source size: {src_size}")
    return ImageTransform(
        src_size=(int(src_w), int(src_h)),
        dst_size=(int(dst_w), int(dst_h)),
        scale_x=dst_w / src_w,
        scale_y=dst_h / src_h,
    )


def pixel_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return u,v pixel grids with shape (H,W)."""
    v, u = np.mgrid[0:int(height), 0:int(width)]
    return u.astype(np.float32), v.astype(np.float32)


def constrain_pointmap_to_intrinsics(pointmap, K, keep_depth: str = "z") -> np.ndarray:
    """Project a pointmap back onto the camera rays defined by K.

    MASt3R pointmaps are useful but not guaranteed to be exactly consistent with
    the real calibrated camera. This keeps a scalar depth from the pointmap and
    rebuilds X/Y from the calibrated pixel ray:

        X = (u - cx) / fx * depth
        Y = (v - cy) / fy * depth
        Z = depth

    keep_depth="z" uses pointmap[...,2]. keep_depth="norm" uses Euclidean norm
    along the calibrated unit-z ray; "z" is the standard pinhole-depth choice.
    """
    pts = np.asarray(pointmap, dtype=np.float32)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError(f"Expected pointmap with shape (H,W,3), got {pts.shape}")
    K = as_intrinsics_matrix(K)
    height, width = pts.shape[:2]
    u, v = pixel_grid(height, width)
    ray_x = (u - K[0, 2]) / K[0, 0]
    ray_y = (v - K[1, 2]) / K[1, 1]
    if keep_depth == "z":
        depth = pts[..., 2]
    elif keep_depth == "norm":
        ray_norm = np.sqrt(ray_x * ray_x + ray_y * ray_y + 1.0)
        depth = np.linalg.norm(pts, axis=-1) / ray_norm
    else:
        raise ValueError(f"Unknown keep_depth mode: {keep_depth}")
    out = np.stack([ray_x * depth, ray_y * depth, depth], axis=-1)
    invalid = ~np.isfinite(pts).all(axis=-1) | ~np.isfinite(depth)
    out[invalid] = np.nan
    return out.astype(np.float32)


def fit_effective_intrinsics(pointmap) -> tuple[np.ndarray, dict]:
    """Fit a diagnostic pinhole K from a pointmap to its own pixel grid."""
    pts = np.asarray(pointmap, dtype=np.float32)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError(f"Expected pointmap with shape (H,W,3), got {pts.shape}")
    height, width = pts.shape[:2]
    u, v = pixel_grid(height, width)
    flat = pts.reshape(-1, 3)
    z = flat[:, 2]
    valid = np.isfinite(flat).all(axis=1) & (np.abs(z) > 1e-6)
    if int(valid.sum()) < 8:
        raise ValueError("Not enough valid pointmap pixels to fit intrinsics.")
    xn = flat[:, 0] / z
    yn = flat[:, 1] / z
    ax = np.stack([xn[valid], np.ones(int(valid.sum()), dtype=np.float32)], axis=1)
    ay = np.stack([yn[valid], np.ones(int(valid.sum()), dtype=np.float32)], axis=1)
    fx, cx = np.linalg.lstsq(ax, u.reshape(-1)[valid], rcond=None)[0]
    fy, cy = np.linalg.lstsq(ay, v.reshape(-1)[valid], rcond=None)[0]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    pred_u = fx * xn + cx
    pred_v = fy * yn + cy
    err = np.sqrt((pred_u - u.reshape(-1)) ** 2 + (pred_v - v.reshape(-1)) ** 2)[valid]
    stats = {
        "valid_pixels": int(valid.sum()),
        "median_reprojection_error": float(np.median(err)),
        "mean_reprojection_error": float(np.mean(err)),
        "p90_reprojection_error": float(np.percentile(err, 90)),
        "p99_reprojection_error": float(np.percentile(err, 99)),
    }
    return K, stats
