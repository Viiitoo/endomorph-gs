from dataclasses import dataclass

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is optional at import time
    cKDTree = None


@dataclass
class Contour3D:
    points: np.ndarray
    weights: np.ndarray
    scores: np.ndarray


def voxel_downsample(points, voxel_size):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0 or voxel_size <= 0:
        return points
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(keep)]


def _auto_voxel_size(points, target_bins=180):
    extent = np.ptp(points, axis=0)
    diag = float(np.linalg.norm(extent))
    if not np.isfinite(diag) or diag <= 0:
        return 0.0
    return diag / float(target_bins)


def _sor_mask(points, k=20, alpha=1.8):
    if cKDTree is None or len(points) <= k + 1:
        return np.ones(len(points), dtype=bool)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=min(k + 1, len(points)))
    mean_dist = dists[:, 1:].mean(axis=1)
    return mean_dist < mean_dist.mean() + float(alpha) * (mean_dist.std() + 1e-8)


def extract_geometric_contours(
    points_world,
    max_points=4000,
    k_neighbors=24,
    edge_percentile=85.0,
    voxel_size=None,
    sor_k=20,
    sor_alpha=1.8,
):
    """Extract dependency-light geometric 3D contours from a point cloud.

    This is only a fallback/debug extractor. It is not STAR-Edge: it scores
    local PCA curvature/linearity and filters isolated outliers.
    """
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if len(points) < max(8, k_neighbors + 1):
        return Contour3D(points=np.empty((0, 3), np.float32), weights=np.empty(0, np.float32), scores=np.empty(0, np.float32))

    if voxel_size is None:
        voxel_size = _auto_voxel_size(points)
    points = voxel_downsample(points, voxel_size=float(voxel_size or 0.0))
    if len(points) < max(8, k_neighbors + 1):
        return Contour3D(points=np.empty((0, 3), np.float32), weights=np.empty(0, np.float32), scores=np.empty(0, np.float32))

    points = points[_sor_mask(points, k=sor_k, alpha=sor_alpha)]
    if len(points) < max(8, k_neighbors + 1):
        return Contour3D(points=np.empty((0, 3), np.float32), weights=np.empty(0, np.float32), scores=np.empty(0, np.float32))

    if cKDTree is None:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(points), size=min(max_points, len(points)), replace=False)
        scores = np.ones(len(keep), dtype=np.float32)
        return Contour3D(points=points[keep].astype(np.float32), weights=scores.copy(), scores=scores)

    tree = cKDTree(points)
    _, indices = tree.query(points, k=min(k_neighbors + 1, len(points)))
    scores = np.zeros(len(points), dtype=np.float32)
    for i, nn in enumerate(indices):
        nbrs = points[nn[1:]]
        centered = nbrs - nbrs.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(1, len(nbrs) - 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        denom = float(eigvals.sum() + 1e-12)
        curvature = eigvals[0] / denom
        linearity = (eigvals[2] - eigvals[1]) / max(float(eigvals[2]), 1e-12)
        anisotropy = (eigvals[2] - eigvals[0]) / max(float(eigvals[2]), 1e-12)
        scores[i] = float(curvature + 0.25 * linearity + 0.1 * anisotropy)

    threshold = np.percentile(scores, float(edge_percentile))
    edge_mask = scores >= threshold
    edge_points = points[edge_mask]
    edge_scores = scores[edge_mask]
    if len(edge_points) == 0:
        return Contour3D(points=np.empty((0, 3), np.float32), weights=np.empty(0, np.float32), scores=np.empty(0, np.float32))
    if len(edge_points) > max_points:
        order = np.argsort(edge_scores)[-int(max_points):]
        edge_points = edge_points[order]
        edge_scores = edge_scores[order]
    weights = edge_scores - edge_scores.min()
    weights = weights / (weights.max() + 1e-8)
    weights = np.clip(weights, 0.05, 1.0).astype(np.float32)
    return Contour3D(points=edge_points.astype(np.float32), weights=weights, scores=edge_scores.astype(np.float32))


def extract_star_edge_contours(*args, **kwargs):
    raise RuntimeError(
        "extract_star_edge_contours no longer maps to the geometric fallback. "
        "Use pose.star_edge.StarEdgeExtractor for real STAR-Edge, or "
        "extract_geometric_contours for the fallback explicitly."
    )
