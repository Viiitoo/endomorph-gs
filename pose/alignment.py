from dataclasses import dataclass

import numpy as np

from .lie import invert_se3_np


@dataclass
class Sim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray


def umeyama_sim3(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 point arrays, got {src.shape} and {dst.shape}")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    xs = src - src_mean
    xd = dst - dst_mean
    cov = (xd.T @ xs) / len(src)
    U, singular_values, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    rotation = U @ S @ Vt
    var_src = (xs * xs).sum(axis=1).mean()
    scale = float((singular_values * np.diag(S)).sum() / max(var_src, 1e-20))
    translation = dst_mean - scale * rotation @ src_mean
    return Sim3(scale=scale, rotation=rotation, translation=translation)


def apply_sim3_to_c2w(c2w_poses, sim3):
    poses = np.asarray(c2w_poses, dtype=np.float64).copy()
    out = poses.copy()
    for i in range(len(poses)):
        out[i, :3, :3] = sim3.rotation @ poses[i, :3, :3]
        out[i, :3, 3] = sim3.scale * (sim3.rotation @ poses[i, :3, 3]) + sim3.translation
        out[i, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    return out


def invert_poses(c2w_poses):
    return np.stack([invert_se3_np(p) for p in np.asarray(c2w_poses)], axis=0)

