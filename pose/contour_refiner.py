from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .lie import se3_exp


@dataclass
class RefineConfig:
    iterations: int = 20
    lr_rot: float = 5e-4
    lr_trans: float = 1e-3
    huber_delta: float = 3.0
    pose_prior_weight: float = 1e-4
    min_valid_points: int = 64
    margin_pixels: int = 20
    max_delta_rot_deg: float = 2.0
    max_delta_trans: float = 5.0
    device: str = "cuda"


@dataclass
class PoseRefineResult:
    refined_w2c: np.ndarray
    refined_c2w: np.ndarray
    success: bool
    reason: str
    loss_before: float | None
    loss_after: float | None
    valid_points: int
    delta_rot_deg: float
    delta_trans: float
    config: dict


def _invert_se3_torch(mat):
    out = torch.eye(4, dtype=mat.dtype, device=mat.device)
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out


def _invert_se3_np(mat):
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    out = np.eye(4, dtype=mat.dtype)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out


def _project(points_world, w2c, K, width, height, margin):
    cam = (w2c[:3, :3] @ points_world.T).T + w2c[:3, 3]
    z = cam[:, 2]
    valid_z = z > 1e-6
    u = K[0, 0] * cam[:, 0] / z.clamp_min(1e-6) + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z.clamp_min(1e-6) + K[1, 2]
    inside = (
        valid_z
        & (u >= margin)
        & (u <= float(width - 1 - margin))
        & (v >= margin)
        & (v <= float(height - 1 - margin))
    )
    uv = torch.stack([u[inside], v[inside]], dim=-1)
    return uv, inside


def _sample_distance(distance_field, uv, width, height):
    if uv.numel() == 0:
        return None
    x_norm = uv[:, 0] / max(1.0, float(width - 1)) * 2.0 - 1.0
    y_norm = uv[:, 1] / max(1.0, float(height - 1)) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(distance_field, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.view(-1)


def _rotation_angle_deg(rot):
    tr = torch.diagonal(rot).sum()
    val = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(val)).detach().cpu())


def refine_pose(coarse_w2c, K, contour_world, distance_field, image_size, weights=None, config=None):
    cfg = config if isinstance(config, RefineConfig) else RefineConfig(**(config or {}))
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    height, width = int(image_size[0]), int(image_size[1])

    coarse = torch.as_tensor(coarse_w2c, dtype=torch.float32, device=device)
    K_t = torch.as_tensor(K, dtype=torch.float32, device=device)
    points = torch.as_tensor(np.asarray(contour_world, dtype=np.float32), dtype=torch.float32, device=device)
    dist = torch.as_tensor(np.asarray(distance_field, dtype=np.float32), dtype=torch.float32, device=device)[None, None]
    if weights is not None:
        weights_t = torch.as_tensor(np.asarray(weights, dtype=np.float32), dtype=torch.float32, device=device)
    else:
        weights_t = torch.ones(points.shape[0], dtype=torch.float32, device=device)

    trans_delta = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))
    rot_delta = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(
        [
            {"params": [trans_delta], "lr": cfg.lr_trans},
            {"params": [rot_delta], "lr": cfg.lr_rot},
        ]
    )

    def current_delta():
        return torch.cat([trans_delta, rot_delta], dim=0)

    def eval_loss(delta_value):
        pose = se3_exp(delta_value) @ coarse
        uv, valid = _project(points, pose, K_t, width, height, cfg.margin_pixels)
        if int(valid.sum().item()) < cfg.min_valid_points:
            return None, pose, valid, None
        sampled = _sample_distance(dist, uv, width, height)
        if sampled is None or sampled.numel() < cfg.min_valid_points:
            return None, pose, valid, sampled
        w = weights_t[valid][: sampled.numel()]
        w = w / (w.sum() + 1e-8)
        edge_loss = (F.huber_loss(sampled, torch.zeros_like(sampled), reduction="none", delta=cfg.huber_delta) * w).sum()
        prior = cfg.pose_prior_weight * (delta_value * delta_value).sum()
        return edge_loss + prior, pose, valid, sampled

    with torch.no_grad():
        initial_loss_t, _, initial_valid, _ = eval_loss(torch.zeros(6, dtype=torch.float32, device=device))
    if initial_loss_t is None:
        refined_w2c = np.asarray(coarse_w2c, dtype=np.float32)
        return PoseRefineResult(
            refined_w2c=refined_w2c,
            refined_c2w=_invert_se3_np(refined_w2c),
            success=False,
            reason="insufficient_initial_valid_points",
            loss_before=None,
            loss_after=None,
            valid_points=int(initial_valid.sum().item()) if initial_valid is not None else 0,
            delta_rot_deg=0.0,
            delta_trans=0.0,
            config=asdict(cfg),
        )

    loss_after = None
    valid_points = int(initial_valid.sum().item())
    for _ in range(int(cfg.iterations)):
        optimizer.zero_grad()
        loss_t, _, valid, _ = eval_loss(current_delta())
        if loss_t is None:
            break
        loss_t.backward()
        optimizer.step()
        loss_after = float(loss_t.detach().cpu())
        valid_points = int(valid.sum().item())

    with torch.no_grad():
        delta_final = current_delta()
        final_loss_t, final_pose, final_valid, _ = eval_loss(delta_final)
        loss_before = float(initial_loss_t.detach().cpu())
        loss_final = float(final_loss_t.detach().cpu()) if final_loss_t is not None else None
        delta_pose = se3_exp(delta_final.detach())
        delta_rot_deg = _rotation_angle_deg(delta_pose[:3, :3])
        delta_trans = float(torch.linalg.norm(delta_pose[:3, 3]).detach().cpu())
        keep = True
        reason = "success"
        if final_loss_t is None or int(final_valid.sum().item()) < cfg.min_valid_points:
            keep = False
            reason = "insufficient_final_valid_points"
        elif loss_final >= loss_before:
            keep = False
            reason = "loss_not_improved"
        elif delta_rot_deg > cfg.max_delta_rot_deg or delta_trans > cfg.max_delta_trans:
            keep = False
            reason = "delta_too_large"
        refined = final_pose if keep else coarse
        refined_w2c = refined.detach().cpu().numpy().astype(np.float32)
        refined_c2w = _invert_se3_torch(refined).detach().cpu().numpy().astype(np.float32)

    return PoseRefineResult(
        refined_w2c=refined_w2c,
        refined_c2w=refined_c2w,
        success=bool(keep),
        reason=reason,
        loss_before=loss_before,
        loss_after=loss_final if loss_final is not None else loss_after,
        valid_points=valid_points,
        delta_rot_deg=delta_rot_deg if keep else 0.0,
        delta_trans=delta_trans if keep else 0.0,
        config=asdict(cfg),
    )
