import torch


def skew_symmetric(x):
    mat = torch.zeros(3, 3, dtype=x.dtype, device=x.device)
    mat[0, 1] = -x[2]
    mat[0, 2] = x[1]
    mat[1, 0] = x[2]
    mat[1, 2] = -x[0]
    mat[2, 0] = -x[1]
    mat[2, 1] = x[0]
    return mat


def so3_exp(theta):
    eye = torch.eye(3, dtype=theta.dtype, device=theta.device)
    omega = skew_symmetric(theta)
    omega2 = omega @ omega
    angle = torch.linalg.norm(theta)
    if angle < 1e-7:
        return eye + omega + 0.5 * omega2
    return (
        eye
        + torch.sin(angle) / angle * omega
        + (1.0 - torch.cos(angle)) / (angle * angle) * omega2
    )


def se3_exp(delta):
    """Exponential map for [tx, ty, tz, rx, ry, rz]."""
    rho = delta[:3]
    theta = delta[3:]
    rot = so3_exp(theta)
    eye = torch.eye(3, dtype=delta.dtype, device=delta.device)
    omega = skew_symmetric(theta)
    omega2 = omega @ omega
    angle = torch.linalg.norm(theta)
    if angle < 1e-7:
        v_mat = eye + 0.5 * omega + (1.0 / 6.0) * omega2
    else:
        v_mat = (
            eye
            + (1.0 - torch.cos(angle)) / (angle * angle) * omega
            + (angle - torch.sin(angle)) / (angle ** 3) * omega2
        )
    trans = v_mat @ rho
    out = torch.eye(4, dtype=delta.dtype, device=delta.device)
    out[:3, :3] = rot
    out[:3, 3] = trans
    return out


def invert_se3_np(mat):
    import numpy as np

    rot = mat[:3, :3]
    trans = mat[:3, 3]
    out = np.eye(4, dtype=mat.dtype)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out

