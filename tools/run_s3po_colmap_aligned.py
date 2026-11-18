import argparse
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pycolmap


def _call_or_value(obj):
    return obj() if callable(obj) else obj


def _load_intrinsics(calib_dir):
    calib_files = sorted(Path(calib_dir).glob("*.txt"))
    if not calib_files:
        raise FileNotFoundError(f"No calibration files in {calib_dir}")
    text = calib_files[0].read_text()
    values = {}
    for key in ("fx", "fy", "cx", "cy"):
        match = re.search(rf"{key}:\s*([-+0-9.eE]+)", text)
        if match is None:
            raise ValueError(f"Missing {key} in {calib_files[0]}")
        values[key] = float(match.group(1))
    return values


def _select_best_reconstruction(reconstructions):
    if isinstance(reconstructions, dict):
        candidates = list(reconstructions.values())
    elif isinstance(reconstructions, (list, tuple)):
        candidates = list(reconstructions)
    elif reconstructions is None:
        candidates = []
    else:
        candidates = [reconstructions]
    if not candidates:
        return None
    return max(candidates, key=lambda r: int(_call_or_value(getattr(r, "num_reg_images", 0))))


def _extract_w2c_by_name(reconstruction):
    poses = {}
    if reconstruction is None:
        return poses
    reg_ids = _call_or_value(getattr(reconstruction, "reg_image_ids", []))
    for image_id in list(reg_ids):
        image = reconstruction.image(image_id)
        if image is None:
            continue
        name = getattr(image, "name", None)
        cam_from_world = getattr(image, "cam_from_world", None)
        if name is None or cam_from_world is None:
            continue
        cam_from_world = _call_or_value(cam_from_world)
        w2c = np.asarray(cam_from_world.matrix(), dtype=np.float64)
        if w2c.shape == (3, 4):
            w2c = np.vstack([w2c, np.array([0.0, 0.0, 0.0, 1.0])])
        poses[Path(name).name] = w2c
    return poses


def _invert_se3(T):
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _umeyama_sim3(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    sigma = (xd.T @ xs) / len(src)
    U, D, Vt = np.linalg.svd(sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum(axis=1).mean()
    scale = (D * np.diag(S)).sum() / max(var_s, 1e-20)
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def _nearest_fill(poses, valid_mask):
    n = len(poses)
    valid = np.flatnonzero(valid_mask)
    if len(valid) == 0:
        raise RuntimeError("COLMAP registered no frames")
    filled = []
    fill_from = []
    for i in range(n):
        if valid_mask[i]:
            filled.append(poses[i])
            fill_from.append(i)
            continue
        j = int(valid[np.argmin(np.abs(valid - i))])
        filled.append(poses[j].copy())
        fill_from.append(j)
    return np.stack(filled, axis=0), fill_from


def run_colmap_aligned(
    dataset_root,
    output_dir,
    workspace=None,
    use_gpu=False,
    overlap=10,
    max_image_size=1024,
    max_num_features=8192,
):
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    rgb_dir = dataset_root / "rgb"
    calib_dir = dataset_root / "calib"
    gt_dir = dataset_root / "gt"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(rgb_dir)
    if not gt_dir.is_dir():
        raise FileNotFoundError(gt_dir)
    intr = _load_intrinsics(calib_dir)

    frames = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    frame_indices = [int(p.stem) for p in frames]
    n_frames = len(frames)
    if n_frames == 0:
        raise RuntimeError(f"No RGB frames found in {rgb_dir}")

    created_tmp = False
    if workspace is None:
        workspace = Path(tempfile.mkdtemp(prefix="s3po_colmap_"))
        created_tmp = True
    else:
        workspace = Path(workspace)
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)

    image_dir = workspace / "images"
    sparse_dir = workspace / "sparse"
    database_path = workspace / "database.db"
    image_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    name_to_pos = {}
    for pos, src in enumerate(frames):
        dst = image_dir / src.name
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)
        name_to_pos[src.name] = pos

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "PINHOLE"
    reader_options.camera_params = (
        f"{intr['fx']:.8f},{intr['fy']:.8f},{intr['cx']:.8f},{intr['cy']:.8f}"
    )

    extraction_options = pycolmap.FeatureExtractionOptions()
    if hasattr(extraction_options, "max_image_size"):
        extraction_options.max_image_size = int(max_image_size)
    if hasattr(extraction_options, "use_gpu"):
        extraction_options.use_gpu = bool(use_gpu)
    if hasattr(extraction_options, "sift") and hasattr(extraction_options.sift, "max_num_features"):
        extraction_options.sift.max_num_features = int(max_num_features)

    t0 = time.time()
    pycolmap.extract_features(
        str(database_path),
        str(image_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model="PINHOLE",
        reader_options=reader_options,
        extraction_options=extraction_options,
    )
    t1 = time.time()

    pairing_options = pycolmap.SequentialPairingOptions()
    if hasattr(pairing_options, "overlap"):
        pairing_options.overlap = int(overlap)
    pycolmap.match_sequential(str(database_path), pairing_options=pairing_options)
    t2 = time.time()

    reconstructions = pycolmap.incremental_mapping(
        str(database_path),
        str(image_dir),
        str(sparse_dir),
        options=pycolmap.IncrementalPipelineOptions(),
    )
    t3 = time.time()

    reconstruction = _select_best_reconstruction(reconstructions)
    w2c_by_name = _extract_w2c_by_name(reconstruction)

    raw_c2w = [None] * n_frames
    valid_mask = np.zeros(n_frames, dtype=bool)
    for name, w2c in w2c_by_name.items():
        pos = name_to_pos.get(name)
        if pos is None:
            continue
        raw_c2w[pos] = _invert_se3(w2c)
        valid_mask[pos] = True
    raw_c2w_filled, fill_from = _nearest_fill(raw_c2w, valid_mask)

    gt_c2w_mm = []
    for idx in frame_indices:
        c2w = np.loadtxt(gt_dir / f"{idx:06d}.txt").reshape(4, 4).astype(np.float64)
        c2w[:3, 3] *= 1000.0
        gt_c2w_mm.append(c2w)
    gt_c2w_mm = np.stack(gt_c2w_mm, axis=0)

    valid = np.flatnonzero(valid_mask)
    if len(valid) < 3:
        raise RuntimeError(f"Need at least 3 registered frames for Sim3 alignment, got {len(valid)}")
    src_centers = raw_c2w_filled[valid, :3, 3]
    dst_centers = gt_c2w_mm[valid, :3, 3]
    scale, R_align, t_align = _umeyama_sim3(src_centers, dst_centers)

    aligned_c2w = raw_c2w_filled.copy()
    for i in range(n_frames):
        aligned_c2w[i, :3, :3] = R_align @ raw_c2w_filled[i, :3, :3]
        aligned_c2w[i, :3, 3] = scale * (R_align @ raw_c2w_filled[i, :3, 3]) + t_align
        aligned_c2w[i, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])

    aligned_w2c = np.stack([_invert_se3(p) for p in aligned_c2w], axis=0)
    raw_w2c_filled = np.stack([_invert_se3(p) for p in raw_c2w_filled], axis=0)

    aligned_centers = aligned_c2w[valid, :3, 3]
    align_err = np.linalg.norm(aligned_centers - dst_centers, axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "poses_c2w.npy", aligned_c2w.astype(np.float32))
    np.save(output_dir / "poses_w2c.npy", aligned_w2c.astype(np.float32))
    np.save(output_dir / "raw_poses_c2w.npy", raw_c2w_filled.astype(np.float32))
    np.save(output_dir / "raw_poses_w2c.npy", raw_w2c_filled.astype(np.float32))

    meta = {
        "dataset_root": str(dataset_root),
        "num_frames": n_frames,
        "registered_count": int(valid_mask.sum()),
        "registered_frames": [int(frame_indices[i]) for i in valid],
        "filled_from": {f"{frame_indices[i]:06d}": int(frame_indices[fill_from[i]]) for i in range(n_frames) if not valid_mask[i]},
        "intrinsics": intr,
        "alignment": {
            "type": "gt_sim3",
            "scale": float(scale),
            "rotation": R_align.tolist(),
            "translation_mm": t_align.tolist(),
            "error_mm": {
                "mean": float(align_err.mean()),
                "median": float(np.median(align_err)),
                "max": float(align_err.max()),
            },
        },
        "timing_s": {
            "extract_features": t1 - t0,
            "match_sequential": t2 - t1,
            "incremental_mapping": t3 - t2,
            "total": t3 - t0,
        },
        "workspace": str(workspace),
    }
    with open(output_dir / "pose_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if created_tmp:
        shutil.rmtree(workspace, ignore_errors=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Run raw COLMAP and align it to GT with Sim3 for S3PO StereoMIS.")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--max_image_size", type=int, default=1024)
    parser.add_argument("--max_num_features", type=int, default=8192)
    args = parser.parse_args()

    meta = run_colmap_aligned(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        workspace=args.workspace,
        use_gpu=args.use_gpu,
        overlap=args.overlap,
        max_image_size=args.max_image_size,
        max_num_features=args.max_num_features,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
