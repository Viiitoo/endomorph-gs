import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from pose.alignment import apply_sim3_to_c2w, invert_poses, umeyama_sim3
from pose.contour_2d import extract_edge_and_distance_rgb
from pose.contour_3d import extract_geometric_contours
from pose.contour_refiner import RefineConfig, refine_pose
from pose.star_edge import StarEdgeConfig, StarEdgeExtractor


def load_intrinsics(calib_dir):
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
    return np.array([[values["fx"], 0.0, values["cx"]], [0.0, values["fy"], values["cy"]], [0.0, 0.0, 1.0]], dtype=np.float32), values


def invert_se3(mat):
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    out = np.eye(4, dtype=mat.dtype)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out


def load_registered_indices(colmap_meta_path, n_frames):
    if not colmap_meta_path:
        return list(range(n_frames))
    with open(colmap_meta_path, "r") as f:
        meta = json.load(f)
    frames = meta.get("registered_frames", [])
    return [int(x) for x in frames if 0 <= int(x) < n_frames]


def load_depth_png(path, depth_scale):
    depth = np.array(Image.open(path)).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / float(depth_scale)


def backproject_depth(depth, K, stride=2, max_points=120000):
    ys, xs = np.where(np.isfinite(depth) & (depth > 0))
    if stride > 1:
        keep_stride = ((ys % stride) == 0) & ((xs % stride) == 0)
        ys, xs = ys[keep_stride], xs[keep_stride]
    if len(ys) > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(ys), size=max_points, replace=False)
        ys, xs = ys[keep], xs[keep]
    z = depth[ys, xs]
    x = (xs.astype(np.float32) - K[0, 2]) / K[0, 0] * z
    y = (ys.astype(np.float32) - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1).astype(np.float32)


def transform_points(mat, points):
    pts_h = np.concatenate([points, np.ones((len(points), 1), dtype=points.dtype)], axis=1)
    return (mat @ pts_h.T).T[:, :3].astype(np.float32)


def load_pointmap_or_depth(dataset_root, frame_idx, K, pointmap_dir, depth_scale, depth_stride, allow_depth_fallback):
    if pointmap_dir is not None:
        path = Path(pointmap_dir) / f"{frame_idx:06d}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        points = np.load(path).astype(np.float32)
        return points.reshape(-1, 3)
    if not allow_depth_fallback:
        raise RuntimeError(
            "No --pointmap_dir provided. For the planned MASt3R->STAR-Edge pipeline, "
            "generate MASt3R pointmaps first or pass --allow_depth_fallback explicitly."
        )
    depth_path = Path(dataset_root) / "depth" / f"{frame_idx:06d}.png"
    depth = load_depth_png(depth_path, depth_scale=depth_scale)
    return backproject_depth(depth, K, stride=depth_stride)


def choose_reference(frame_idx, registered):
    candidates = [i for i in registered if i != frame_idx]
    if not candidates:
        return None
    previous = [i for i in candidates if i < frame_idx]
    if previous:
        return max(previous)
    return min(candidates)


def load_gt_c2w(dataset_root, frame_indices, pose_scale):
    out = []
    for idx in frame_indices:
        mat = np.loadtxt(Path(dataset_root) / "gt" / f"{idx:06d}.txt").reshape(4, 4).astype(np.float64)
        mat[:3, 3] *= float(pose_scale)
        out.append(mat)
    return np.stack(out, axis=0)


def run_contour_refine(args):
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    K, intrinsics = load_intrinsics(dataset_root / "calib")
    rgb_paths = sorted((dataset_root / "rgb").glob("*.png"))
    if not rgb_paths:
        raise RuntimeError(f"No RGB images found in {dataset_root / 'rgb'}")
    frame_indices = [int(p.stem) for p in rgb_paths]
    n_frames = len(frame_indices)

    coarse_c2w = np.load(args.coarse_c2w).astype(np.float64)
    if coarse_c2w.shape != (n_frames, 4, 4):
        raise ValueError(f"Expected coarse poses shape {(n_frames, 4, 4)}, got {coarse_c2w.shape}")
    coarse_w2c = invert_poses(coarse_c2w)
    registered = load_registered_indices(args.colmap_meta, n_frames)
    registered = [i for i in registered if 0 <= i < n_frames]
    if args.max_refine_frames is not None:
        registered = registered[: int(args.max_refine_frames)]

    refined_c2w = coarse_c2w.copy()
    refined_w2c = coarse_w2c.copy()
    per_frame = {}
    edge_dir = output_dir / "debug_edges"
    overlay_dir = output_dir / "debug_overlays"
    if args.save_debug:
        edge_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

    config = RefineConfig(
        iterations=args.iterations,
        lr_rot=args.lr_rot,
        lr_trans=args.lr_trans,
        huber_delta=args.huber_delta,
        pose_prior_weight=args.pose_prior_weight,
        min_valid_points=args.min_valid_points,
        margin_pixels=args.margin_pixels,
        max_delta_rot_deg=args.max_delta_rot_deg,
        max_delta_trans=args.max_delta_trans,
        device=args.device,
    )

    contour_cache = {}
    star_edge_extractor = None
    if args.contour3d_method == "star_edge":
        if not args.star_edge_root:
            raise RuntimeError("--star_edge_root is required when --contour3d_method star_edge")
        star_edge_extractor = StarEdgeExtractor(
            StarEdgeConfig(
                star_edge_root=args.star_edge_root,
                model_path=args.star_edge_model,
                threshold=args.star_edge_threshold,
                bw=args.star_edge_bw,
                kk=args.star_edge_kk,
                sample_num=args.star_edge_sample_num,
                max_points=args.star_edge_max_points,
                use_cuda_mlp=args.star_edge_use_cuda_mlp,
                postprocess=not args.disable_star_edge_postprocess,
                sor_k=args.star_edge_sor_k,
                sor_alpha=args.star_edge_sor_alpha,
                curvature_k=args.star_edge_curvature_k,
                curvature_max_percentile=args.star_edge_curvature_max_percentile,
                min_neighbor_support=args.star_edge_min_neighbor_support,
                bbox_trim_quantile=args.star_edge_bbox_trim_quantile,
                output_max_points=args.star_edge_output_max_points,
            )
        )

    for frame_idx in tqdm(registered, desc="Contour pose refinement"):
        ref_idx = choose_reference(frame_idx, registered)
        if ref_idx is None:
            per_frame[str(frame_idx)] = {"status": "skipped", "reason": "no_reference"}
            continue
        if ref_idx not in contour_cache:
            ref_points_cam = load_pointmap_or_depth(
                dataset_root,
                frame_indices[ref_idx],
                K,
                args.pointmap_dir,
                args.depth_scale,
                args.depth_stride,
                args.allow_depth_fallback,
            )
            ref_points_world = transform_points(coarse_c2w[ref_idx], ref_points_cam)
            if args.contour3d_method == "star_edge":
                try:
                    contour = star_edge_extractor.extract(ref_points_world)
                except Exception:
                    if not args.allow_geometric_fallback:
                        raise
                    contour = extract_geometric_contours(
                        ref_points_world,
                        max_points=args.max_contour_points,
                        k_neighbors=args.k_neighbors,
                        edge_percentile=args.edge_percentile,
                        voxel_size=args.voxel_size,
                    )
            else:
                contour = extract_geometric_contours(
                    ref_points_world,
                    max_points=args.max_contour_points,
                    k_neighbors=args.k_neighbors,
                    edge_percentile=args.edge_percentile,
                    voxel_size=args.voxel_size,
                )
            contour_cache[ref_idx] = contour
        contour = contour_cache[ref_idx]
        if len(contour.points) < args.min_valid_points:
            per_frame[str(frame_idx)] = {
                "status": "skipped",
                "reason": "insufficient_3d_contour",
                "reference_frame": int(ref_idx),
                "contour_points": int(len(contour.points)),
            }
            continue

        rgb = np.array(Image.open(rgb_paths[frame_idx]).convert("RGB"), dtype=np.uint8)
        edge_result = extract_edge_and_distance_rgb(rgb, num_levels=args.edge_levels, border_width=args.margin_pixels)
        result = refine_pose(
            coarse_w2c[frame_idx],
            K,
            contour.points,
            edge_result.distance_field,
            image_size=rgb.shape[:2],
            weights=contour.weights,
            config=config,
        )
        refined_w2c[frame_idx] = result.refined_w2c
        refined_c2w[frame_idx] = result.refined_c2w
        per_frame[str(frame_idx)] = {
            "status": "refined" if result.success else "coarse_kept",
            "success": bool(result.success),
            "reason": result.reason,
            "reference_frame": int(ref_idx),
            "contour_points": int(len(contour.points)),
            "valid_points": int(result.valid_points),
            "loss_before": result.loss_before,
            "loss_after": result.loss_after,
            "delta_rot_deg": float(result.delta_rot_deg),
            "delta_trans": float(result.delta_trans),
        }
        if args.save_debug:
            cv2.imwrite(str(edge_dir / f"{frame_indices[frame_idx]:06d}.png"), edge_result.edge * 255)

    np.save(output_dir / "poses_refined_c2w.npy", refined_c2w.astype(np.float32))
    np.save(output_dir / "poses_refined_w2c.npy", refined_w2c.astype(np.float32))

    align_meta = None
    if args.align_gt:
        fit_indices = [i for i in registered if str(i) in per_frame and per_frame[str(i)].get("status") in ("refined", "coarse_kept")]
        if len(fit_indices) < 3:
            raise RuntimeError(f"Need at least 3 frames for GT Sim3, got {len(fit_indices)}")
        gt = load_gt_c2w(dataset_root, frame_indices, pose_scale=args.gt_pose_scale)
        sim3 = umeyama_sim3(refined_c2w[fit_indices, :3, 3], gt[fit_indices, :3, 3])
        aligned_c2w = apply_sim3_to_c2w(refined_c2w, sim3)
        aligned_w2c = invert_poses(aligned_c2w)
        np.save(output_dir / "poses_refined_sim3_c2w.npy", aligned_c2w.astype(np.float32))
        np.save(output_dir / "poses_refined_sim3_w2c.npy", aligned_w2c.astype(np.float32))
        err = np.linalg.norm(aligned_c2w[fit_indices, :3, 3] - gt[fit_indices, :3, 3], axis=1)
        align_meta = {
            "type": "gt_sim3_after_refine",
            "scale": float(sim3.scale),
            "rotation": sim3.rotation.tolist(),
            "translation": sim3.translation.tolist(),
            "fit_frames": [int(frame_indices[i]) for i in fit_indices],
            "error": {
                "mean": float(err.mean()),
                "median": float(np.median(err)),
                "max": float(err.max()),
            },
        }

    meta = {
        "dataset_root": str(dataset_root),
        "num_frames": n_frames,
        "coarse_c2w": str(args.coarse_c2w),
        "pointmap_source": f"mast3r_pointmap_dir:{args.pointmap_dir}" if args.pointmap_dir else "depth_backproject_fallback",
        "contour3d_method": args.contour3d_method,
        "geometric_fallback_allowed": bool(args.allow_geometric_fallback),
        "depth_fallback_allowed": bool(args.allow_depth_fallback),
        "registered_frames": [int(frame_indices[i]) for i in registered],
        "intrinsics": intrinsics,
        "config": vars(args),
        "alignment": align_meta,
        "per_frame": per_frame,
    }
    with open(output_dir / "pose_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Refine registered COLMAP poses with 3D/2D contour distance fields.")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--coarse_c2w", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--colmap_meta", default="")
    parser.add_argument("--pointmap_dir", default=None)
    parser.add_argument("--contour3d_method", choices=["star_edge", "geometric"], default="star_edge")
    parser.add_argument("--star_edge_root", default="")
    parser.add_argument("--star_edge_model", default="")
    parser.add_argument("--star_edge_threshold", type=float, default=0.5)
    parser.add_argument("--star_edge_bw", type=int, default=10)
    parser.add_argument("--star_edge_kk", type=int, default=26)
    parser.add_argument("--star_edge_sample_num", type=int, default=40)
    parser.add_argument("--star_edge_max_points", type=int, default=50000)
    parser.add_argument("--star_edge_use_cuda_mlp", action="store_true")
    parser.add_argument("--disable_star_edge_postprocess", action="store_true")
    parser.add_argument("--star_edge_sor_k", type=int, default=20)
    parser.add_argument("--star_edge_sor_alpha", type=float, default=1.8)
    parser.add_argument("--star_edge_curvature_k", type=int, default=24)
    parser.add_argument("--star_edge_curvature_max_percentile", type=float, default=99.0)
    parser.add_argument("--star_edge_min_neighbor_support", type=float, default=0.35)
    parser.add_argument("--star_edge_bbox_trim_quantile", type=float, default=0.0)
    parser.add_argument("--star_edge_output_max_points", type=int, default=4000)
    parser.add_argument("--allow_geometric_fallback", action="store_true")
    parser.add_argument("--allow_depth_fallback", action="store_true")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--gt_pose_scale", type=float, default=1000.0)
    parser.add_argument("--depth_stride", type=int, default=2)
    parser.add_argument("--max_contour_points", type=int, default=4000)
    parser.add_argument("--k_neighbors", type=int, default=24)
    parser.add_argument("--edge_percentile", type=float, default=85.0)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--edge_levels", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--lr_rot", type=float, default=5e-4)
    parser.add_argument("--lr_trans", type=float, default=1e-3)
    parser.add_argument("--huber_delta", type=float, default=3.0)
    parser.add_argument("--pose_prior_weight", type=float, default=1e-4)
    parser.add_argument("--min_valid_points", type=int, default=64)
    parser.add_argument("--margin_pixels", type=int, default=20)
    parser.add_argument("--max_delta_rot_deg", type=float, default=2.0)
    parser.add_argument("--max_delta_trans", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--align_gt", action="store_true")
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--max_refine_frames", type=int, default=None)
    args = parser.parse_args()
    meta = run_contour_refine(args)
    print(json.dumps({"output_dir": args.output_dir, "alignment": meta.get("alignment")}, indent=2))


if __name__ == "__main__":
    main()
