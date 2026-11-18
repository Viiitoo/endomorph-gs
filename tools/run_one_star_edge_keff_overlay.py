import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pose.contour_2d import extract_edge_and_distance_rgb
from pose.mast3r_pointmap import infer_mast3r_pointmap, load_mast3r_model, setup_s3po_imports
from pose.star_edge import StarEdgeConfig, StarEdgeExtractor


K_EFF = np.array(
    [
        [406.08, 0.0, 320.29],
        [0.0, 407.30, 239.65],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def fit_effective_k(pointmap):
    h, w = pointmap.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    points = pointmap.reshape(-1, 3)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (np.abs(z) > 1e-6)
    xn = points[:, 0] / z
    yn = points[:, 1] / z
    ax = np.stack([xn[valid], np.ones(int(valid.sum()))], axis=1)
    ay = np.stack([yn[valid], np.ones(int(valid.sum()))], axis=1)
    fx, cx = np.linalg.lstsq(ax, xx.reshape(-1)[valid], rcond=None)[0]
    fy, cy = np.linalg.lstsq(ay, yy.reshape(-1)[valid], rcond=None)[0]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def draw_points(image_rgb, uv, color_rgb, radius=1):
    out = image_rgb.copy()
    color_bgr = tuple(int(c) for c in color_rgb[::-1])
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    for x, y in uv.astype(np.int32):
        cv2.circle(out_bgr, (int(x), int(y)), radius, color_bgr, -1, lineType=cv2.LINE_AA)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def project_keff(points_cam, width, height, K, margin=0):
    points = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    u = K[0, 0] * points[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    v = K[1, 1] * points[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    valid &= (u >= margin) & (u < width - margin) & (v >= margin) & (v < height - margin)
    return np.stack([u[valid], v[valid]], axis=-1).astype(np.float32), valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="data/stereo_s3po_seq_easy_adapt")
    parser.add_argument("--frame", type=int, default=20)
    parser.add_argument("--pair_frame", type=int, default=23)
    parser.add_argument("--output_dir", default="output/phase0/star_edge_keff_single")
    parser.add_argument("--s3po_root", default="external/S3PO-GS_original")
    parser.add_argument("--checkpoint", default="external/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    parser.add_argument("--star_edge_root", default="external/STAR-Edge")
    parser.add_argument("--star_edge_model", default="external/STAR-Edge/model/best.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_max_points", type=int, default=4000)
    parser.add_argument("--bbox_trim_quantile", type=float, default=0.01)
    parser.add_argument("--point_stride", type=int, default=2)
    parser.add_argument("--fit_keff", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = dataset_root / "images" / f"{args.frame:06d}.png"
    pair_path = dataset_root / "images" / f"{args.pair_frame:06d}.png"
    rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    rgb_pair = np.asarray(Image.open(pair_path).convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]

    setup_s3po_imports(args.s3po_root)
    model = load_mast3r_model(args.s3po_root, args.checkpoint, device=args.device)
    pointmap, confidence = infer_mast3r_pointmap(rgb, rgb_pair, model, device=args.device, size=args.size)
    np.save(out_dir / f"{args.frame:06d}_pointmap.npy", pointmap.astype(np.float32))
    if confidence is not None:
        np.save(out_dir / f"{args.frame:06d}_confidence.npy", confidence.astype(np.float32))
    K = fit_effective_k(pointmap) if args.fit_keff else K_EFF

    point_stride = max(1, int(args.point_stride))
    points = pointmap[::point_stride, ::point_stride].reshape(-1, 3)
    raw_extractor = StarEdgeExtractor(
        StarEdgeConfig(
            star_edge_root=args.star_edge_root,
            model_path=args.star_edge_model,
            threshold=args.threshold,
            postprocess=False,
            use_original_refine=True,
        )
    )
    post_extractor = StarEdgeExtractor(
        StarEdgeConfig(
            star_edge_root=args.star_edge_root,
            model_path=args.star_edge_model,
            threshold=args.threshold,
            postprocess=True,
            use_original_refine=True,
            bbox_trim_quantile=args.bbox_trim_quantile,
            output_max_points=args.output_max_points,
        )
    )
    raw = raw_extractor.extract(points)
    post = post_extractor._postprocess_contour(raw)

    raw_uv, raw_valid = project_keff(raw.points, w, h, K, margin=0)
    post_uv, post_valid = project_keff(post.points, w, h, K, margin=0)
    edge = extract_edge_and_distance_rgb(rgb, num_levels=3, border_width=0)
    edge_mask = edge.edge.astype(bool)

    edge_overlay = rgb.copy()
    edge_overlay[edge_mask] = (0.35 * edge_overlay[edge_mask] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    raw_overlay = draw_points(edge_overlay, raw_uv, np.array([255, 0, 255], dtype=np.uint8), radius=1)
    post_overlay = draw_points(edge_overlay, post_uv, np.array([0, 255, 255], dtype=np.uint8), radius=1)
    combined = draw_points(raw_overlay, post_uv, np.array([0, 255, 255], dtype=np.uint8), radius=1)

    Image.fromarray(edge_overlay).save(out_dir / f"{args.frame:06d}_2d_edge_green.png")
    Image.fromarray(raw_overlay).save(out_dir / f"{args.frame:06d}_raw_star_edge_keff_magenta.png")
    Image.fromarray(post_overlay).save(out_dir / f"{args.frame:06d}_post_star_edge_keff_cyan.png")
    Image.fromarray(combined).save(out_dir / f"{args.frame:06d}_combined_raw_magenta_post_cyan.png")

    meta = {
        "frame": args.frame,
        "pair_frame": args.pair_frame,
        "image": str(img_path),
        "K_eff": K.tolist(),
        "K_eff_source": "fit_current_pointmap" if args.fit_keff else "hardcoded_docs",
        "raw_points": int(len(raw.points)),
        "raw_projected_inside": int(len(raw_uv)),
        "post_points": int(len(post.points)),
        "post_projected_inside": int(len(post_uv)),
        "threshold": args.threshold,
        "point_stride": point_stride,
        "input_points": int(len(points)),
        "bbox_trim_quantile": args.bbox_trim_quantile,
        "output_max_points": args.output_max_points,
        "outputs": {
            "edge": f"{args.frame:06d}_2d_edge_green.png",
            "raw": f"{args.frame:06d}_raw_star_edge_keff_magenta.png",
            "post": f"{args.frame:06d}_post_star_edge_keff_cyan.png",
            "combined": f"{args.frame:06d}_combined_raw_magenta_post_cyan.png",
        },
    }
    with open(out_dir / f"{args.frame:06d}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
