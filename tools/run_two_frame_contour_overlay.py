import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pose.contour_2d import extract_edge_and_distance_rgb
from pose.mast3r_pointmap import infer_mast3r_pointmap, load_mast3r_model, setup_s3po_imports
from pose.star_edge import StarEdgeConfig, StarEdgeExtractor


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


def project(points_cam, K, width, height, margin=0):
    points = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    u = K[0, 0] * points[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    v = K[1, 1] * points[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    valid &= (u >= margin) & (u < width - margin) & (v >= margin) & (v < height - margin)
    return np.stack([u[valid], v[valid]], axis=-1).astype(np.float32)


def parse_frames(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def process_frame(args, frame, extractor):
    dataset_root = Path(args.dataset_root)
    img_path = dataset_root / "images" / f"{frame:06d}.png"
    pair_path = dataset_root / "images" / f"{frame + args.pair_offset:06d}.png"
    if not pair_path.exists():
        pair_path = dataset_root / "images" / f"{frame - args.pair_offset:06d}.png"
    rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    rgb_pair = np.asarray(Image.open(pair_path).convert("RGB"), dtype=np.uint8)
    bgr_for_edge = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr_for_edge is None:
        raise RuntimeError(f"Could not read {img_path}")

    pointmap, _ = infer_mast3r_pointmap(rgb, rgb_pair, args.model, device=args.device, size=args.size)
    K = fit_effective_k(pointmap)
    h, w = rgb.shape[:2]
    points = pointmap[:: args.point_stride, :: args.point_stride].reshape(-1, 3)
    raw = extractor.extract(points)
    post = extractor._postprocess_contour(raw)
    uv3d = project(post.points, K, w, h, margin=0)

    edge = extract_edge_and_distance_rgb(bgr_for_edge, num_levels=args.edge_levels, border_width=args.edge_border)
    y2d, x2d = np.nonzero(edge.edge > 0)
    uv2d = np.stack([x2d, y2d], axis=-1).astype(np.float32)

    return {
        "frame": frame,
        "pair_frame": int(Path(pair_path).stem),
        "rgb": rgb,
        "K": K,
        "uv2d": uv2d,
        "uv3d": uv3d,
        "raw_points": len(raw.points),
        "post_points": len(post.points),
        "projected_3d": len(uv3d),
    }


def draw_panel(ax, result):
    ax.imshow(result["rgb"])
    uv2d = result["uv2d"]
    uv3d = result["uv3d"]
    if len(uv2d):
        ax.scatter(uv2d[:, 0], uv2d[:, 1], s=3.0, c="#00ff35", alpha=0.85, linewidths=0, label="2D")
    if len(uv3d):
        ax.scatter(uv3d[:, 0], uv3d[:, 1], s=3.0, c="#ff00ff", alpha=0.75, linewidths=0, label="3D")
    k = result["K"]
    ax.set_title(
        f"frame {result['frame']}  2D={len(uv2d)}  3D={len(uv3d)}\n"
        f"Kfit fx={k[0,0]:.1f} fy={k[1,1]:.1f} cx={k[0,2]:.1f} cy={k[1,2]:.1f}",
        fontsize=9,
    )
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="data/stereo_s3po_seq_easy_adapt")
    parser.add_argument("--frames", default="20,30")
    parser.add_argument("--pair_offset", type=int, default=3)
    parser.add_argument("--output_dir", default="output/phase0/two_frame_contour_overlay")
    parser.add_argument("--s3po_root", default="external/S3PO-GS_original")
    parser.add_argument("--checkpoint", default="external/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    parser.add_argument("--star_edge_root", default="external/STAR-Edge")
    parser.add_argument("--star_edge_model", default="external/STAR-Edge/model/best.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--point_stride", type=int, default=2)
    parser.add_argument("--edge_levels", type=int, default=3)
    parser.add_argument("--edge_border", type=int, default=20)
    args = parser.parse_args()
    args.point_stride = max(1, int(args.point_stride))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_s3po_imports(args.s3po_root)
    args.model = load_mast3r_model(args.s3po_root, args.checkpoint, device=args.device)
    extractor = StarEdgeExtractor(
        StarEdgeConfig(
            star_edge_root=args.star_edge_root,
            model_path=args.star_edge_model,
            threshold=args.threshold,
            postprocess=True,
            use_original_refine=True,
            bbox_trim_quantile=0.01,
            output_max_points=0,
        )
    )

    results = [process_frame(args, frame, extractor) for frame in parse_frames(args.frames)]
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5.6), dpi=170)
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        draw_panel(ax, result)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("green=2D contour points, magenta=postprocessed 3D contour projected with fitted K", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out_png = out_dir / f"frames_{'_'.join(str(r['frame']) for r in results)}_2d3d_overlay.png"
    fig.savefig(out_png)
    plt.close(fig)

    meta = []
    for r in results:
        meta.append(
            {
                "frame": r["frame"],
                "pair_frame": r["pair_frame"],
                "K_fit": r["K"].tolist(),
                "2d_points": int(len(r["uv2d"])),
                "raw_3d_points": int(r["raw_points"]),
                "post_3d_points": int(r["post_points"]),
                "projected_3d_points": int(r["projected_3d"]),
            }
        )
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"output": out_png.name, "frames": meta}, f, indent=2)
    print(json.dumps({"output_dir": str(out_dir), "output": out_png.name, "frames": meta}, indent=2))


if __name__ == "__main__":
    main()
