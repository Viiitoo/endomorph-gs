import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from pose.mast3r_pointmap import infer_mast3r_pointmap, load_mast3r_model, setup_s3po_imports


def parse_frame_ids(text):
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def choose_pair(pos, total, strategy):
    if total < 2:
        raise RuntimeError("MASt3R pointmap generation needs at least two RGB frames.")
    if strategy == "next":
        return pos + 1 if pos + 1 < total else pos - 1
    if pos > 0:
        return pos - 1
    return 1


def run(args):
    dataset_root = Path(args.dataset_root)
    rgb_paths = sorted((dataset_root / "rgb").glob("*.png"))
    if not rgb_paths:
        raise RuntimeError(f"No RGB frames found in {dataset_root / 'rgb'}")

    requested = parse_frame_ids(args.frame_ids)
    selected_positions = list(range(len(rgb_paths)))
    if requested is not None:
        stem_to_pos = {int(path.stem): i for i, path in enumerate(rgb_paths)}
        selected_positions = [stem_to_pos[i] for i in requested if i in stem_to_pos]
    if args.max_frames is not None:
        selected_positions = selected_positions[: int(args.max_frames)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conf_dir = output_dir / "confidence"
    if args.save_confidence:
        conf_dir.mkdir(parents=True, exist_ok=True)

    setup_s3po_imports(args.s3po_root)
    model = load_mast3r_model(args.s3po_root, args.checkpoint, device=args.device)

    frames_meta = []
    for pos in tqdm(selected_positions, desc="MASt3R pointmaps"):
        pair_pos = choose_pair(pos, len(rgb_paths), args.pair_strategy)
        rgb_a = np.array(Image.open(rgb_paths[pos]).convert("RGB"), dtype=np.uint8)
        rgb_b = np.array(Image.open(rgb_paths[pair_pos]).convert("RGB"), dtype=np.uint8)
        pointmap, confidence = infer_mast3r_pointmap(rgb_a, rgb_b, model, device=args.device, size=args.size)
        frame_id = int(rgb_paths[pos].stem)
        pair_id = int(rgb_paths[pair_pos].stem)
        np.save(output_dir / f"{frame_id:06d}.npy", pointmap.astype(np.float32))
        if args.save_confidence and confidence is not None:
            np.save(conf_dir / f"{frame_id:06d}.npy", confidence.astype(np.float32))
        frames_meta.append(
            {
                "frame": frame_id,
                "paired_frame": pair_id,
                "pointmap": f"{frame_id:06d}.npy",
                "confidence": f"confidence/{frame_id:06d}.npy" if args.save_confidence and confidence is not None else None,
            }
        )

    meta = {
        "source": "MASt3R",
        "dataset_root": str(dataset_root),
        "s3po_root": str(args.s3po_root),
        "checkpoint": str(args.checkpoint),
        "size": int(args.size),
        "pair_strategy": args.pair_strategy,
        "num_frames": len(frames_meta),
        "frames": frames_meta,
    }
    with open(output_dir / "pointmap_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Generate MASt3R pointmaps from adjacent RGB frame pairs.")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--s3po_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--pair_strategy", choices=["previous", "next"], default="previous")
    parser.add_argument("--frame_ids", default="")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--save_confidence", action="store_true")
    args = parser.parse_args()
    meta = run(args)
    print(json.dumps({"output_dir": args.output_dir, "num_frames": meta["num_frames"]}, indent=2))


if __name__ == "__main__":
    main()
