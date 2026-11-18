import sys
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision.transforms as tvf
from PIL.ImageOps import exif_transpose


ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def setup_s3po_imports(s3po_root):
    root = Path(s3po_root)
    if not root.exists():
        raise FileNotFoundError(f"S3PO root not found: {root}")
    for path in (root, root / "dust3r"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _resize_pil_image(img, long_edge_size):
    scale = max(img.size)
    interp = PIL.Image.LANCZOS if scale > long_edge_size else PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / scale)) for x in img.size)
    return img.resize(new_size, interp)


def rgb_to_dust3r_view(rgb, size, idx, square_ok=False):
    img = PIL.Image.fromarray(np.asarray(rgb, dtype=np.uint8), "RGB")
    img = exif_transpose(img).convert("RGB")
    w0, h0 = img.size
    if size == 224:
        img = _resize_pil_image(img, round(size * max(w0 / h0, h0 / w0)))
    else:
        img = _resize_pil_image(img, size)
    w, h = img.size
    cx, cy = w // 2, h // 2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not square_ok and w == h:
            halfh = 3 * halfw // 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
    return {
        "img": ImgNorm(img)[None],
        "true_shape": np.int32([img.size[::-1]]),
        "idx": idx,
        "instance": str(idx),
    }


def load_mast3r_model(s3po_root, checkpoint, device="cuda"):
    setup_s3po_imports(s3po_root)
    from mast3r.model import AsymmetricMASt3R

    model = AsymmetricMASt3R.from_pretrained(str(checkpoint))
    model.to(device)
    model.eval()
    return model


def infer_mast3r_pointmap(rgb_a, rgb_b, model, device="cuda", size=512):
    setup_device = torch.device(device)
    from dust3r.inference import inference

    h, w = rgb_a.shape[:2]
    view_a = rgb_to_dust3r_view(rgb_a, size=size, idx=0)
    view_b = rgb_to_dust3r_view(rgb_b, size=size, idx=1)
    with torch.no_grad():
        output = inference([(view_a, view_b)], model, str(setup_device), batch_size=1, verbose=False)
    pts = output["pred1"]["pts3d"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    conf = output["pred1"].get("conf")
    conf_np = None
    if conf is not None:
        conf_np = conf.squeeze(0).detach().cpu().numpy().astype(np.float32)

    pts_resized = np.stack(
        [cv2.resize(pts[..., c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(3)],
        axis=-1,
    ).astype(np.float32)
    if conf_np is not None:
        conf_np = cv2.resize(conf_np, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return pts_resized, conf_np
