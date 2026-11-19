import os
import re

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from scene.cameras import Camera
from utils.graphics_utils import BasicPointCloud, focal2fov


class S3POStereoMISDataset(object):
    """Native loader for S3PO_recons StereoMIS train folders."""

    def __init__(
        self,
        datadir,
        test_every=8,
        depth_scale=1.0,
        pose_scale=1000.0,
        mask_mode="exclude_instrument",
        max_init_points=50000,
        pose_source="gt",
        pose_path="",
    ):
        self.root_dir = datadir
        self.test_every = int(test_every)
        self.depth_scale = float(depth_scale)
        self.pose_scale = float(pose_scale)
        self.mask_mode = str(mask_mode)
        if self.mask_mode not in ("exclude_instrument", "instrument_only", "depth_only"):
            raise ValueError(f"Unknown StereoMIS mask mode: {self.mask_mode}")
        self.max_init_points = int(max_init_points)
        self.pose_source = str(pose_source)
        self.pose_path = str(pose_path)
        self.transform = T.ToTensor()

        self.image_paths = self._sorted_pngs("rgb")
        self.depth_paths = self._sorted_pngs("depth")
        self.pose_paths = self._sorted_txts("gt") if self.pose_source == "gt" else []
        self.mask_paths = self._sorted_pngs("instrument_mask", required=False)
        if not self.mask_paths:
            self.mask_paths = [None] * len(self.image_paths)

        n_frames = len(self.image_paths)
        assert n_frames > 0, f"No RGB frames found under {os.path.join(datadir, 'rgb')}"
        assert len(self.depth_paths) == n_frames, "depth frame count must match rgb"
        if self.pose_source == "gt":
            assert len(self.pose_paths) == n_frames, "gt pose count must match rgb"
        assert len(self.mask_paths) == n_frames, "instrument_mask frame count must match rgb when present"

        first_image = Image.open(self.image_paths[0])
        self.width, self.height = first_image.size
        self.fx, self.fy, self.cx, self.cy = self._load_intrinsics()
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        self.image_poses = []
        self.c2w_poses = []
        self.image_times = []
        c2w_list = self._load_c2w_poses(n_frames)
        for idx, c2w in enumerate(c2w_list):
            if c2w.shape != (4, 4):
                raise ValueError(f"Expected 4x4 c2w pose at frame {idx}, got {c2w.shape}")
            w2c = np.linalg.inv(c2w)
            self.c2w_poses.append(c2w)
            self.image_poses.append((np.transpose(w2c[:3, :3]), w2c[:3, 3]))
            self.image_times.append(idx / max(1, n_frames - 1))

        self.train_idxs = [i for i in range(n_frames) if (i - 1) % self.test_every != 0]
        self.test_idxs = [i for i in range(n_frames) if (i - 1) % self.test_every == 0]
        self.video_idxs = list(range(n_frames))
        self.maxtime = 1.0

    def _sorted_pngs(self, subdir, required=True):
        folder = os.path.join(self.root_dir, subdir)
        if not os.path.isdir(folder):
            if required:
                raise FileNotFoundError(folder)
            return []
        return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".png"))

    def _sorted_txts(self, subdir):
        folder = os.path.join(self.root_dir, subdir)
        if not os.path.isdir(folder):
            raise FileNotFoundError(folder)
        return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".txt"))

    def _resolve_pose_path(self):
        if not self.pose_path:
            raise ValueError("pose_path must be set when pose_source is not 'gt'")
        candidates = [self.pose_path]
        if not os.path.isabs(self.pose_path):
            candidates.append(os.path.join(self.root_dir, self.pose_path))
            candidates.append(os.path.abspath(self.pose_path))
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Could not find pose_path {self.pose_path}")

    def _load_c2w_poses(self, n_frames):
        if self.pose_source == "gt":
            poses = []
            for pose_path in self.pose_paths:
                c2w = np.loadtxt(pose_path).astype(np.float32)
                c2w[:3, 3] *= self.pose_scale
                poses.append(c2w)
            return poses

        if self.pose_source not in ("colmap_aligned",):
            raise ValueError(f"Unsupported pose_source: {self.pose_source}")

        poses = np.load(self._resolve_pose_path()).astype(np.float32)
        if poses.shape != (n_frames, 4, 4):
            raise ValueError(f"Expected external poses with shape {(n_frames, 4, 4)}, got {poses.shape}")
        return [poses[i] for i in range(n_frames)]

    def _load_intrinsics(self):
        calib_dir = os.path.join(self.root_dir, "calib")
        if not os.path.isdir(calib_dir):
            raise FileNotFoundError(calib_dir)
        calib_files = sorted(f for f in os.listdir(calib_dir) if f.lower().endswith(".txt"))
        if not calib_files:
            raise FileNotFoundError(f"No calibration files in {calib_dir}")

        with open(os.path.join(calib_dir, calib_files[0]), "r") as f:
            text = f.read()
        values = {}
        for key in ("fx", "fy", "cx", "cy"):
            match = re.search(rf"{key}:\s*([-+0-9.eE]+)", text)
            if match is None:
                raise ValueError(f"Missing {key} in {calib_files[0]}")
            values[key] = float(match.group(1))
        return values["fx"], values["fy"], values["cx"], values["cy"]

    def _load_depth_m(self, idx):
        depth = np.array(Image.open(self.depth_paths[idx])).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth / self.depth_scale

    def _load_valid_mask(self, idx, depth_m=None):
        if depth_m is None:
            depth_m = self._load_depth_m(idx)
        valid = depth_m > 0
        mask_path = self.mask_paths[idx]
        if mask_path is not None and self.mask_mode != "depth_only":
            inst = np.array(Image.open(mask_path))
            if inst.ndim == 3:
                inst = inst[..., 0]
            if self.mask_mode == "exclude_instrument":
                valid = np.logical_and(valid, inst == 0)
            elif self.mask_mode == "instrument_only":
                valid = np.logical_and(valid, inst != 0)
        return valid.astype(bool)

    def format_infos(self, split):
        if split == "train":
            idxs = self.train_idxs
        elif split == "test":
            idxs = self.test_idxs
        else:
            idxs = self.video_idxs

        cameras = []
        fovx = focal2fov(self.fx, self.width)
        fovy = focal2fov(self.fy, self.height)
        for uid, idx in enumerate(tqdm(idxs)):
            color = np.array(Image.open(self.image_paths[idx]).convert("RGB"), dtype=np.float32) / 255.0
            image = self.transform(color)
            depth_m = self._load_depth_m(idx)
            depth = torch.from_numpy(depth_m)
            mask = torch.from_numpy(self._load_valid_mask(idx, depth_m)).unsqueeze(0).bool()
            R, T = self.image_poses[idx]
            cameras.append(
                Camera(
                    colmap_id=idx,
                    R=R,
                    T=T,
                    FoVx=fovx,
                    FoVy=fovy,
                    image=image,
                    depth=depth,
                    mask=mask,
                    gt_alpha_mask=None,
                    image_name=f"{idx:06d}",
                    uid=uid,
                    data_device=torch.device("cuda"),
                    time=self.image_times[idx],
                    Znear=0.03,
                    Zfar=250.0,
                    K=self.K,
                    h=self.height,
                    w=self.width,
                )
            )
        return cameras

    def get_sparse_pts(self):
        depth_m = self._load_depth_m(0)
        mask = self._load_valid_mask(0, depth_m)
        ys, xs = np.where(mask)
        if ys.size == 0:
            raise RuntimeError("No valid depth pixels for point-cloud initialization")
        if ys.size > self.max_init_points:
            rng = np.random.default_rng(0)
            keep = rng.choice(ys.size, self.max_init_points, replace=False)
            ys = ys[keep]
            xs = xs[keep]

        z = depth_m[ys, xs]
        x = (xs.astype(np.float32) - self.cx) / self.fx * z
        y = (ys.astype(np.float32) - self.cy) / self.fy * z
        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1).astype(np.float32)
        pts_world = (self.c2w_poses[0] @ pts_cam.T).T[:, :3]

        rgb = np.array(Image.open(self.image_paths[0]).convert("RGB"), dtype=np.float32) / 255.0
        colors = rgb[ys, xs]
        normals = np.zeros_like(pts_world, dtype=np.float32)
        return pts_world.astype(np.float32), colors.astype(np.float32), normals

    def get_maxtime(self):
        return self.maxtime


def read_s3po_stereomis_scene(datadir, scene_info_cls, norm_fn, args=None):
    dataset_kwargs = {}
    if args is not None:
        dataset_kwargs["pose_source"] = getattr(args, "pose_source", "gt")
        dataset_kwargs["pose_path"] = getattr(args, "pose_path", "")
        dataset_kwargs["mask_mode"] = getattr(args, "mask_mode", "exclude_instrument")
    dataset = S3POStereoMISDataset(datadir=datadir, **dataset_kwargs)
    train_cam_infos = dataset.format_infos(split="train")
    test_cam_infos = dataset.format_infos(split="test")
    video_cam_infos = dataset.format_infos(split="video")
    nerf_normalization = norm_fn(train_cam_infos)
    xyz, rgb, normals = dataset.get_sparse_pts()
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals)

    return scene_info_cls(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        video_cameras=video_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=None,
        maxtime=dataset.get_maxtime(),
    )
