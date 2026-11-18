import glob
import importlib.util
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pose.contour_3d import Contour3D

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


@dataclass
class StarEdgeConfig:
    star_edge_root: str
    model_path: str = ""
    threshold: float = 0.5
    bw: int = 10
    kk: int = 26
    sample_num: int = 40
    max_points: int = 50000
    use_cuda_mlp: bool = False
    use_original_refine: bool = True
    refine_mu: float = 0.1
    mlp_batch_size: int = 2048
    postprocess: bool = True
    sor_k: int = 20
    sor_alpha: float = 1.8
    curvature_k: int = 24
    curvature_max_percentile: float = 99.0
    min_neighbor_support: float = 0.35
    bbox_trim_quantile: float = 0.0
    output_max_points: int = 4000


class StarEdgeExtractor:
    """Real STAR-Edge extractor: LocalSH descriptors + DescClassifier MLP."""

    def __init__(self, config: StarEdgeConfig):
        self.config = config
        self.root = Path(config.star_edge_root)
        self.model_path = Path(config.model_path) if config.model_path else self.root / "model" / "best.ckpt"
        self._localsh = None
        self._net = None
        self._device = None

    def _add_import_paths(self):
        paths = [
            self.root,
        ]
        for path in paths:
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))

    def _find_localsh_extension(self):
        suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
        candidates = []
        if suffix:
            candidates.extend(self.root.glob(f"LocalSH/build_cmake*/LocalSH{suffix}"))
            candidates.extend(self.root.glob(f"LocalSH/**/LocalSH{suffix}"))
            candidates.extend(self.root.glob(f"pre_process/LocalSH{suffix}"))
        candidates.extend(self.root.glob("LocalSH/build_cmake*/LocalSH*.so"))
        candidates.extend(self.root.glob("LocalSH/**/LocalSH*.so"))
        candidates.extend(self.root.glob("pre_process/LocalSH*.so"))
        exact = [p for p in candidates if suffix and p.name.endswith(suffix)]
        if exact:
            return exact[0]
        found = ", ".join(str(p) for p in candidates[:5])
        raise RuntimeError(
            "No STAR-Edge LocalSH extension matches this Python ABI. "
            f"Expected suffix {suffix!r}. Found: {found or 'none'}"
        )

    def _load_localsh(self):
        if self._localsh is not None:
            return self._localsh
        self._add_import_paths()
        extension_path = self._find_localsh_extension()
        try:
            spec = importlib.util.spec_from_file_location("LocalSH", extension_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not build import spec for {extension_path}")
            LocalSH = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(LocalSH)
        except Exception as exc:
            raise RuntimeError(
                "STAR-Edge LocalSH is unavailable. This must be the compiled "
                "LocalSH module matching the current Python ABI; check whether "
                "the .so is built for this container's Python version."
            ) from exc
        if not hasattr(LocalSH, "LocalSHFeature"):
            raise RuntimeError("STAR-Edge LocalSH module does not expose LocalSHFeature.")
        self._localsh = LocalSH
        return self._localsh

    def _load_classifier(self):
        if self._net is not None:
            return self._net, self._device
        if not self.model_path.exists():
            raise FileNotFoundError(f"STAR-Edge checkpoint not found: {self.model_path}")
        self._add_import_paths()
        try:
            from net import DescClassifier  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"Could not import STAR-Edge DescClassifier from {self.root / 'net.py'}") from exc

        device = torch.device("cuda" if self.config.use_cuda_mlp and torch.cuda.is_available() else "cpu")
        net = DescClassifier()
        ckpt = torch.load(str(self.model_path), map_location="cpu")
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(state, strict=False)
        net.to(device=device)
        net.eval()
        self._net = net
        self._device = device
        return net, device

    def _run_localsh(self, points):
        localsh = self._load_localsh()
        result = localsh.LocalSHFeature.ComLSHF_knn_upsample(
            np.asarray(points[:, :3], dtype=np.float64),
            int(self.config.bw),
            int(self.config.kk),
            int(self.config.sample_num),
        )
        if "Descs" not in result:
            raise RuntimeError("LocalSH result does not contain 'Descs'.")
        return result

    def _classify_desc(self, desc):
        desc = np.asarray(desc, dtype=np.float32)
        net, device = self._load_classifier()
        probs = []
        batch_size = int(max(1, self.config.mlp_batch_size))
        with torch.no_grad():
            for start in range(0, len(desc), batch_size):
                batch = torch.from_numpy(desc[start : start + batch_size]).to(device=device)
                prob = net(batch.float()).detach().cpu().numpy()
                probs.append(np.asarray(prob, dtype=np.float32).reshape(-1))
        if not probs:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(probs, axis=0).astype(np.float32)

    def _extract_subset(self, points):
        result = self._run_localsh(points)
        desc = np.asarray(result["Descs"], dtype=np.float32)
        return self._classify_desc(desc)

    def _empty_contour(self):
        empty = np.empty((0,), dtype=np.float32)
        return Contour3D(points=np.empty((0, 3), dtype=np.float32), weights=empty, scores=empty)

    def _filter_by_bbox_quantile(self, points):
        q = float(self.config.bbox_trim_quantile)
        if q <= 0.0 or len(points) == 0:
            return np.ones(len(points), dtype=bool)
        q = min(q, 0.49)
        lo = np.quantile(points, q, axis=0)
        hi = np.quantile(points, 1.0 - q, axis=0)
        return np.all((points >= lo) & (points <= hi), axis=1)

    def _sor_mask(self, points):
        if cKDTree is None:
            return np.ones(len(points), dtype=bool)
        k = int(self.config.sor_k)
        if k <= 0 or len(points) <= k + 1:
            return np.ones(len(points), dtype=bool)
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=min(k + 1, len(points)))
        mean_dist = dists[:, 1:].mean(axis=1)
        limit = mean_dist.mean() + float(self.config.sor_alpha) * (mean_dist.std() + 1e-8)
        return mean_dist <= limit

    def _curvature_mask(self, points):
        if cKDTree is None:
            return np.ones(len(points), dtype=bool)
        k = int(self.config.curvature_k)
        if k <= 3 or len(points) <= k + 1:
            return np.ones(len(points), dtype=bool)
        tree = cKDTree(points)
        _, indices = tree.query(points, k=min(k + 1, len(points)))
        curvature = np.zeros(len(points), dtype=np.float32)
        for i, nn in enumerate(indices):
            nbrs = points[np.asarray(nn[1:], dtype=np.int64)]
            centered = nbrs - nbrs.mean(axis=0, keepdims=True)
            cov = centered.T @ centered / max(1, len(nbrs) - 1)
            eigvals = np.maximum(np.linalg.eigvalsh(cov), 0.0)
            curvature[i] = float(eigvals[0] / (eigvals.sum() + 1e-12))

        percentile = float(self.config.curvature_max_percentile)
        if percentile <= 0.0 or percentile >= 100.0:
            base = np.ones(len(points), dtype=bool)
        else:
            threshold = np.percentile(curvature, percentile)
            base = curvature <= threshold

        min_support = float(self.config.min_neighbor_support)
        if min_support <= 0.0:
            return base
        support = np.zeros(len(points), dtype=np.float32)
        for i, nn in enumerate(indices):
            nbr_idx = np.asarray(nn[1:], dtype=np.int64)
            support[i] = float(base[nbr_idx].mean()) if len(nbr_idx) else 0.0
        return base & (support >= min_support)

    def _limit_points(self, points, scores):
        max_points = int(self.config.output_max_points)
        if max_points <= 0 or len(points) <= max_points:
            return np.arange(len(points), dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(scores) != len(points):
            scores = np.ones(len(points), dtype=np.float32)
        order = np.argsort(scores)[-max_points:]
        return np.sort(order.astype(np.int64))

    def _postprocess_contour(self, contour):
        if not self.config.postprocess:
            return contour
        points = np.asarray(contour.points, dtype=np.float32).reshape(-1, 3)
        if len(points) == 0:
            return contour
        scores = np.asarray(contour.scores, dtype=np.float32).reshape(-1)
        weights = np.asarray(contour.weights, dtype=np.float32).reshape(-1)
        if len(scores) != len(points):
            scores = np.ones(len(points), dtype=np.float32)
        if len(weights) != len(points):
            weights = np.clip(scores, 0.05, 1.0).astype(np.float32)

        keep = np.isfinite(points).all(axis=1)
        keep &= np.linalg.norm(points, axis=1) > 1e-8
        keep &= self._filter_by_bbox_quantile(points)
        filtered = points[keep]
        if len(filtered) == 0:
            return self._empty_contour()

        keep_local = self._sor_mask(filtered)
        filtered2 = filtered[keep_local]
        if len(filtered2) == 0:
            return self._empty_contour()

        keep_curv = self._curvature_mask(filtered2)
        filtered3 = filtered2[keep_curv]
        if len(filtered3) == 0:
            return self._empty_contour()

        idx0 = np.flatnonzero(keep)
        idx1 = idx0[keep_local]
        idx2 = idx1[keep_curv]
        idx3_local = self._limit_points(filtered3, scores[idx2])
        idx3 = idx2[idx3_local]
        out_points = points[idx3].astype(np.float32)
        out_scores = scores[idx3].astype(np.float32)
        out_weights = weights[idx3].astype(np.float32)
        return Contour3D(points=out_points, weights=out_weights, scores=out_scores)

    def extract_original_refined(self, points_world):
        points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        if len(points) == 0:
            return self._empty_contour()

        localsh = self._load_localsh()
        result = self._run_localsh(points)
        desc = np.asarray(result["Descs"], dtype=np.float32)
        normals = np.asarray(result["normals"], dtype=np.float64)
        neighboor = result["neighboor"]
        scores = np.clip(self._classify_desc(desc), 0.0, 1.0)
        flag = (scores >= float(self.config.threshold)).astype(np.int32)

        refined = localsh.NormalRefine.EdgePointRefine(
            flag,
            neighboor,
            np.asarray(points[:, :3], dtype=np.float64),
            normals,
            float(self.config.refine_mu),
        )
        refined = np.asarray(refined, dtype=np.float32).reshape(-1, 3)
        keep = np.isfinite(refined).all(axis=1) & (np.linalg.norm(refined, axis=1) > 1e-8)
        refined = refined[keep]
        if len(refined) == 0:
            return self._empty_contour()
        refined_scores = np.ones(len(refined), dtype=np.float32)
        contour = Contour3D(points=refined.astype(np.float32), weights=refined_scores.copy(), scores=refined_scores)
        return self._postprocess_contour(contour)

    def extract_fast_propagated(self, points_world):
        points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        if len(points) == 0:
            return self._empty_contour()

        max_points = int(self.config.max_points)
        if len(points) > max_points:
            rng = np.random.default_rng(0)
            subset_idx = rng.choice(len(points), size=max_points, replace=False)
            subset = points[subset_idx]
            subset_scores = self._extract_subset(subset)
            if cKDTree is None:
                scores = np.zeros(len(points), dtype=np.float32)
                scores[subset_idx] = subset_scores
                fill = float(subset_scores.mean()) if len(subset_scores) else 0.0
                scores[scores == 0] = fill
            else:
                tree = cKDTree(subset)
                _, nn = tree.query(points, k=1)
                scores = subset_scores[np.asarray(nn, dtype=np.int64)].astype(np.float32)
        else:
            scores = self._extract_subset(points)

        scores = np.clip(scores.astype(np.float32), 0.0, 1.0)
        mask = scores >= float(self.config.threshold)
        edge_points = points[mask]
        edge_scores = scores[mask]
        if len(edge_points) == 0:
            return self._empty_contour()
        weights = np.clip(edge_scores, 0.05, 1.0).astype(np.float32)
        contour = Contour3D(points=edge_points.astype(np.float32), weights=weights, scores=edge_scores.astype(np.float32))
        return self._postprocess_contour(contour)

    def extract(self, points_world):
        if self.config.use_original_refine:
            return self.extract_original_refined(points_world)
        return self.extract_fast_propagated(points_world)
