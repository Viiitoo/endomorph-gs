# MASt3R Camera Intrinsics Alignment

This note records how we should handle camera intrinsics when using MASt3R
pointmaps for 3D contour extraction and pose refinement.

## Problem

MASt3R predicts a dense pointmap for the image frame it sees after resize/crop
preprocessing. The pointmap is useful, but it is not guaranteed to be exactly
consistent with our raw camera intrinsics.

Using raw `K` directly on a resized/cropped MASt3R frame mixes coordinate
systems:

```text
processed-frame pointmap + raw-image K = wrong projection
```

This is why we previously saw different fitted `K_eff` values when mixing:

```text
clean adapted images: 640x512
old recons RGB path: 640x480
```

## K Frame

`K_frame` is the intrinsics matrix in the pixel coordinate system of the image
frame used by MASt3R and downstream contour extraction.

For direct resize from `(W0,H0)` to `(W1,H1)`:

```text
fx' = fx * W1 / W0
fy' = fy * H1 / H0
cx' = cx * W1 / W0
cy' = cy * H1 / H0
```

If crop is applied after resize:

```text
cx' = cx * scale_x - crop_x
cy' = cy * scale_y - crop_y
```

The resulting matrix is:

```text
K_frame = [[fx', 0,   cx'],
           [0,   fy', cy'],
           [0,   0,   1  ]]
```

## Ray Alignment

Following the calibrated idea used in MASt3R-SLAM, we should use known
intrinsics to constrain MASt3R points to calibrated camera rays before using
them as geometry.

For every pixel `(u,v)` and MASt3R point `P=(X,Y,Z)`, keep the pointmap depth
`Z` and rebuild the point on the ray defined by `K_frame`:

```text
x_ray = (u - cx') / fx'
y_ray = (v - cy') / fy'

P_aligned = [x_ray * Z, y_ray * Z, Z]
```

After this:

```text
project(K_frame, P_aligned[u,v]) = (u,v)
```

So 3D contour points extracted from the aligned pointmap can be projected back
with `K_frame` without per-frame fitted `K_eff`.

## Intended Workflow

```text
raw RGB + K_raw
-> deterministic resize/crop
-> K_frame
-> MASt3R pointmap
-> constrain_pointmap_to_intrinsics(pointmap, K_frame)
-> STAR-Edge 3D contour extraction
-> 3D contour postprocess
-> project with K_frame
-> compare/refine against 2D contour
```

## Repository Utility

The reusable implementation is in:

```text
pose/camera_intrinsics.py
```

Main functions:

```python
scale_intrinsics(K, src_size, dst_size)
transform_intrinsics(K, ImageTransform(...))
constrain_pointmap_to_intrinsics(pointmap, K_frame)
fit_effective_intrinsics(pointmap)
```

`fit_effective_intrinsics()` is diagnostic only. It helps verify whether a
pointmap is internally close to a pinhole model, but it should not be the final
system path.

## Current Recommendation

For phase0 experiments, use the copied old-recons RGB data as canonical input:

```text
/workspace/data/old_recons_stereo_seq_easy_raw/rgb
```

It is `640x480`, and the fitted effective intrinsics are close to:

```text
fx ~= 407
fy ~= 407
cx ~= 321
cy ~= 240
```

Next, we should replace ad-hoc `K_eff` fitting in diagnostic scripts with
explicit `K_frame` construction and `constrain_pointmap_to_intrinsics()`.
