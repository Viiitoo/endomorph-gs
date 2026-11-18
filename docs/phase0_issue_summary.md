# Phase0 Issue Summary

This document summarizes the main problems found in phase0 contour/pose
validation on `stereo_seq_easy`.

## 1. STAR-Edge 3D Contours Are Not Aligned With 2D Contours

Observation:

- Projected STAR-Edge points mainly look like point-cloud boundaries,
  discontinuities, and noisy geometric artifacts.
- Many projected points do not correspond to meaningful tissue contours in the
  RGB image.
- Border artifacts are especially strong.

Conclusion:

- Directly applying STAR-Edge on dense MASt3R pointmaps does not produce the
  same contour type as the 2D `get_edge_local` extractor.
- STAR-Edge should not be used as the primary 3D contour source in its current
  form.

Current better direction:

- Use 2D contour pixels on the keyframe to sample MASt3R pointmap points.
- Clean the sampled 3D contour by border filtering, depth filtering, connected
  components, and outlier rejection.
- Treat STAR-Edge only as an optional auxiliary score until it is validated.

## 2. COLMAP Pose And MASt3R Pointmaps Lack A Shared Scale

Observation:

- MASt3R pointmaps should be projected with an effective camera, not raw
  StereoMIS intrinsics.
- Fitted MASt3R effective intrinsics are roughly:

```text
fx ~= 406
fy ~= 407
cx ~= 320
cy ~= 240
```

- Raw StereoMIS intrinsics are roughly:

```text
fx ~= 517
fy ~= 517
cx ~= 302
cy ~= 241
```

- COLMAP relative translation cannot be directly applied to MASt3R local
  pointmaps.
- For frame 20 -> 30:

```text
GT translation norm ~= 0.056 mm
COLMAP aligned translation norm ~= 1.03 mm
```

- For frame 0 -> 100, COLMAP global scale is closer to GT, but direct COLMAP
  translation still projects MASt3R contour points badly.

Conclusion:

- `K_eff != K_raw` can be handled with a camera/pointmap coordinate conversion.
- The bigger issue is that COLMAP/SfM poses and MASt3R local pointmaps do not
  share a reliable metric scale or coordinate frame.
- Monocular COLMAP does not recover metric scale from intrinsics alone.

Current better direction:

- Do not directly use full COLMAP translation in MASt3R local pointmap space.
- Use COLMAP mostly as a weak rotation/coarse trajectory prior.
- Estimate or constrain translation scale through MASt3R pairwise alignment,
  stereo/depth, backend render-map alignment, or another metric source.

## 3. Standalone DT Loss Does Not Guarantee Motion Toward GT

Observation:

- Distance-transform contour loss can reduce projected contour distance.
- However, lower DT loss often does not mean the pose is closer to GT.
- From identity on frame 0 -> 100:

```text
DT mean: 27.07 px -> 20.03 px
rotation error to GT: 13.75 deg -> 14.75 deg
translation error to GT: 0.0097 -> 0.1622
```

- From GT(0 -> 50) initialization toward target frame 100:

```text
rotation error improved: 5.34 deg -> 4.26 deg
translation error worsened: 0.0054 -> 0.1627
```

- From near-GT GT(0 -> 75) initialization:

```text
coarse pose was already near GT(0 -> 100)
refinement moved it away from GT
```

Conclusion:

- DT loss has useful contour-alignment signal, especially for local shape
  alignment.
- It is ambiguous as a standalone pose objective.
- It can trade incorrect translation/rotation for lower contour distance.

Current better direction:

- Use contour DT only as an auxiliary loss.
- Add strong pose priors, especially translation prior.
- Prefer local, small-step refinement rather than unconstrained pose search.
- Combine with optical flow, MASt3R correspondences, photometric/render losses,
  or backend map alignment.

## Phase0 Takeaway

The contour-refinement route is not invalid, but the current minimal version is
under-constrained.

The next implementation should not be:

```text
STAR-Edge pointmap edge -> DT-only SE3 optimization
```

Instead, the next version should be:

```text
keyframe 2D contour -> clean MASt3R 3D contour
coarse pose / motion prior -> small pose update
DT contour loss + strong pose prior + additional correspondence/render losses
```

The phase0 evidence suggests contour is useful as a supporting geometric cue,
not as the sole driver of pose estimation.
