# MASt3R Effective Camera

MASt3R pointmaps should not be projected with the raw StereoMIS calibration directly. The model predicts `pts3d` in its own effective camera coordinate system after DUSt3R/MASt3R image preprocessing.

For `stereo_seq_easy`, fitting `u = fx * X/Z + cx`, `v = fy * Y/Z + cy` on 20 sampled pointmaps gives a stable effective camera:

```text
fx ~= 406.08
fy ~= 407.30
cx ~= 320.29
cy ~= 239.65
median reprojection residual ~= 0.97 px
p90 reprojection residual ~= 2.10 px
```

Raw StereoMIS calibration is about `fx=516.95, fy=516.86, cx=302.29, cy=241.49`, which explains the previous 50-70 px systematic projection offset.

For contour visualization/refinement, project MASt3R/STAR-Edge 3D points with per-frame or averaged `K_eff`, then map to the working image coordinates. Use raw dataset K only after explicitly converting the pointmap/geometry into the raw camera coordinate system.
