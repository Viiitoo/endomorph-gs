# S3PO StereoMIS Easy Experiment Metrics

Date: 2026-06-07
Branch: `phase0`
Dataset: prepared StereoMIS `stereo_seq_easy/train` sequence
Container: `eh_surgs_clean`

## Main Run

| Setting | Value |
| --- | --- |
| Experiment | `phase0/s3po_native_3000iter_mmfix_mask` |
| Iterations | 3000 |
| GPU | GPU 2 |
| Scale fix | depth in mm, pose translation x1000 |
| Mask mode | `exclude_instrument` |
| Initial points | 50000 |
| Final points | 54184 |
| Training time | 115.42 s |
| Train speed | ~26.0 iter/s overall, ~29.4 iter/s at end |
| Render/eval wall time | ~33.8 s |
| Output | `output/phase0/s3po_native_3000iter_mmfix_mask` |

## Test Metrics

| PSNR | SSIM | LPIPS |
| ---: | ---: | ---: |
| 34.9203 | 0.9074 | 0.1624 |

## Short-Run Comparison

| Experiment | Iterations | Mask mode | PSNR | SSIM | LPIPS |
| --- | ---: | --- | ---: | ---: | ---: |
| `s3po_native_300iter_mmfix_mask` | 300 | `exclude_instrument` | 23.3896 | 0.8320 | 0.3593 |
| `s3po_native_300iter_mmfix_depthmask` | 300 | `depth_only` | 21.8509 | 0.7517 | 0.6164 |
| `s3po_native_3000iter_mmfix_mask` | 3000 | `exclude_instrument` | 34.9203 | 0.9074 | 0.1624 |

## Raw COLMAP + GT Sim3 Pose

| Setting | Value |
| --- | --- |
| Experiment | `phase0/s3po_colmap_aligned_3000` |
| Config | `arguments/s3po/stereo_seq_easy_colmap_aligned.py` |
| Pose source | COLMAP sparse reconstruction, GT Sim3 aligned |
| COLMAP registered frames | 78 / 200 |
| COLMAP alignment error | mean 1.49 mm, median 1.42 mm, max 4.37 mm |
| Filled frames | unregistered frames use nearest registered pose |
| Iterations | 3000 |
| Initial / final points | 50000 / 53130 |
| Training time | 114.83 s |
| Output | `output/phase0/s3po_colmap_aligned_3000` |

| PSNR | SSIM | LPIPS |
| ---: | ---: | ---: |
| 35.4949 | 0.9161 | 0.1378 |

Note: `render.py` still fails when writing mp4 because `imageio-ffmpeg` is missing, but PNG rendering completes before that and `metrics.py` runs normally.
