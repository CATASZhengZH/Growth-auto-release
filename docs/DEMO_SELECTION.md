# Demo Selection

The five public examples were copied from the held-out test split. No training
or validation image is included. Only RGB images are released; split manifests,
segmentation labels, and posture annotations remain private.

| Public filename | Original test filename | Scene type | Selection reason |
|---|---|---|---|
| `sample01_normal.jpg` | `IMG_20260506_145119.jpg` | normal | Clearly visible stalk and stable geometry. |
| `sample02_curved.jpg` | `IMG_20260506_152838.jpg` | curved | Curvature challenges a single global line approximation. |
| `sample03_occluded.jpg` | `IMG_20260506_152156.jpg` | partially occluded | Short visible stalk with orchard occlusion. |
| `sample04_complex_background.jpg` | `IMG_20260506_144901.jpg` | complex background | Dense foliage and trunk structure surround the target. |
| `sample05_difficult.jpg` | `IMG_20260506_153117.jpg` | difficult | Local-axis geometry is visually challenging. |

The release copies were verified byte-for-byte against the corresponding test
images during release preparation. Redistribution of these five examples still
requires final author and institutional approval before public publication.
