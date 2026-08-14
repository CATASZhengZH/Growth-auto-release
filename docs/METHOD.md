# Growth-auto Method

## Scope

Growth-auto estimates a signed banana fruit-stalk posture angle from one RGB
image. It is a deterministic, prior-guided geometric optimization method rather
than a learned angle regressor.

## Final Pipeline

1. Banana-Gpose predicts fruit-stalk instance masks at confidence threshold
   `0.25`.
2. If multiple masks are present, the polygon with the largest area is selected.
3. GroundingDINO-tiny detects banana fruit regions using the prompt
   `banana fruit. banana finger. banana bunch.` at threshold `0.18`.
4. SAM2-tiny converts detected boxes into fruit-region polygons.
5. The selected fruit region provides a fruit-to-stalk direction prior.
6. Growth-auto computes a mask-only OBB-spine candidate and a prior-guided
   quadratic-Bezier candidate family.
7. Bezier candidates are rasterized as swept bands and scored using fruit-stalk
   mask coverage, band precision, terminal-tangent alignment, and a small
   curvature penalty.
8. Growth-auto selects the prior-guided axis for a large fruit prior, an
   irregular/short stalk mask, or a sufficiently different candidate angle;
   otherwise it retains the stable OBB-spine candidate.
9. The terminal tangent of the selected curve defines the local posture angle.

## Frozen Parameters

| Parameter | Value |
|---|---:|
| Banana-Gpose confidence | 0.25 |
| GroundingDINO box/text threshold | 0.18 |
| Fruit prompt | `banana fruit. banana finger. banana bunch.` |
| Axis extension ratio | 0.30 |
| Allow large SAM fruit prior | yes |
| Bezier samples per candidate | 96 |
| Prior-axis selection aspect threshold | 2.2 |
| Prior/OBB angular-difference threshold | 14 deg |

The complete candidate grid and scoring implementation are preserved in
`inference/growth_auto.py`, migrated from the frozen paper evaluation script.

## Angle Convention

The image vertical direction is the zero-degree reference. Positive values
indicate right inclination and negative values indicate left inclination. The
reported angle is computed from the selected axis after identifying its top and
bottom relation in image coordinates.

## Fallbacks

- No Banana-Gpose mask: status `no_detection`; no angle is reported.
- No valid fruit prior: use the mask-only OBB-spine candidate.
- Prior-guided candidate failure: use the OBB-spine candidate.
- OBB-spine failure with a valid prior axis: use the prior-guided axis.

## Reproducibility Boundary

The public repository contains the final Banana-Gpose checkpoint, inference
code, fixed configuration, test examples, and generated demo outputs. It does
not contain training data, full orchard imagery, private annotations, ablation
checkpoints, failed runs, or reviewer-only materials.
