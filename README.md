# Growth-auto

Official release workspace for the manuscript:

> **Growth-auto: A prior-guided geometric optimization framework for banana fruit-stalk posture estimation in complex orchard scenes**

## Overview

Growth-auto is a prior-guided geometric optimization framework for banana
fruit-stalk posture estimation. Banana-Gpose first segments the visible
fruit-stalk. GroundingDINO and SAM then provide a banana fruit-region prior,
which guides geometric axis optimization and signed posture-angle estimation.

## Pipeline

```text
RGB image
  -> Banana-Gpose
  -> fruit-stalk segmentation
  -> fruit-region prior
  -> prior-guided geometric optimization
  -> posture angle estimation
```

Banana-Gpose is the final paper model. It is based on YOLO26l-Seg with a
C2PSA-SimAM refinement block. Growth-auto is the geometric posture-estimation
method; GroundingDINO/SAM supplies the fruit-region prior and does not segment
the fruit-stalk or directly regress the posture angle.

## Installation

Python 3.12 is recommended. The default requirements reproduce the recorded
PyTorch 2.3.0 / CUDA 12.1 paper environment. Create a clean environment and
install the release dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a different CUDA runtime, install the corresponding PyTorch 2.3.0 wheel as
described by PyTorch before installing the remaining dependencies. The first
run downloads the public GroundingDINO-tiny and SAM2-tiny model files
when they are not already cached. See [docs/TESTING.md](docs/TESTING.md) for
CUDA and offline-use notes.

## Quick Start

From the repository root:

```bash
python inference/run_demo.py \
  --weights weights/Banana-Gpose-best.pt \
  --source demo/images \
  --output demo/results
```

Use CPU inference when CUDA is unavailable:

```bash
python inference/run_demo.py \
  --weights weights/Banana-Gpose-best.pt \
  --source demo/images \
  --output demo/results \
  --device cpu
```

## Outputs

For each input image, the demo writes:

- `*_mask.png`: binary Banana-Gpose fruit-stalk mask.
- `*_fruit_prior.png`: GroundingDINO/SAM fruit-region-prior overlay.
- `*_growth_auto.png`: prior-guided geometric optimization overlay.
- `*_result.json`: signed posture angle and traceable output paths.
- `*_visualization.png`: final combined visualization.
- `results.csv`: one summary row per image.

Angles are measured relative to the image vertical direction. Right inclination
is positive and left inclination is negative.

## Model Weights

The provided pretrained paper checkpoint is:

```text
weights/Banana-Gpose-best.pt
SHA-256: 169503cb8f172dd70ae7004c443b52a05debf7a0a4e71892bc00f3cf9dbc3491
```

Only the final `best.pt` checkpoint is included. Training checkpoints,
ablation weights, failed experiments, pruning artifacts, and reviewer-only
materials are excluded.

## Demo

The repository contains five representative examples drawn exclusively from
the held-out test split:

| Public file | Scene type | Demo angle |
|---|---|---:|
| `sample01_normal.jpg` | clearly visible fruit-stalk | +7.836 deg |
| `sample02_curved.jpg` | curved fruit-stalk | -12.664 deg |
| `sample03_occluded.jpg` | partially occluded / short visible fruit-stalk | -7.850 deg |
| `sample04_complex_background.jpg` | complex orchard background | -18.968 deg |
| `sample05_difficult.jpg` | difficult local-axis geometry | +10.812 deg |

| Normal | Curved | Occluded |
|---|---|---|
| ![Normal example](demo/results/sample01_normal_visualization.png) | ![Curved example](demo/results/sample02_curved_visualization.png) | ![Occluded example](demo/results/sample03_occluded_visualization.png) |

| Complex background | Difficult |
|---|---|
| ![Complex-background example](demo/results/sample04_complex_background_visualization.png) | ![Difficult example](demo/results/sample05_difficult_visualization.png) |

The files are included solely as representative reproduction examples. See
[docs/DEMO_SELECTION.md](docs/DEMO_SELECTION.md) for their test-split audit
trail and public filename mapping.

## Method Details

The frozen implementation and its parameter values are described in
[docs/METHOD.md](docs/METHOD.md). The release performs inference only and does
not contain training code or private annotations.

The checkpoint was trained with a YOLO26-compatible head and a C2PSA-SimAM
module. `inference/segmentation.py` registers those frozen model classes at
load time, so the checkpoint runs with the official Ultralytics 8.3.226 wheel
without patching the installed package.

## Dataset Availability

The complete banana orchard dataset and annotations are not publicly released
due to field-collected agricultural data ownership restrictions. They are
available from the corresponding author upon reasonable request, subject to
the applicable data-use and ownership conditions.

## License

The final repository license is pending author/institutional confirmation.
Ultralytics code and models are distributed under AGPL-3.0 unless an applicable
enterprise license is held. Do not publish this draft workspace until the
root `LICENSE` is replaced with the approved final license. See the external
release audit and [docs/LICENSE_REVIEW.md](docs/LICENSE_REVIEW.md) for details.

## Citation

Please update the author list, article identifier, DOI, and publication status
after editorial acceptance:

```bibtex
@article{growthauto2026,
  title   = {Growth-auto: A prior-guided geometric optimization framework for banana fruit-stalk posture estimation in complex orchard scenes},
  author  = {Author list to be confirmed},
  journal = {Plant Phenomics},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## Acknowledged Dependencies

This release uses Ultralytics, PyTorch, OpenCV, Hugging Face Transformers,
GroundingDINO, and SAM/SAM2. Their respective licenses and model terms remain
applicable. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
