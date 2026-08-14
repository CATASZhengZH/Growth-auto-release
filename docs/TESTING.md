# Testing Instructions

## Validated Reference Environment

The paper training/evaluation records report:

| Component | Version / setting |
|---|---|
| OS | Linux x86_64 |
| Python | 3.12.2 |
| PyTorch | 2.3.0+cu121 |
| CUDA used by PyTorch | 12.1 |
| Ultralytics | 8.3.226 |
| OpenCV | 4.12.0.88 |
| Transformers | 4.57.6 |
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| Inference image size | 640 |

The release pins NumPy to `2.2.6`, below OpenCV 4.12's `<2.3` upper bound, and
pairs it with SciPy 1.15.3 and scikit-learn 1.7.2 builds that support NumPy 2.

The release checkpoint contains frozen YOLO26 and C2PSA-SimAM classes that are
not shipped by the official Ultralytics wheel. The compatibility registration
in `inference/segmentation.py` supplies the exact inference definitions before
checkpoint loading; it does not patch site-packages or alter checkpoint values.

PyTorch wheels are CUDA-specific. Install the wheel appropriate for your
driver/runtime before installing `requirements.txt` when the default wheel is
not suitable. The release pins PyTorch 2.3.0 and torchvision 0.18.0 to match the
recorded paper runtime. `run_demo.py` also sets
`TORCH_CUDNN_V8_API_DISABLED=1` before importing PyTorch, matching the recorded
training launcher and avoiding a cuDNN v8 execution-plan failure observed with
this checkpoint.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Conda users may instead run:

```bash
conda env create -f environment.yml
conda activate growth-auto
```

## Run The Five Examples

```bash
python inference/run_demo.py \
  --weights weights/Banana-Gpose-best.pt \
  --source demo/images \
  --output demo/results
```

The first run may download:

- `IDEA-Research/grounding-dino-tiny` through Hugging Face Transformers.
- `sam2.1_t.pt` through Ultralytics.

For offline testing, download/cache those third-party model files beforehand
and provide the local SAM path:

```bash
python inference/run_demo.py \
  --weights weights/Banana-Gpose-best.pt \
  --source demo/images \
  --output demo/results \
  --sam-model /path/to/sam2.1_t.pt
```

## CPU Test

```bash
python inference/run_demo.py \
  --weights weights/Banana-Gpose-best.pt \
  --source demo/images/sample01_normal.jpg \
  --output demo/results_cpu \
  --device cpu
```

CPU execution is supported but substantially slower because Banana-Gpose,
GroundingDINO, and SAM all run in one pipeline.

When `--device` is omitted and CUDA is unavailable, `run_demo.py` automatically
uses CPU. Passing `--device 0` or `--device cpu` overrides that selection.

## Expected Outputs

For each successful image, verify that these files exist:

```text
*_mask.png
*_fruit_prior.png
*_growth_auto.png
*_result.json
*_visualization.png
```

The output directory must also contain `results.csv`. A successful row has:

- `status=ok`
- `fruit_prior=yes` for the supplied examples
- a finite `signed_angle_from_vertical_deg`
- paths to all four visual products

## Integrity Check

Verify the paper checkpoint before inference:

```bash
sha256sum weights/Banana-Gpose-best.pt
```

Expected SHA-256:

```text
169503cb8f172dd70ae7004c443b52a05debf7a0a4e71892bc00f3cf9dbc3491
```

## Known External Requirements

- Internet access is required on the first run unless GroundingDINO and SAM2
  are already cached.
- CUDA is optional, but a correctly matched PyTorch/CUDA installation is
  required for GPU execution.
- The final repository license must be approved before public publication.

## Release Validation

The five supplied examples were run successfully with the official Ultralytics
8.3.226 wheel and the recorded PyTorch 2.3.0 / torchvision 0.18.0 / CUDA 12.1
runtime. The release checkpoint and cuDNN compatibility setting reproduced the
frozen demo angles. CPU fallback was also verified, but tiny mask-boundary
differences can occur because CPU and CUDA convolution paths are not numerically
identical for pixels close to the mask threshold.
