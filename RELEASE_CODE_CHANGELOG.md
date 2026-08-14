# Release Code Changelog

## 2026-08-14

- Created an isolated GitHub release workspace without modifying the original
  experiment directories.
- Copied the final paper checkpoint as `weights/Banana-Gpose-best.pt`.
- Migrated the frozen `estimate_stalk_angle.py` implementation to
  `inference/growth_auto.py` without algorithm changes.
- Added `segmentation.py` to load Banana-Gpose, register the frozen YOLO26 and
  C2PSA-SimAM checkpoint classes, and preserve largest-area instance selection.
- Added `posture_estimation.py` to orchestrate the frozen segmentation, prior,
  geometric optimization, signed-angle calculation, and output export.
- Added `run_demo.py`, fixed configuration, environment files, method notes,
  and testing instructions.
- Pinned a mutually compatible NumPy 2.2.6 / SciPy 1.15.3 / scikit-learn 1.7.2
  stack after the host's newer NumPy 2.3.5 environment failed before
  GroundingDINO model loading.
- Pinned PyTorch 2.3.0 and torchvision 0.18.0 after a clean resolver selected an
  unvalidated newer runtime and GPU testing exposed a CUDA 13 NVRTC mismatch.
- Added five held-out test examples and generated release-demo outputs using
  the final Banana-Gpose checkpoint.
- Verified all five examples with the official Ultralytics 8.3.226 wheel. The
  compatibility layer produced the same masks and numerical posture results as
  the local training-environment package.
- Added automatic CPU fallback when no CUDA device is available and no device
  override is supplied.
- Set the recorded `TORCH_CUDNN_V8_API_DISABLED=1` compatibility option before
  importing PyTorch; this restored successful GPU inference and the frozen demo
  angles in the recorded CUDA 12.1 runtime.
- Added demo provenance and third-party dependency notices.
- Re-exported the public demo visualizations from an isolated environment using
  the exact pinned OpenCV 4.12.0.88 stack; `pip check` passed in that environment.
- Excluded training data, private annotations, reviewer-only files, training
  checkpoints, ablation weights, failed experiments, and intermediate tables.

No training code, model retraining, method redesign, or paper-result changes
were introduced by this release preparation.
