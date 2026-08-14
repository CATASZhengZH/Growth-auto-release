# Third-Party Notices

This repository depends on third-party libraries and pretrained models. Their
licenses and terms apply independently of the final project license.

| Component | Role | Upstream license identified during release audit |
|---|---|---|
| Ultralytics 8.3.226 | Banana-Gpose model loading and segmentation inference | AGPL-3.0 or applicable Ultralytics enterprise terms |
| PyTorch / torchvision | Tensor computation and model runtime | BSD-style licenses |
| OpenCV | Image processing and visualization | Apache-2.0 |
| Hugging Face Transformers | GroundingDINO processor and model loading | Apache-2.0 |
| GroundingDINO-tiny | Banana fruit-region box prior | Apache-2.0 upstream repository |
| SAM/SAM2 | Fruit-region mask prior | Apache-2.0 upstream repository |
| NumPy / SciPy / scikit-learn | Numerical geometry and robust fitting | Permissive upstream licenses |

GroundingDINO and SAM/SAM2 model files are not redistributed in this release;
they are downloaded from their respective providers or supplied by the user.
The final project license must be approved by the authors and institutional
rights holders before publication. This notice is informational and is not
legal advice.
