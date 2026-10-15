# Self-Diagnosing Incremental Structure-from-Motion

A final project for the 3D Computer Vision course at the Hebrew University of Jerusalem. The pipeline reconstructs sparse 3D structure and camera poses from calibrated images, and records why reconstruction completes, stops partially, or fails.

The initialization policy considers both two-view geometry and whether the seed tracks can support registration of additional cameras. Reconstruction coverage is reported alongside residual error so that a small partial reconstruction is not mistaken for a better complete result.

## Implementation boundary

Project code implements Essential-matrix estimation and RANSAC, conflict-aware tracks, seed ranking, triangulation and validation, reconstruction policy, bundle-adjustment problem construction and gauge handling, diagnostics, and evaluation. OpenCV provides SIFT, matching primitives, PnP/refinement and other geometry helpers. SciPy provides nonlinear least squares; NumPy provides numerical operations.

```text
images + shared K
  -> SIFT / pair matching
  -> geometric verification / feature tracks
  -> expansion-aware two-view seed
  -> PnP registration / triangulation / bundle adjustment
  -> sparse reconstruction / evaluation / termination diagnostics
```

| Code | Responsibility |
| --- | --- |
| `main.py` | Dataset selection and run options |
| `src/sfm.py` | Public `run_sfm()` boundary |
| `src/pipeline/` | Verification, initialization, registration and finalization |
| `src/geometry/`, `src/triangulation.py` | Essential geometry, coordinate operations and triangulation |
| `src/tracks.py`, `src/initialization.py` | Track construction and expandable seed selection |
| `src/bundle_adjustment.py` | Residual construction, gauge constraints and SciPy solve |
| `src/validation/`, `src/evaluation.py` | Diagnostics and aligned pose/reprojection evaluation |

Poses map world coordinates to camera coordinates: `X_camera = R @ X_world + t`. Camera center is `-R.T @ t`.

## Setup and data

Requires Python 3.13+ and uv.

```bash
git clone https://github.com/ronketer/self-diagnosing-sfm.git
cd self-diagnosing-sfm
uv sync --locked
```

Datasets are not included. Obtain Middlebury TempleRing or TempleSparseRing from the [Middlebury multi-view stereo dataset](https://vision.middlebury.edu/mview/data/) under its terms, including the scene images and camera calibration/pose files expected by `src/dataset.py`. Place each scene directly under its named directory:

```text
data/templeRing/
data/templeSparseRing/
```

The loader reads the scene's `*_par.txt` camera file and named images; see its parser for the exact layout. The optional `data/truck/` path expects images and a prepared COLMAP reference/calibration model; it is not required for the Temple demonstration.

```bash
uv run python main.py --dataset templeRing --no-cache
uv run python main.py --dataset templeSparseRing --no-cache
```

## Reported experiments

The previous project README records these runs. Their machine-readable output files, environment/hardware details and complete run manifests are not committed, so treat the numbers as reported observations until independently reproduced for the current revision.

| Dataset / configuration | Cameras | Landmarks | Reprojection RMS | Status |
| --- | ---: | ---: | ---: | --- |
| TempleRing / full pipeline | 47/47 | 14,471 | 0.391 px | SUCCESS |
| TempleSparseRing / full pipeline | 4/16 | 655 | 0.238 px | PARTIAL |
| TempleRing / no BA | 47/47 | 14,684 | 8.591 px | Reported ablation |
| TempleRing / no geometric verification | 47/47 | 11,816 | 0.424 px | Reported ablation |

The sparse case stops with `NO_2D3D_BRIDGE`. Its lower RMS describes the reconstructed subset and does not imply better overall coverage. These observations do not establish performance on general scenes or robot perception systems.

```bash
uv run python main.py --dataset templeRing --no-cache --skip-ba --output-label no_ba
uv run python main.py --dataset templeRing --no-cache --skip-geometric-verification --output-label no_geom
```

Runs write under `output/<dataset>/` or the labeled directory. Inspect `reconstruction.json`, `diagnostics.json` and, when reference poses exist, `metrics.json` together. Plots and point-cloud exports provide visual checks; diagnostic status and camera coverage remain necessary.

## Tests and validation

```bash
uv run python -m tests.unit.test_initialization
uv run python -m tests.synthetic.run_all
uv run python tests/integration/test_pipeline.py
```

Synthetic checks cover geometry, tracks, pose recovery, bundle-adjustment gauge constraints and degeneracies. The integration command needs TempleSparseRing. In the current runner, absent data prints a skip message and returns; the script then prints an `OK` line. That is not a completed dataset validation. Confirm that reconstruction actually ran before citing the integration result.

No full-dataset result is implied by the existence of test functions or a successful import. Record the revision, dependency versions, inputs, configuration, metrics and diagnostic status when repeating the experiments.

## Limitations

- Known shared intrinsics, pinhole/pre-undistorted images, a static scene and sufficient overlap are assumed.
- Reconstruction is sparse. Dense reconstruction, online operation and robotics deployment are not established.
- Pair matching is quadratic in image count; repeated global bundle adjustment is costly for larger scenes.
- Sparse or disconnected tracks can prevent camera registration despite a geometrically valid seed.
- Numerical results depend on data, feature matching, configuration and dependency versions. Report coverage and status beside error metrics.

This is course-derived work. Course scaffolding, external datasets and third-party libraries retain their own attribution and terms.
