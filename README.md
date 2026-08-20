# Robust and Self-Diagnosing Incremental Structure-from-Motion

A calibrated incremental Structure-from-Motion (SfM) pipeline for recovering
camera poses and sparse 3D geometry from overlapping images.

The project focuses not only on reconstructing a scene, but also on diagnosing
whether reconstruction is complete, partial, or geometrically blocked. The
system reports reconstruction status, termination reasons, camera-registration
state, view-graph connectivity, triangulation rejection causes, and PnP support.

## Pipeline

```mermaid
flowchart TD
    A[Input images + shared calibration K]
    B[SIFT features + descriptor matching]
    C[Essential-matrix geometric verification]
    D[Conflict-aware feature tracks]
    E[Expansion-aware two-view initialization]
    F[Initial 3D landmarks]
    G[Incremental PnP camera registration]
    H[Triangulate newly visible landmarks]
    I[Periodic bundle adjustment]
    J{Another camera can register?}
    K[Final BA + landmark pruning]
    L[Evaluation + diagnostics]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J -- Yes --> G
    J -- No --> K
    K --> L
```

The final reconstruction is classified as `SUCCESS`, `PARTIAL`, or `FAILED`
and receives an explicit termination reason such as `COMPLETED`,
`NO_2D3D_BRIDGE`, or `NO_REGISTRABLE_CAMERA`.

## Why This Project Is Interesting

A geometrically strong two-view pair is not automatically a good seed for
incremental SfM. A pair may have strong parallax and many inliers but still be
poorly connected to the rest of the image set.

The initializer therefore considers both:

- **geometric quality** — Essential-matrix support, cheirality, parallax,
  reprojection quality, and homography degeneracy;
- **expansion support** — whether the initial tracks can provide enough
  2D↔3D correspondences to register additional cameras.

The project also treats reconstruction coverage as a first-class metric. A
small partial reconstruction can have a lower reprojection RMS than a complete
reconstruction, so RMS alone is not treated as evidence of success.

## Representative Results

| Dataset | Registered cameras | Landmarks | Final RMS | Status |
|---|---:|---:|---:|---|
| Middlebury TempleRing | 47 / 47 | 13,811 | 0.383 px | `SUCCESS` |
| Middlebury TempleSparseRing | 4 / 16 | 655 | 0.238 px | `PARTIAL` |

TempleSparseRing is intentionally useful as a failure-analysis case: despite
its lower residual, most cameras are missing and the reconstruction terminates
with `NO_2D3D_BRIDGE`.

## Architecture

The code separates high-level SfM sequencing from geometry, numerical
optimization, typed reconstruction state, and experiment support.

```text
main.py
  │
  ▼
src/sfm.py                     public API: run_sfm()
  │
  ▼
src/pipeline/orchestration.py  high-level stage sequencing
  │
  ├── verification.py         pairwise geometric verification
  ├── initialization_stage.py initial cameras + initial map
  ├── registration.py         incremental PnP registration
  └── finalization.py         final optimization + diagnostics
          │
          ├── reconstruction_ops.py
          ├── bundle.py
          └── diagnostics.py
```

Core numerical and domain modules:

| Module | Responsibility |
|---|---|
| `src/models.py` | Canonical typed cameras, observations, tracks, landmarks, reconstruction, and state |
| `src/geometry/essential.py` | Essential estimation, Sampson error, RANSAC, decomposition, and pose selection |
| `src/geometry/projection.py` | Projection, normalization, camera centers, depth, and parallax geometry |
| `src/geometry/pose.py` | Pose/alignment utilities |
| `src/initialization.py` | Two-view evaluation and expansion-aware initial-pair ranking |
| `src/tracks.py` | Conflict-aware multi-view feature-track construction |
| `src/triangulation.py` | Two-view / multi-view DLT and geometric validation |
| `src/pnp.py` | Production PnP registration and camera-quality gates |
| `src/bundle_adjustment.py` | Joint nonlinear optimization of cameras and landmarks |
| `src/validation/` | Degeneracy checks and structured reconstruction diagnostics |
| `src/evaluation.py` | Sim(3) alignment and quantitative pose/reprojection evaluation |
| `src/visualization.py` | Reconstruction plots, diagnostic figures, and PLY export |
| `src/colmap_io.py` | Optional COLMAP binary-model reader for external calibrated scenes |
| `src/colmap_export.py` | Optional export to COLMAP sparse text format |

### Coordinate Convention

All camera poses follow the world-to-camera convention:

```text
X_camera = R @ X_world + t
C = -R.T @ t
x ~ K(RX + t)
```

where `C` is the camera center in world coordinates.

### Reconstruction State

`ReconstructionState` is the canonical state passed through the incremental
pipeline.

```text
ReconstructionState
├── Reconstruction
│   ├── CameraIntrinsics
│   ├── RegisteredCamera
│   └── Landmark
├── TrackManager
├── keypoints
├── SfMConfig
├── diagnostics
└── runtime statistics
```

Numerical solvers may temporarily operate on NumPy arrays, but optimized values
are applied back to the typed reconstruction model rather than maintaining a
second mutable reconstruction representation.

## Implementation Boundaries

The project deliberately combines custom geometric reasoning with established
library components.

### Implemented in the Project

- Hartley-normalized 8-point Essential matrix estimation
- Sampson-error scoring and deterministic Essential RANSAC
- Essential decomposition and cheirality-based pose selection
- conflict-aware feature-track construction
- expansion-aware initial-pair selection
- multi-view DLT triangulation
- depth, parallax, and reprojection validation
- incremental reconstruction state and camera-selection logic
- PnP acceptance / rejection quality gates
- bundle-adjustment residual construction and gauge handling
- landmark pruning
- view-graph analysis and reconstruction diagnostics
- Sim(3) alignment and quantitative evaluation

### Library-Assisted Components

- **OpenCV**
  - SIFT feature detection
  - brute-force descriptor matching
  - `solvePnPRansac`
  - `solvePnPRefineLM`
  - Rodrigues conversions
  - homography estimation used as degeneracy evidence
- **SciPy**
  - nonlinear least-squares optimization
- **NumPy**
  - matrix algebra, SVD, and vectorized geometry
- **Matplotlib**
  - reconstruction and diagnostic figures

## Setup

Requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/).

```powershell
uv sync
```

Datasets are stored locally under:

```text
data/<dataset_name>/
```

For example:

```text
data/templeRing/
data/templeSparseRing/
data/truck/
```

## Running a Reconstruction

Primary demonstration dataset:

```powershell
uv run python main.py --dataset templeRing
```

Wide-baseline / partial-reconstruction case:

```powershell
uv run python main.py --dataset templeSparseRing
```

Optional larger COLMAP-calibrated scene:

```powershell
uv run python main.py --dataset truck
```

Useful options:

| Option | Purpose |
|---|---|
| `--skip-ba` | Disable bundle adjustment for an ablation |
| `--skip-geometric-verification` | Disable Essential-matrix pair verification for an ablation |
| `--no-cache` | Recompute descriptor matches instead of loading the cached match graph |
| `--output-label <label>` | Write to `output/<dataset>_<label>/` |
| `--skip-colmap-export` | Disable optional COLMAP text-model export |

## Tests

Run the final regression suite:

```powershell
uv run python -m tests.unit.test_initialization
uv run python -m tests.synthetic.run_all
uv run python tests/integration/test_pipeline.py
```

The synthetic suite covers core geometry and reconstruction invariants,
including Essential geometry, triangulation, PnP pose recovery, bundle
adjustment, gauge constraints, degeneracies, and disconnected view graphs.

## Outputs

Each run writes artifacts under:

```text
output/<dataset>/
```

Depending on the dataset and whether reference poses are available:

| Artifact | Purpose |
|---|---|
| `reconstruction.json` | Intrinsics, registered camera poses, landmarks, and reconstruction status |
| `diagnostics.json` | Diagnostic events, camera statuses, rejection counts, and termination reason |
| `metrics.json` | Machine-readable quantitative evaluation when reference poses are available |
| `metrics.txt` | Human-readable evaluation summary |
| `pointcloud.ply` | Colored sparse point cloud |
| `matches_init_pair.png` | Verified correspondences for the selected initialization pair |
| `reconstruction_3d.png` | Sparse reconstruction and camera visualization |
| `camera_ring.png` | Estimated vs. reference camera trajectory |
| `reproj_hist.png` | Reprojection-error distribution |
| `ba_history.png` | Bundle-adjustment RMS history |
| `pose_error_by_registration.png` | Pose error by camera registration order |
| `colmap/` | Optional COLMAP-compatible sparse text model and registered images |

## Dataset Roles

- **Middlebury TempleRing** — primary complete reconstruction and quantitative
  evaluation dataset.
- **Middlebury TempleSparseRing** — controlled wide-baseline/connectivity
  failure case used to demonstrate expansion-aware initialization and
  structured failure diagnosis.
- **Tanks & Temples Truck** — optional larger-scene generalization experiment
  using a COLMAP calibration/reference reconstruction.
- **Phone/video capture** — optional qualitative experiment without reference
  poses.

## Optional Video Preprocessing

`extract_frames.py` is a standalone preprocessing utility for self-captured
video. It is not part of the SfM reconstruction pipeline.

Example:

```powershell
uv run python extract_frames.py input.mp4 data/my_capture --n-frames 70
```

It selects sharp frames from temporal windows, downsizes them, and writes
`f35_equiv.txt` for approximate phone-camera intrinsics.

The phone-image loader uses Pillow for EXIF access. Install the optional extra
when using that path:

```powershell
uv sync --extra phone
```

## Current Limitations

The current implementation assumes:

- one known shared intrinsic matrix `K`;
- an ideal pinhole camera or pre-undistorted images;
- a static scene;
- sufficient feature overlap between views;
- sparse reconstruction rather than dense multi-view stereo.

Exhaustive pair matching scales quadratically with image count, and repeated
global bundle adjustment becomes increasingly expensive on larger datasets.
