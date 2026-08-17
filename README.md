# Robust and Self-Diagnosing Incremental 3D Reconstruction

An incremental Structure-from-Motion (SfM) pipeline built from first principles in Python, NumPy, and SciPy, with OpenCV support for local feature detection, matching, and PnP resectioning. The system recovers camera poses and sparse 3D geometry from calibrated images while actively diagnosing geometric health, failure modes, and view-graph connectivity.

---

## What the Project Does

The reconstruction pipeline executes a structured sequential workflow:

```text
Images
  ↓
SIFT feature detection & descriptor matching
  ↓
Pairwise Essential-matrix geometric verification
  ↓
Conflict-free feature-track construction
  ↓
Expansion-aware two-view initialization
  ↓
Incremental PnP camera registration loop
  ↓
Multi-view DLT triangulation with geometric quality gates
  ↓
Periodic & final bundle adjustment (BA) with gauge anchoring
  ↓
Reconstruction finalization, status classification & failure diagnostics
```

Rather than silently halting or treating partial reconstructions as complete, the system outputs explicit diagnostics including status (`SUCCESS`, `PARTIAL`, `FAILED`), termination reasons (such as `COMPLETED`, `NO_2D3D_BRIDGE`, `NO_REGISTRABLE_CAMERA`), per-camera registration states, and view-graph component analysis.

---

## Implementation Boundaries

The project distinguishes custom mathematical/pipeline implementations from library-assisted components:

### Custom Project Logic
- **Two-View Epipolar Geometry**: Hartley-normalized 8-point Essential matrix estimation, Sampson distance scoring, deterministic RANSAC, Essential matrix decomposition, and cheirality-based pose recovery.
- **Track Management**: Union-find feature-track builder with strict multi-observation conflict detection and rejection.
- **Expansion-Aware Initialization**: Multi-criteria pair selection evaluating inlier ratio, parallax angles, cheirality, homography degeneracy, and 2D↔3D expansion support for subsequent views.
- **Triangulation**: Multi-view DLT triangulation with strict validation gates (positive depth in all views, minimum parallax angle, maximum reprojection error).
- **State Machine & Diagnostics**: Immutable camera registration state machine, outlier landmark pruning, view-graph connected components, and failure categorization.
- **Bundle Adjustment Integration**: Typed-state-to-vector flattening, sparse Jacobian structure construction, and gauge constraint enforcement (fixed anchor camera pose and scale camera baseline distance).
- **Evaluation**: Sim(3) Umeyama trajectory alignment, geodesic rotation error metrics, normalized camera-center translation errors, and COLMAP text/binary export.

### Library-Assisted Components
- **OpenCV (`cv2`)**: SIFT feature detection and brute-force descriptor matching (`cv2.SIFT_create`, `cv2.BFMatcher`), PnP RANSAC pose estimation (`cv2.solvePnPRansac`), Rodrigues axis-angle conversion (`cv2.Rodrigues`), and homography model evidence (`cv2.findHomography`).
- **SciPy (`scipy.optimize.least_squares`)**: Sparse Levenberg-Marquardt / trust-region reflective nonlinear solver for bundle adjustment.
- **NumPy**: Matrix algebra, SVD decompositions, and vectorized geometric projections.

---

## Architecture

The codebase separates high-level pipeline orchestration from pure numerical algorithms:

```text
src/sfm.py (Public facade: exposes run_sfm)
  ↓
src/pipeline/orchestration.py
  ├── verification.py          (Pairwise geometric verification before track building)
  ├── initialization_stage.py  (Expansion-aware initialization & map seeding)
  ├── registration.py          (Incremental PnP candidate selection & map expansion)
  └── finalization.py          (Final BA, outlier pruning, camera statuses & statistics)
```

**Shared Pipeline Helpers & Domain Models:**
- `src/pipeline/reconstruction_ops.py`: Typed triangulation, outlier pruning, and reprojection RMS.
- `src/pipeline/bundle.py`: Boundary between typed `ReconstructionState` and numerical BA.
- `src/pipeline/diagnostics.py`: Match-graph connectivity analysis and failure statistics.
- `src/geometry/`: Pure geometric primitives (`essential.py`, `projection.py`, `pose.py`).
- `src/initialization.py`: Two-view pair evaluation and expansion scoring.
- `src/triangulation.py`: Multi-view DLT triangulation algorithms.
- `src/pnp.py`: Resectioning, conditioned DLT PnP, and nonlinear refinement.
- `src/bundle_adjustment.py`: Joint camera pose and 3D landmark optimization.
- `src/tracks.py`: Union-find feature tracking with conflict prevention.
- `src/models.py`: Typed immutable data classes (`CameraPose`, `Landmark`, `Reconstruction`, `ReconstructionState`).
- `src/validation/`: Degeneracy detection (`degeneracy.py`) and diagnostic schemas (`diagnostics.py`).

---

## Setup

Requires Python 3.13+ and `uv`.

```powershell
uv sync
```

Place datasets under `data/<dataset_name>/` (e.g. `data/templeRing/`, `data/templeSparseRing/`, `data/truck/`).

---

## Tests & Validation

Run the deterministic sanity test suite:

```powershell
# 1. Initialization unit tests
uv run python -m tests.unit.test_initialization

# 2. Synthetic geometry, PnP, and BA gauge tests
uv run python -m tests.synthetic.run_all

# 3. End-to-end reconstruction integration test
uv run python tests/integration/test_pipeline.py
```

---

## Running Reconstruction

Run incremental reconstruction using `main.py`:

```powershell
# Run Middlebury TempleRing (47 images)
uv run python main.py --dataset templeRing

# Run Middlebury TempleSparseRing (16 wide-baseline images)
uv run python main.py --dataset templeSparseRing

# Run Tanks & Temples Truck (84 images)
uv run python main.py --dataset truck
```

### Useful CLI Options
- `--skip-ba`: Disables periodic and final bundle adjustment (for ablation comparison).
- `--skip-geometric-verification`: Disables Essential-matrix verification of pairwise matches.
- `--no-cache`: Forces recomputation of SIFT matches instead of loading cached `.npz`.
- `--output-label <label>`: Writes results to a custom directory under `output/<dataset>_<label>/`.
- `--skip-colmap-export`: Skips exporting COLMAP-compatible sparse text files.

---

## Outputs

Each run produces artifacts in `output/<dataset>/`:
- `metrics.txt`: Summary of registered views, landmarks, reprojection RMS, rotation errors, and camera center errors.
- `reconstruction.json`: Serialized camera poses, 3D point coordinates, track observations, and intrinsics.
- `diagnostics.json`: Diagnostic events, view-graph components, camera statuses, and termination reasons.
- `reconstruction.ply`: 3D point cloud colored from original pixel intensities.
- `colmap/sparse/0/`: COLMAP-compatible `cameras.txt`, `images.txt`, `points3D.txt`.
- Diagnostic plots: Camera trajectory top-down view, 3D scatter plots, reprojection error histograms, and BA history curves.

---

## Optional Video Preprocessing

An optional preprocessing script is provided to sample sharp, evenly spaced frames from continuous video captures:

```powershell
uv run python extract_frames.py data/myvideo.mp4 data/myvideo --n-frames 70
```

*(Note: `extract_frames.py` is a standalone capture-preparation utility and is not part of the core reconstruction pipeline.)*
