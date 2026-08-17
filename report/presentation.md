---
marp: true
title: Robust Self-Diagnosing Incremental SfM
paginate: true
---

# Robust Self-Diagnosing Incremental 3D Reconstruction

**Final project — 3D Computer Vision**  
Ron Keter

> Recover sparse 3D geometry from calibrated images, while explaining when reconstruction is reliable, partial, or geometrically blocked.

**Suggested visual:** best TempleRing point cloud with camera frustums.

---

# 1. Problem

Given overlapping calibrated images of a static scene:

```text
2D images
   ↓
camera poses
   ↓
sparse 3D landmarks
   ↓
bundle-adjusted reconstruction
```

The challenge is not only to reconstruct 3D structure, but to know when the geometry is trustworthy.

**Suggested visual:** 3–4 input images → sparse 3D reconstruction.

---

# 2. Main idea

A normal SfM pipeline may simply succeed or fail.

This project adds **diagnostic reasoning**:

```text
Can this image pair initialize a stable 3D model?
Can this camera be registered from existing 3D points?
Why did reconstruction stop?
Was the failure PnP, triangulation, low parallax, or graph disconnection?
```

**Key message:**  
A low reprojection error alone does not prove that the whole scene was reconstructed.

---

# 3. Final pipeline architecture

```text
Images
  ↓
SIFT features + descriptor matching
  ↓
Essential-matrix verification
  ↓
Conflict-aware feature tracks
  ↓
Expansion-aware two-view initialization
  ↓
Incremental PnP camera registration
  ↓
Validated triangulation
  ↓
Periodic bundle adjustment
  ↓
Final diagnostics + quality status
```

**Suggested visual:** architecture diagram from code modules.

---

# 4. Code architecture

```text
src/sfm.py
  └── run_sfm()

src/pipeline/
  ├── orchestration.py          high-level stage sequencing
  ├── verification.py           pairwise geometric verification
  ├── initialization_stage.py   initial cameras + initial map
  ├── registration.py           incremental PnP loop
  ├── finalization.py           final BA, status, metrics
  ├── reconstruction_ops.py     triangulation / pruning / RMS
  ├── bundle.py                 BA adapter
  └── diagnostics.py            view-graph / failure stats
```

The goal was to make the code readable as the same story as the algorithm.

---

# 5. Geometry conventions

World-to-camera convention:

\[
X_c = R X_w + t
\]

Camera center:

\[
C = -R^Tt
\]

Projection:

\[
x \sim K(RX+t)
\]

Essential constraint:

\[
x_2^T E x_1 = 0
\]

Bundle adjustment objective:

\[
\min_{R_i,t_i,X_j}
\sum_{ij}\left\|x_{ij}-\pi(K(R_iX_j+t_i))\right\|^2
\]

---

# 6. What I implemented vs. what I used

**Implemented in the project**

- normalized Essential-matrix estimation and RANSAC logic;
- Essential decomposition and cheirality selection;
- feature-track construction with conflict prevention;
- triangulation validation: depth, parallax, reprojection;
- incremental SfM orchestration and diagnostics;
- bundle-adjustment residual construction and gauge handling;
- evaluation, Sim(3) alignment, and metrics.

**Used libraries**

- OpenCV: SIFT, matching support, production PnP, Rodrigues, homography evidence;
- SciPy: nonlinear least-squares optimization;
- NumPy: numerical linear algebra and geometry.

---

# 7. Correctness checks

A point is accepted only if:

```text
positive depth
AND sufficient parallax
AND low reprojection error
```

A camera is accepted only if:

```text
enough 2D↔3D correspondences
AND enough PnP inliers
AND sufficient inlier ratio
AND acceptable reprojection RMS
```

The reconstruction also reports:

```text
SUCCESS / PARTIAL / FAILED
COMPLETED / NO_2D3D_BRIDGE / NO_REGISTRABLE_CAMERA / ...
```

**Suggested visual:** triangulation rays with low-vs-good parallax.

---

# 8. Why initialization changed

The first version selected the valid pair with the largest parallax.

That can choose an isolated pair:

```text
pair (11,15)
good 2-view geometry
but no third camera has enough 2D↔3D support
→ reconstruction stops at 2 cameras
```

Final rule:

```text
choose a geometrically valid pair
that can also expand to more cameras
```

This fixed TempleSparseRing:

```text
selected pair: (0,1)
best next support: 358 correspondences to image 10
registered: 4 / 16 cameras
```

---

# 9. Result: TempleRing successful case

```text
Dataset: Middlebury TempleRing
Registered cameras: 47 / 47 (100%)
3D landmarks: 13,811
Observations: 56,601
Mean track length: 4.10
Final RMS: 0.383 px
Mean rotation error: 0.265°
Max rotation error: 0.776°
Mean center error: 0.13% of trajectory diameter
Status: SUCCESS
Termination: COMPLETED
```

**Suggested visual:** final TempleRing reconstruction + camera ring.

---

# 10. Result: TempleSparseRing diagnostic partial case

```text
Dataset: Middlebury TempleSparseRing
Registered cameras: 4 / 16
3D landmarks: 655
Observations: 2,007
Final RMS: 0.238 px
Mean rotation error: 0.217°
Status: PARTIAL
Termination: NO_2D3D_BRIDGE
Verified graph: 8 components
Active component: [0, 1, 9, 10]
```

Interpretation:

```text
The cameras that could be connected were reconstructed accurately.
The remaining images lacked 2D↔3D bridge support.
```

**Suggested visual:** active component vs disconnected components.

---

# 11. Important lesson

SparseRing has **lower RMS** than TempleRing:

```text
TempleRing RMS:       0.383 px, 47/47 cameras
TempleSparseRing RMS: 0.238 px, 4/16 cameras
```

So:

> Low reprojection error can describe a small accurate submodel, not necessarily a complete reconstruction.

That is why the system reports coverage, graph connectivity, PnP attempts, and termination reason.

---

# 12. Bundle adjustment

Periodic BA improves the map used for later PnP.

Example TempleRing behavior:

```text
register cameras
  ↓
triangulate new landmarks
  ↓
RMS rises temporarily
  ↓
BA reduces RMS
  ↓
outliers pruned
```

Final TempleRing:

```text
final BA:      0.518 → 0.518 px
post-prune BA: 0.383 → 0.383 px
```

**Interpretation:** the solution was already near local optimum; pruning removed remaining high-error landmarks.

---

# 13. Degeneracy and failure reasoning

The system identifies or reports evidence for:

```text
LOW_PARALLAX
NEGATIVE_DEPTH
HIGH_REPROJECTION_ERROR
POOR_CHEIRALITY
HOMOGRAPHY_DOMINANT
INSUFFICIENT_PNP_CORRESPONDENCES
LOW_PNP_INLIER_RATIO
NO_2D3D_BRIDGE
```

For TempleRing, the top triangulation rejection was:

```text
LOW_PARALLAX: 7,376 tracks
```

That means many candidate tracks existed, but were not geometrically reliable enough for stable depth.

---

# 14. Tests and validation

Regression coverage includes:

```text
Essential matrix and pose recovery
deterministic RANSAC
triangulation and multi-view parallax
track conflict handling
PnP quality gates
bundle-adjustment gauge invariants
degenerate scenes
disconnected view graph
end-to-end integration test
```

Final test commands passed:

```powershell
uv run python -m tests.unit.test_initialization
uv run python -m tests.synthetic.run_all
uv run python tests/integration/test_pipeline.py
```

---

# 15. Limitations

Current assumptions:

```text
known calibrated intrinsics
shared pinhole camera model
zero or pre-corrected lens distortion
static scene
sparse reconstruction only
```

Scalability limitations:

```text
exhaustive pair matching is expensive
global BA can become slow
SIFT may fail on wide-baseline / low-texture / reflective scenes
```

Natural next steps:

```text
sequence-aware matching
better wide-baseline descriptors
radial-distortion calibration
rotation averaging / loop closure
dense reconstruction or meshing
```

---

# 16. Final takeaway

This project is not only an SfM implementation.

It is a **self-diagnosing SfM pipeline**:

```text
It reconstructs cameras and sparse 3D points
AND
it explains whether the result is complete, partial, or geometrically blocked.
```

One-sentence contribution:

> I implemented an incremental SfM system that reconstructs sparse 3D geometry from calibrated images and reports the geometric evidence behind success or failure, including low parallax, PnP support, triangulation rejection, and disconnected view-graph components.

---

# Backup: questions to be ready for

## What is the Essential Matrix?

It relates corresponding normalized image points from two calibrated cameras:

\[
x_2^T E x_1 = 0
\]

It encodes relative rotation and translation direction.

---

## Why is scale ambiguous?

With one moving calibrated camera, the reconstruction can be scaled arbitrarily:

```text
camera translations × s
3D points × s
same image projections
```

So the system fixes a gauge: one camera pose is fixed and the initial baseline defines scale.

---

## Why use PnP after initialization?

After some landmarks exist, a new image can be registered from:

```text
2D keypoints in the new image
↔
existing 3D landmarks
```

That is exactly the Perspective-n-Point problem.

---

## Why can RMS be misleading?

RMS measures only the observations included in the reconstructed model.

A small partial model can have excellent RMS while most cameras are missing.

Therefore the project reports:

```text
registered cameras
view-graph components
PnP attempts
termination reason
```

alongside RMS.

---

## What would improve SparseRing?

The current failure is a feature-graph bridge problem, not a PnP failure.

Possible improvements:

```text
better wide-baseline local features
more images / smaller angular gaps
guided matching from known approximate geometry
rotation averaging across disconnected verified pairs
```
