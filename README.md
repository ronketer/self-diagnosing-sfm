# Incremental Structure-from-Motion from Scratch

This project implements a complete incremental Structure-from-Motion pipeline
in NumPy and SciPy. OpenCV is used only for SIFT detection and descriptor
matching; essential-matrix estimation, RANSAC, pose recovery, triangulation,
PnP, track management, bundle adjustment, and evaluation are implemented in
the repository.

## Headline results

| Dataset | Registered | Points | RMS reprojection | Mean rotation error |
|---|---:|---:|---:|---:|
| Middlebury TempleRing | 47/47 | 13,811 | 0.383 px | 0.265° |
| Tanks & Temples Truck | 84/84 | 49,641 | 0.520 px | 0.055° against COLMAP reference |

The final project scope is classical SfM: quantitative accuracy, robustness
ablations, generalization to a larger real scene, and analysis of sparse and
self-captured failure cases.

## Setup and tests

Python 3.13 or newer and `uv` are required.

```powershell
uv sync
uv run python test_synthetic.py
```

Datasets are local and ignored by Git. Place each dataset under `data/<name>`.

## Reconstruction

```powershell
uv run python main.py --dataset templeRing
```

Useful options:

- `--skip-ba` disables bundle adjustment.
- `--skip-geometric-verification` keeps raw appearance matches for an ablation.
- `--output-label <label>` writes into a separate output directory.
- `--skip-colmap-export` omits the optional standards-interoperability export.
- `--no-cache` rebuilds feature matches.

The two TempleRing ablations used in the report are reproducible with:

```powershell
uv run python main.py --dataset templeRing --skip-ba --output-label no_ba --skip-colmap-export
uv run python main.py --dataset templeRing --skip-geometric-verification --output-label no_geom --skip-colmap-export
uv run python summarize_results.py
```

Generated metrics, figures, point clouds, caches, and exports are written under
`output/` and ignored by Git. The report and selected final figures are under
`report/`.
