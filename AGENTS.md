# Repository Guidelines

## Project Structure & Module Organization

- `src/` contains the reconstruction pipeline: feature matching, tracks, two-view geometry, PnP, triangulation, bundle adjustment, evaluation, visualization, and COLMAP I/O/export.
- `main.py` runs incremental structure-from-motion; `extract_frames.py` prepares sharp, evenly spaced frames from a video.
- `test_synthetic.py` contains deterministic synthetic geometry and optimization sanity checks.
- `data/` holds local datasets and videos (ignored by Git). `output/` holds reconstruction products and caches; `report/` contains the written report and figures.
- Keep new reusable algorithms in the relevant `src/` module and keep command-line orchestration in root scripts.

## Build, Test, and Development Commands

Install/synchronize dependencies with `uv sync` (Python 3.13+). Run the fast sanity suite with:

```powershell
uv run python test_synthetic.py
```

Run a reconstruction using a dataset under `data/`:

```powershell
uv run python main.py --dataset templeSparseRing
```

Use `--skip-ba` for a faster run without bundle adjustment or `--no-cache` to rebuild feature matches. Extract video frames with, for example, `uv run python extract_frames.py data/myvideo.mp4 data/myvideo --n-frames 70`.

## Coding Style & Naming Conventions

Use four-space indentation, standard-library imports before third-party imports, and type hints for public functions. Follow the existing `snake_case` naming for modules, functions, and variables; use `PascalCase` for classes. Prefer NumPy arrays and vectorized operations where they match the existing numerical code. No formatter or linter is configured, so keep changes PEP 8-compatible and run the synthetic tests after edits.

## Testing Guidelines

Add focused `test_*` functions to `test_synthetic.py` or a new `test_*.py` module. Tests should be deterministic (seed random generators), assert numerical tolerances explicitly, and avoid requiring large local datasets. Run `uv run python test_synthetic.py` before submitting changes.

## Commit & Pull Request Guidelines

The existing history uses short, imperative-style summaries (for example, `ex01 - Refactor code structure`). Keep commits focused and describe the algorithm or behavior changed. Pull requests should include a concise summary, validation commands and results, relevant dataset/CLI options, and updated report figures or metrics when reconstruction output changes. Do not commit local datasets, videos, caches, or unrelated generated output.

## Configuration & Data Safety

Keep datasets under `data/` and generated match caches under `output/*/cache/`. Avoid hard-coding machine-specific paths; pass dataset and output choices through the existing CLI arguments.
