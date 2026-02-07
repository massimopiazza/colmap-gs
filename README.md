# colmap-gs

Google Colab workflow for running COLMAP (sparse + dense reconstruction) and generating outputs commonly used in Gaussian Splatting pipelines, plus Blender utilities for post-processing and area estimation.

This repository currently contains:

- `COLMAP-colab.ipynb`: the main end-to-end pipeline in Colab.
- `utils/colmap2nerf.py`: converts COLMAP outputs to `transforms.json` (Instant-NGP / NeRF style format).
- `utils/colmap_depth_viz.py`: renders COLMAP dense depth `.bin` maps into PNG visualizations.
- `utils/barrierEstimate.py`: Blender script to estimate protective barrier surface area over a selected region of terrain mesh.

## What this repository is for

Use this project when you want to:

1. Reconstruct a scene from photos in Google Colab with COLMAP.
2. Export camera transforms for NeRF or Gaussian Splatting tooling.
3. Produce dense depth, fused point clouds, and meshes (Poisson method).
4. Import the reconstructed mesh into Blender and estimate a draped barrier area over an area of interest.

## Prerequisites

## Accounts & platform

- Google account with Google Drive access.
- Google Colab (**GPU runtime**).
- Blender (only for `utils/barrierEstimate.py`).

## Colab runtime requirements

The notebook installs required system packages in Colab, including build tools and graphics dependencies. You still need:

- A GPU runtime for acceleration purposes (also mandatory for COLMAP dense reconstruction).
- Enough Drive free space for:
  - COLMAP build cache tarballs
  - Input images
  - Output reconstruction projects

## Local Python requirements (for scripts outside Colab)

`utils/barrierEstimate.py` must run inside Blender Python, not a standard terminal Python interpreter.

## Blender requirements (`utils/barrierEstimate.py`)

- Blender with Python API support (standard Blender install).
- Terrain mesh imported and scaled correctly (metric units recommended).
- Script execution from Blender Scripting workspace or text editor.

## Image requirements (important)

For stable SfM and dense MVS:

- High overlap between photos (typically 60 to 80 percent or more).
- Consistent exposure where possible.
- Limited motion blur.
- Avoid too many near-duplicate frames with minimal baseline.

## Recommended Google Drive folder setup

The notebook is configured around paths like these:

- Input images source:
  - `/content/drive/MyDrive/Gaussian-Splatting/images/<your_dataset_folder>`
- COLMAP build cache:
  - `/content/drive/MyDrive/COLMAP/colmap-build-cache`
- Final copied outputs:
  - `/content/drive/MyDrive/COLMAP/output`

If you use different folders, update notebook parameters in the setup cells.

## Repository Layout

```text
colmap-gs/
├── COLMAP-colab.ipynb
└── utils/
    ├── barrierEstimate.py
    ├── colmap2nerf.py
    └── colmap_depth_viz.py
```

## End-to-End Workflow (COLMAP-colab.ipynb)

The notebook is organized in blocks.

## 1) Install dependencies and mount Drive

- Mounts `/content/drive`.
- Clones this repository into `/content/colmap_gs`.
- Installs build and COLMAP dependencies via `apt`.

## 2) Validate or build COLMAP with CUDA

- Uses cache metadata and tarball from Drive.
- If cache is valid for current settings, it restores quickly.
- If cache is missing or incompatible, it builds:
  - Abseil
  - Ceres
  - COLMAP (headless, CUDA enabled)
- Caches the resulting install prefix back to Drive as a compressed tarball.

This design avoids rebuilding COLMAP every new Colab session.

## 3) Initialize per-run project folder and copy images

The notebook creates a timestamped project root:

```text
/content/<image_folder_name>_<timestamp>
```

with key paths:

- `images/`
- `database.db`
- `sparse/`
- `undistorted/`
- `dense/`

It copies images from the selected Drive source into `images/`.

## 4) Feature extraction and matching

Runs:

- `colmap feature_extractor`
- `colmap exhaustive_matcher`

with GPU toggled automatically based on runtime detection.

## 5) Sparse reconstruction and transforms export

Runs:

- `colmap mapper`
- `colmap image_undistorter` (for undistorted sparse workspace)
- `colmap model_converter` (binary model to TXT)
- `python colmap_gs/utils/colmap2nerf.py ...` to create `transforms.json`

## 6) Dense reconstruction and meshing

Runs:

- `colmap image_undistorter` (dense workspace)
- `colmap patch_match_stereo`
- `colmap stereo_fusion` -> `fused.ply`
- `colmap poisson_mesher` -> `meshed-poisson.ply`

## 7) Optional depth map rendering

The notebook includes an optional block using:

- `from colmap_gs.utils.colmap_depth_viz import render_colmap_depth_maps`

to generate colored PNG depth visualizations.

## 8) Copy results back to Drive

Copies entire timestamped project folder into:

- `/content/drive/MyDrive/COLMAP/output`

## Typical output tree

```text
<project_root>/
├── images/
├── database.db
├── sparse/
│   └── 0/
├── undistorted/
├── dense/
│   ├── stereo/
│   │   └── depth_maps/
│   ├── fused.ply
│   └── meshed-poisson.ply
└── transforms.json
```

If depth visualization is enabled, an additional `depth_viz/` folder is created.

## Utility Scripts

## `utils/colmap2nerf.py`

Purpose:

- Convert COLMAP text model (`cameras.txt`, `images.txt`) into a NeRF style `transforms.json`.
- Optionally run FFmpeg to extract frames from video.
- Optionally run COLMAP end-to-end from images.

Key points:

- Supports several camera models from COLMAP.
- Computes sharpness per image via variance of Laplacian.
- Can keep original COLMAP coordinates (`--keep_colmap_coords`) or reorient and recenter.
- Adds `aabb_scale` metadata for downstream NeRF tooling.
- Optional dynamic object masking with Detectron2 categories (`--mask_categories`).

Minimal usage (after COLMAP sparse model was exported to TXT):

```bash
python utils/colmap2nerf.py \
  --images /path/to/project/images \
  --text /path/to/project/sparse/0 \
  --out /path/to/project/transforms.json \
  --aabb_scale 16
```

## `utils/colmap_depth_viz.py`

Purpose:

- Read COLMAP dense depth maps (`*.geometric.bin`, `*.photometric.bin`) and produce:
  - colormapped RGB PNGs.
  - optional normalized 16-bit PNG depth images.

Expected input location:

- `<project_dir>/dense/stereo/depth_maps/...`

Basic usage:

```python
from utils.colmap_depth_viz import render_colmap_depth_maps

summary = render_colmap_depth_maps(
    "/path/to/project_root",
    cmap="turbo",
    kinds=("geometric", "photometric"),
    pmin=2,
    pmax=98,
)
print(summary)
```

Output location:

- `<project_dir>/depth_viz/<stereo_subfolder>/...`

## `utils/barrierEstimate.py`

This script runs inside Blender and estimates the one-sided surface area of a draped "umbrella" mesh over a selected terrain region. It is especially useful for rough sizing of rockfall protection barriers after importing the reconstructed mesh from COLMAP outputs.

## Intended workflow

1. Run `COLMAP-colab.ipynb` and generate a mesh (`dense/meshed-poisson.ply`).
2. Download or open that mesh in Blender.
3. In Blender Edit Mode, select vertices or faces corresponding to your area of interest.
4. Open `utils/barrierEstimate.py` in Blender's scripting editor.
5. Tune `CFG` values if needed, then run the script.
6. Read area result from Blender console:
   - `"[BarrierUmbrella] One-sided area estimate: <value> m^2"`

The script also creates a new object (default name `BarrierUmbrella`).

## What the script does internally

1. Detects which mesh actually has an active selection (handles multi-object edit mode).
2. Reads selected points in world coordinates and estimates a mean surface normal.
3. Builds a 2D projection plane and computes convex hull of selected points.
4. Offsets the hull outward by `margin_m`.
5. Lifts the initial polygon above terrain along `clearance_axis`.
6. Builds an n-gon mesh, triangulates it, and subdivides it.
7. Applies shrinkwrap to drape onto terrain and smooths the result.
8. Enforces a minimum clearance from terrain along selected axis via BVH ray casting.
9. Computes total world-space one-sided area by summing triangulated face areas.

## Most important configuration fields (`CFG`)

- `margin_m`: outward buffer around selected hull.
- `subdivide_cuts`: draping resolution.
- `smooth_iterations`, `smooth_factor`: smoothing strength.
- `use_shrinkwrap`, `shrinkwrap_offset_m`, `shrinkwrap_method`: draping behavior.
- `force_world_xy`: project in world XY instead of mean-normal plane.
- `clearance_axis`: axis used as "up" for clearance (`X+`, `X-`, `Y+`, `Y-`, `Z+`, `Z-`).
- `axis_clearance_m`: required minimum stand-off distance.
- `initial_lift_along_axis_m`: initial pre-shrinkwrap lift to reduce wrong-side snapping.
- `apply_scale_to_terrain`: applies terrain scale so units are consistent.

## Practical tuning guidance

- Start with moderate subdivision (`20` to `40`) and inspect result.
- If umbrella sticks to wrong side, increase `initial_lift_along_axis_m`, switch to `PROJECT` shrinkwrap, and verify `clearance_axis`.
- If area is too jagged, increase smoothing slightly or reduce subdivision.
- If area is too small, increase `margin_m`.
- Confirm Blender scene units are metric and that imported mesh scale is correct.

## Limitations to keep in mind

- The area of interest boundary is convexified by hull construction, so concave regions can be overestimated.
- Output is a one-sided surface area.
- Quality depends on mesh quality and selection quality.
- Very noisy meshes can bias draping and area.

## Mathematical Background

## COLMAP optimization (high-level)

Sparse reconstruction estimates camera poses and 3D points by minimizing reprojection error:

$$
\min_{\{R_i,t_i,X_j\}}
\sum_{(i,j)\in\mathcal{O}}
\left\|
\pi(K_i, R_i, t_i, X_j) - x_{ij}
\right\|^2
$$

where:

- $R_i,t_i$: camera pose for image $i$
- $X_j$: 3D point $j$
- $x_{ij}$: observed pixel
- $\pi(\cdot)$: camera projection model.

## Barrier surface construction and area

Selected 3D points are projected to a 2D basis, hull is computed, and polygon area in 2D can be written with the shoelace formula:

$$
A_{2D}=\frac{1}{2}\left|\sum_{k=1}^{n}(x_k y_{k+1}-x_{k+1}y_k)\right|
$$

After draping and triangulation, 3D surface area is:

$$
A_{3D}=\sum_{f\in\mathcal{F}}
\frac{1}{2}
\left\|
(v_1-v_0)\times(v_2-v_0)
\right\|
$$

for each triangle $(v_0,v_1,v_2)$.

Clearance constraint enforced per umbrella vertex:

$$
\delta_k=(p_k-q_k)\cdot a \ge c
$$

where:

- $p_k$: umbrella vertex
- $q_k$: terrain hit point from raycast or nearest fallback
- $a$: chosen clearance axis unit vector
- $c$: required clearance (`axis_clearance_m`).

## Suggested End-to-End Procedure

1. Organize images in Google Drive under a dedicated folder.
2. Open and run `COLMAP-colab.ipynb` top to bottom.
3. Confirm outputs in timestamped `project_root` and copied Drive output folder.
4. Import `meshed-poisson.ply` into Blender.
5. Select area of interest and run `utils/barrierEstimate.py`.
6. Save reported area with your project metadata (dataset name, timestamp, CFG values used).

## Troubleshooting

- `colmap` not found in Colab cells:
  - Ensure COLMAP build or cache-restore step completed successfully.
- Sparse model folder missing `0`:
  - Check feature matching quality and image overlap.
- Dense reconstruction is too noisy:
  - Adjust PatchMatch triangulation thresholds and improve image quality.
- `barrierEstimate.py` says no selection found:
  - Ensure terrain mesh is in Edit Mode and vertices or faces are selected.
- Unrealistic barrier area:
  - Verify scale and units in Blender, then retune `margin_m`, subdivision, and clearance settings.

## Notes

- `barrierEstimate.py` and Blender operations assume geometry scale represents real meters.
- Reproducibility improves when you keep a record of notebook parameters and script `CFG` values for each run.
