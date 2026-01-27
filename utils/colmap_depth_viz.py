# colmap_depth_viz.py
# ------------------------------------------------------------
# Render COLMAP dense depth maps (*.geometric.bin / *.photometric.bin)
# into colormapped PNGs by scanning a COLMAP project folder.
#
# Usage:
#   from colmap_depth_viz import render_colmap_depth_maps
#   render_colmap_depth_maps("/Users/massimopiazza/Downloads/drone-footage_2026-01-19T145721")
#
# Output:
#   <project_dir>/depth_viz/<stereo_subfolder>/<basename>.<kind>.<cmap>.png
#   and also a 16-bit normalized depth PNG:
#   <project_dir>/depth_viz/<stereo_subfolder>/<basename>.<kind>.u16.png
# ------------------------------------------------------------

from __future__ import annotations

import os
import glob
from typing import Iterable, Optional, Tuple, List

import numpy as np
from PIL import Image

# matplotlib is only used for colormap lookup (no plotting)
from matplotlib import cm


def _read_colmap_bin_array(path: str) -> np.ndarray:
    """
    Read COLMAP depth/normal map stored as:
      ASCII header: "width&height&channels&"
      followed by float32 data in row-major order.
    Returns:
      - depth map as HxW float32 if channels==1
      - normal/other map as HxWxC if channels>1
    """
    with open(path, "rb") as f:
        header = b""
        ampersands = 0
        # read until we've seen 3 '&' (width&, height&, channels&)
        while ampersands < 3:
            c = f.read(1)
            if not c:
                raise ValueError(f"Unexpected EOF while reading header: {path}")
            header += c
            if c == b"&":
                ampersands += 1

        # header example: b'640&480&1&'
        parts = header.decode("ascii", errors="strict").strip().split("&")
        if len(parts) < 3:
            raise ValueError(f"Invalid header in {path}: {header!r}")
        w, h, ch = map(int, parts[:3])

        data = np.fromfile(f, dtype=np.float32, count=w * h * ch)
        if data.size != w * h * ch:
            raise ValueError(
                f"Expected {w*h*ch} floats, got {data.size} in {path}"
            )

    arr = data.reshape((h, w, ch))
    if ch == 1:
        return arr[:, :, 0]
    return arr


def _normalize_depth(
    depth: np.ndarray,
    pmin: float,
    pmax: float,
    invalid_leq_zero: bool = True,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Normalize depth into 0..1 with percentile clipping.
    Returns normalized float32 HxW and (lo, hi) used.
    """
    d = depth.astype(np.float32, copy=True)

    invalid = ~np.isfinite(d)
    if invalid_leq_zero:
        invalid |= (d <= 0)

    d[invalid] = np.nan
    # If everything is invalid, bail gracefully
    if np.all(np.isnan(d)):
        dn = np.zeros_like(d, dtype=np.float32)
        return dn, (np.nan, np.nan)

    lo = float(np.nanpercentile(d, pmin))
    hi = float(np.nanpercentile(d, pmax))
    # Guard against degenerate ranges
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        dn = np.nan_to_num(d, nan=0.0).astype(np.float32)
        # bring into 0..1 as best-effort
        mn, mx = float(np.min(dn)), float(np.max(dn))
        if abs(mx - mn) < 1e-12:
            return np.zeros_like(dn, dtype=np.float32), (mn, mx)
        dn = (dn - mn) / (mx - mn)
        return dn.astype(np.float32), (mn, mx)

    d = np.clip(d, lo, hi)
    dn = (d - lo) / (hi - lo + 1e-12)
    dn = np.nan_to_num(dn, nan=0.0).astype(np.float32)
    return dn, (lo, hi)


def _save_colormap_png(
    dn01: np.ndarray,
    out_png: str,
    cmap_name: str = "turbo",
) -> None:
    """
    dn01: HxW float in 0..1
    Saves RGB PNG using matplotlib colormap.
    """
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(dn01)  # HxW x4 in 0..1
    rgb = (rgba[:, :, :3] * 255.0).round().astype(np.uint8)
    Image.fromarray(rgb).save(out_png)


def _save_uint16_png(dn01: np.ndarray, out_png: str) -> None:
    """
    Save normalized depth 0..1 as 16-bit PNG (useful for post-processing).
    """
    u16 = (dn01 * 65535.0).round().astype(np.uint16)
    Image.fromarray(u16).save(out_png)


def _find_depth_maps(project_dir: str) -> List[str]:
    """
    Find all COLMAP dense depth maps under:
      <project_dir>/dense/stereo/depth_maps/*/*.bin
    and also handle if depth_maps is directly inside dense/stereo/depth_maps/<scene>.
    """
    dense_root = os.path.join(project_dir, "dense", "stereo", "depth_maps")
    if not os.path.isdir(dense_root):
        return []

    # Typical: depth_maps/<stereo_subfolder>/*.bin
    patterns = [
        os.path.join(dense_root, "*", "*.bin"),
        os.path.join(dense_root, "*.bin"),
    ]

    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    # Deduplicate and sort
    files = sorted(set(files))
    return files


def render_colmap_depth_maps(
    project_dir: str,
    *,
    kinds: Iterable[str] = ("geometric", "photometric"),
    cmap: str = "turbo",
    pmin: float = 2.0,
    pmax: float = 98.0,
    invalid_leq_zero: bool = True,
    overwrite: bool = False,
    output_subdir: str = "depth_viz",
    write_u16: bool = True,
) -> dict:
    """
    Scan a COLMAP project directory for depth maps and render colormapped PNGs.

    Args:
      project_dir: e.g. "/Users/massimopiazza/Downloads/croatia-house_2026-01-19T145721"
      kinds: which depth maps to render: "geometric", "photometric" (both by default)
      cmap: matplotlib colormap name (e.g., "turbo", "magma", "viridis")
      pmin/pmax: percentile clipping for visualization robustness
      invalid_leq_zero: treat depth<=0 as invalid
      overwrite: if False, skips outputs that already exist
      output_subdir: outputs go into <project_dir>/<output_subdir>/...
      write_u16: also write a normalized 16-bit PNG

    Returns:
      summary dict with counts and output folder.
    """
    project_dir = os.path.abspath(os.path.expanduser(project_dir))
    all_bins = _find_depth_maps(project_dir)

    if not all_bins:
        raise FileNotFoundError(
            f"No depth maps found under {project_dir}/dense/stereo/depth_maps"
        )

    # Filter by kind suffix
    kinds = tuple(kinds)
    def is_wanted(path: str) -> bool:
        name = os.path.basename(path)
        for k in kinds:
            if f".{k}.bin" in name:
                return True
        return False

    bins = [p for p in all_bins if is_wanted(p)]
    if not bins:
        raise FileNotFoundError(
            f"Found depth_maps, but none matched kinds={kinds}. "
            f"Example files: {all_bins[:3]}"
        )

    out_root = os.path.join(project_dir, output_subdir)
    os.makedirs(out_root, exist_ok=True)

    rendered = 0
    skipped = 0
    failed = 0

    for path in bins:
        try:
            stereo_subfolder = os.path.basename(os.path.dirname(path))
            # If depth_maps/*.bin (no subfolder), keep a default bucket
            if stereo_subfolder == "depth_maps":
                stereo_subfolder = "default"

            base = os.path.basename(path)
            # base like: IMG.JPG.geometric.bin -> stem: IMG.JPG.geometric
            stem = base[:-4] if base.endswith(".bin") else base

            out_dir = os.path.join(out_root, stereo_subfolder)
            os.makedirs(out_dir, exist_ok=True)

            out_png = os.path.join(out_dir, f"{stem}.{cmap}.png")
            out_u16 = os.path.join(out_dir, f"{stem}.u16.png")

            if not overwrite and os.path.exists(out_png) and (not write_u16 or os.path.exists(out_u16)):
                skipped += 1
                continue

            arr = _read_colmap_bin_array(path)

            # Skip non-depth maps (channels>1) unless you explicitly want them
            if arr.ndim != 2:
                # You can extend this to normals if you want.
                skipped += 1
                continue

            dn, _ = _normalize_depth(arr, pmin=pmin, pmax=pmax, invalid_leq_zero=invalid_leq_zero)
            _save_colormap_png(dn, out_png, cmap_name=cmap)
            if write_u16:
                _save_uint16_png(dn, out_u16)

            rendered += 1

        except Exception:
            failed += 1

    return {
        "project_dir": project_dir,
        "output_dir": out_root,
        "rendered": rendered,
        "skipped": skipped,
        "failed": failed,
        "kinds": kinds,
        "cmap": cmap,
        "percentiles": (pmin, pmax),
    }