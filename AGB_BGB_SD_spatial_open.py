# -*- coding: utf-8 -*-
"""
AGB / BGB / carbon loss estimation with spatially adjusted uncertainty.

Setup: put your input files under DATA_DIR (edit the paths in the CONFIG section below if your names differ)

Dependencies: geopandas, numpy, pandas, rasterio, scipy, shapely.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy.spatial import cKDTree
from shapely.geometry import mapping
from shapely.ops import unary_union

# =========================================================
# CONFIG -- edit these paths/names for your own data
# =========================================================

DATA_DIR = Path("data")

REFERENCE_TIF = DATA_DIR / ""     # defines target CRS/grid
STUDY_AREA_SHP = DATA_DIR / ""      # state/region polygons
ROAD_SHP = DATA_DIR / ""                 # road-expansion polygons
LOGGING_SHP = DATA_DIR / ""            # logging-activity polygons
AGB2010_TIF = DATA_DIR / ""
SD2010_TIF = DATA_DIR / ""
PARAMS_JSON = DATA_DIR / "variogram_params.json"  # from the variogram-fitting script

OUTPUT_DIR = DATA_DIR / "output"
BUFFER_DIR = DATA_DIR / "buffer_output" / "output"
AGB_PROJECTED = BUFFER_DIR / "AGB2010_projected.tif"
SD_PROJECTED = BUFFER_DIR / "SD2010_projected.tif"

DETAIL_CSV = OUTPUT_DIR / "TABLE_S8_spatial_long.csv"
STATE_CSV = OUTPUT_DIR / "STATE_ROAD_AGB_LOSS_spatial.csv"

OVERWRITE_PROJECTED = False

# =========================================================
# MODEL PARAMETERS
# =========================================================

# Fraction of AGB converted to carbon (IPCC default for biomass -> C).
CARBON_FACTOR = 0.51

# BGB/AGB ratios for the three uncertainty scenarios.
BGB_RATIO_SCENARIO_1_LOW = 0.24   # applies where AGB > threshold (dry forests)
BGB_RATIO_SCENARIO_1_HIGH = 0.39  # applies where AGB <= threshold
BGB_RATIO_SCENARIO_2 = 0.75
BGB_RATIO_SCENARIO_3 = 0.43
AGB_THRESHOLD = 75.0  # Mg/ha; switches between the two scenario-1 ratios

# Vector features are processed in chunks to bound memory usage.
VECTOR_CHUNK_SIZE = 3000
STATE_VECTOR_CHUNK_SIZE = 1000
SPATIAL_QUERY_CHUNK_SIZE = 1000

# Exact pixel-pair covariance is too expensive for million-pixel objects.
# Road/logging/state spatial SD uses 5-km block-level covariance; the whole
# study area uses coarser 25-km blocks.
OBJECT_SPATIAL_BLOCK_KM = 5.0
CALCULATE_WHOLE_STUDY_AREA_SPATIAL = True
WHOLE_STUDY_AREA_BLOCK_KM = 25.0

# Name of the state/region column in the study-area shapefile. If absent,
# the first attribute column is used.
STATE_NAME_COLUMN = "name2"


# =========================================================
# COMMON HELPERS
# =========================================================

def assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")


def load_variogram_params() -> tuple[float, float, float, dict]:
    assert_exists(PARAMS_JSON)
    with open(PARAMS_JSON, "r", encoding="utf-8") as f:
        params = json.load(f)
    range_m = float(params["range_m"])
    range_km = float(params["range_km"])
    partial_sill = float(params.get("partial_sill", 1.0))
    sill = float(params.get("sill", partial_sill))
    correlation_scale = partial_sill / sill if sill > 0 else 1.0
    return range_m, range_km, correlation_scale, params


def reproject_match_reference(input_raster: Path, output_raster: Path, reference_raster: Path) -> None:
    if output_raster.exists() and output_raster.stat().st_size > 0 and not OVERWRITE_PROJECTED:
        print(f"Using existing projected raster: {output_raster}")
        return

    print(f"Reprojecting {input_raster.name} -> {output_raster.name}")
    with rasterio.open(reference_raster) as ref, rasterio.open(input_raster) as src:
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": ref.crs,
            "transform": ref.transform,
            "width": ref.width,
            "height": ref.height,
            "dtype": "float32",
            "nodata": np.nan,
            "compress": "lzw",
        })
        with rasterio.open(output_raster, "w", **kwargs) as dst:
            dest = np.full((ref.height, ref.width), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
            dst.write(dest, 1)


def read_vector_projected(path: Path, target_crs) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS. Define its source CRS before projecting to the raster CRS.")
    gdf = gdf.to_crs(target_crs)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def build_study_mask(src, study_area: gpd.GeoDataFrame) -> np.ndarray:
    shapes = [(geom, 1) for geom in study_area.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        raise RuntimeError("Study area shapefile contains no valid geometry.")
    mask_arr = rasterize(
        shapes,
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="uint8",
    )
    return mask_arr.astype(bool)


def spherical_rho(distance_m: np.ndarray, range_m: float, correlation_scale: float = 1.0) -> np.ndarray:
    """Spherical variogram correlation function rho(h)."""
    distance_m = np.asarray(distance_m, dtype=float)
    hr = distance_m / range_m
    rho = np.where(distance_m <= range_m, 1.0 - 1.5 * hr + 0.5 * hr ** 3, 0.0)
    return np.clip(rho * correlation_scale, 0.0, 1.0)


def independent_sd(s_values: np.ndarray) -> float:
    """SD of a sum of independent errors: sqrt(sum of variances)."""
    s_values = np.asarray(s_values, dtype=float)
    if len(s_values) == 0:
        return 0.0
    return float(np.sqrt(np.sum(s_values ** 2)))


def spatial_sd(coords: np.ndarray, s_values: np.ndarray, range_m: float, correlation_scale: float, label: str) -> float:
    return block_spatial_sds(
        coords,
        {label: s_values},
        range_m,
        correlation_scale,
        label,
        OBJECT_SPATIAL_BLOCK_KM,
    )[label]


def aggregate_to_spatial_blocks(coords: np.ndarray, s_values_by_category: dict[str, np.ndarray], block_km: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Aggregate pixels onto a regular block grid.

    Each block is represented by its centroid, and the SD of a category
    within a block is the quadrature sum of its pixel SDs (errors inside a
    block are assumed independent; correlations between blocks are handled
    downstream by the spatial covariance term).
    """
    coords = np.asarray(coords, dtype=float)
    s_values_by_category = {
        key: np.asarray(values, dtype=float)
        for key, values in s_values_by_category.items()
    }
    if len(coords) == 0:
        return np.empty((0, 2), dtype=float), {key: np.array([], dtype=float) for key in s_values_by_category}

    block_m = block_km * 1000.0
    x0 = float(np.min(coords[:, 0]))
    y0 = float(np.min(coords[:, 1]))
    bx = np.floor((coords[:, 0] - x0) / block_m).astype(np.int64)
    by = np.floor((coords[:, 1] - y0) / block_m).astype(np.int64)
    block_keys, inverse = np.unique(np.column_stack([bx, by]), axis=0, return_inverse=True)
    n_blocks = len(block_keys)

    block_x_sum = np.bincount(inverse, weights=coords[:, 0], minlength=n_blocks)
    block_y_sum = np.bincount(inverse, weights=coords[:, 1], minlength=n_blocks)
    block_count = np.bincount(inverse, minlength=n_blocks).astype(float)
    block_coords = np.column_stack([block_x_sum / block_count, block_y_sum / block_count])

    block_s_values = {}
    for category, values in s_values_by_category.items():
        block_variance = np.bincount(inverse, weights=values ** 2, minlength=n_blocks)
        block_s_values[category] = np.sqrt(block_variance)

    return block_coords, block_s_values


def block_spatial_sds(coords: np.ndarray, s_values_by_category: dict[str, np.ndarray], range_m: float, correlation_scale: float, label: str, block_km: float | None) -> dict[str, float]:
    if block_km is None:
        return spatial_sds(coords, s_values_by_category, range_m, correlation_scale, label)

    block_coords, block_s_values = aggregate_to_spatial_blocks(coords, s_values_by_category, block_km)
    n_pixels = len(coords)
    n_blocks = len(block_coords)
    print(f"  Aggregated {label}: {n_pixels:,} pixels -> {n_blocks:,} blocks ({block_km:.1f} km)")
    return spatial_sds(block_coords, block_s_values, range_m, correlation_scale, f"{label} [{block_km:.1f} km blocks]")


def spatial_sds(coords: np.ndarray, s_values_by_category: dict[str, np.ndarray], range_m: float, correlation_scale: float, label: str) -> dict[str, float]:
    """SD of a sum of spatially correlated errors.

    For each category the total variance is
        Var = sum_i s_i^2 + 2 * sum_{i<j} rho(d_ij) * s_i * s_j,
    where rho is the spherical correlation function scaled by the
    nugget-adjusted correlation scale. Neighbor pairs are found with a
    KD-tree restricted to the variogram range, so cost scales with the
    number of pairs within the range rather than n^2.
    """
    coords = np.asarray(coords, dtype=float)
    s_values_by_category = {
        key: np.asarray(values, dtype=float)
        for key, values in s_values_by_category.items()
    }
    if not s_values_by_category:
        return {}

    first_key = next(iter(s_values_by_category))
    n = len(s_values_by_category[first_key])
    for key, values in s_values_by_category.items():
        if len(values) != n:
            raise ValueError(f"Length mismatch for {label} - {key}: {len(values)} != {n}")

    if n == 0:
        return {key: 0.0 for key in s_values_by_category}
    if n == 1:
        return {key: float(abs(values[0])) for key, values in s_values_by_category.items()}

    variance = {
        key: float(np.sum(values ** 2))
        for key, values in s_values_by_category.items()
    }
    covariance_sum = {key: 0.0 for key in s_values_by_category}
    tree = cKDTree(coords)

    print(f"  Spatial SD for {label}: {n:,} points, {len(s_values_by_category)} categories, range = {range_m / 1000:.2f} km")
    t0 = time.time()
    for start in range(0, n, SPATIAL_QUERY_CHUNK_SIZE):
        end = min(start + SPATIAL_QUERY_CHUNK_SIZE, n)
        neighbors = tree.query_ball_point(coords[start:end], r=range_m)
        for offset, neighbor_list in enumerate(neighbors):
            i = start + offset
            neighbor_arr = np.fromiter(neighbor_list, dtype=np.int64)
            js = neighbor_arr[neighbor_arr > i]
            if js.size == 0:
                continue
            d = np.sqrt(np.sum((coords[js] - coords[i]) ** 2, axis=1))
            rho = spherical_rho(d, range_m, correlation_scale)
            for key, values in s_values_by_category.items():
                covariance_sum[key] += float(np.sum(rho * values[i] * values[js]))
        if start == 0 or end == n or (start // SPATIAL_QUERY_CHUNK_SIZE) % 20 == 0:
            elapsed = time.time() - t0
            speed = end / elapsed if elapsed > 0 else np.nan
            remaining = (n - end) / speed if speed and speed > 0 else np.nan
            if np.isfinite(remaining):
                print(f"    processed {end:,}/{n:,} points, elapsed {elapsed/60:.1f} min, ETA {remaining/60:.1f} min")
            else:
                print(f"    processed {end:,}/{n:,} points")

    out = {}
    for key in s_values_by_category:
        total_variance = variance[key] + 2.0 * covariance_sum[key]
        if total_variance < 0 and abs(total_variance) < 1e-6:
            total_variance = 0.0
        if total_variance < 0:
            raise RuntimeError(f"Negative variance for {label} - {key}: {total_variance}")
        out[key] = float(np.sqrt(total_variance))
    return out


def safe_inflation(sd_spatial: float, sd_independent: float) -> float:
    if sd_independent == 0:
        return np.nan
    return float(sd_spatial / sd_independent)


def get_pixel_centers(transform, valid_mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(valid_mask)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    return np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])

# =========================================================
# PIXEL EXTRACTION
# =========================================================

def empty_pixel_data() -> dict:
    return {
        "coords": np.empty((0, 2), dtype=float),
        "agb_pixel": np.array([], dtype=float),
        "s_agb": np.array([], dtype=float),
        "s_bgb1": np.array([], dtype=float),
        "s_bgb2": np.array([], dtype=float),
        "s_bgb3": np.array([], dtype=float),
        "s_cagb": np.array([], dtype=float),
        "s_cbgb1": np.array([], dtype=float),
        "s_cbgb2": np.array([], dtype=float),
        "s_cbgb3": np.array([], dtype=float),
        "s_total_c1": np.array([], dtype=float),
        "s_total_c2": np.array([], dtype=float),
        "s_total_c3": np.array([], dtype=float),
        "coef1": np.array([], dtype=float),
    }


def concat_pixel_chunks(chunks: list[dict]) -> dict:
    if not chunks:
        return empty_pixel_data()
    keys = chunks[0].keys()
    out = {}
    for key in keys:
        out[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)
    return out


def collect_pixels_single(vector_gdf: gpd.GeoDataFrame, src_agb, src_sd) -> dict:
  
    if len(vector_gdf) == 0:
        return empty_pixel_data()

    geometries = [mapping(geom) for geom in vector_gdf.geometry if geom is not None and not geom.is_empty]
    if not geometries:
        return empty_pixel_data()

    agb_image, out_transform = mask(src_agb, geometries, crop=True, filled=False)
    sd_image, _ = mask(src_sd, geometries, crop=True, filled=False)

    agb = np.ma.filled(agb_image[0].astype(np.float32), np.nan)
    sd = np.ma.filled(sd_image[0].astype(np.float32), np.nan)
    valid = np.isfinite(agb) & np.isfinite(sd)
    if not np.any(valid):
        return empty_pixel_data()

    pixel_area_ha = abs(out_transform.a * out_transform.e) / 10000.0
    coords = get_pixel_centers(out_transform, valid)

    agb_valid = agb[valid].astype(float)
    sd_valid = sd[valid].astype(float)

    agb_pixel = agb_valid * pixel_area_ha
    s_agb = sd_valid * pixel_area_ha
    coef1 = np.where(agb_valid > AGB_THRESHOLD, BGB_RATIO_SCENARIO_1_LOW, BGB_RATIO_SCENARIO_1_HIGH)

    s_bgb1 = s_agb * coef1
    s_bgb2 = s_agb * BGB_RATIO_SCENARIO_2
    s_bgb3 = s_agb * BGB_RATIO_SCENARIO_3

    return {
        "coords": coords,
        "agb_pixel": agb_pixel,
        "s_agb": s_agb,
        "s_bgb1": s_bgb1,
        "s_bgb2": s_bgb2,
        "s_bgb3": s_bgb3,
        "s_cagb": s_agb * CARBON_FACTOR,
        "s_cbgb1": s_bgb1 * CARBON_FACTOR,
        "s_cbgb2": s_bgb2 * CARBON_FACTOR,
        "s_cbgb3": s_bgb3 * CARBON_FACTOR,
        "s_total_c1": s_agb * (1.0 + coef1) * CARBON_FACTOR,
        "s_total_c2": s_agb * (1.0 + BGB_RATIO_SCENARIO_2) * CARBON_FACTOR,
        "s_total_c3": s_agb * (1.0 + BGB_RATIO_SCENARIO_3) * CARBON_FACTOR,
        "coef1": coef1,
    }


def collect_pixels_chunked(vector_gdf: gpd.GeoDataFrame, src_agb, src_sd, chunk_size: int, label: str) -> dict:
    print(f"Collecting pixels for {label}: {len(vector_gdf):,} vector features")
    chunks = []
    for start in range(0, len(vector_gdf), chunk_size):
        end = min(start + chunk_size, len(vector_gdf))
        chunk = vector_gdf.iloc[start:end]
        data = collect_pixels_single(chunk, src_agb, src_sd)
        if len(data["s_agb"]) > 0:
            chunks.append(data)
        print(f"  features {end:,}/{len(vector_gdf):,}, pixels collected in chunk: {len(data['s_agb']):,}")
    out = concat_pixel_chunks(chunks)
    print(f"Total pixels for {label}: {len(out['s_agb']):,}")
    return out

# =========================================================
# SUMMARY BUILDING
# =========================================================

def summarize_object(
    object_name: str,
    data: dict,
    range_m: float,
    range_km: float,
    correlation_scale: float,
    compute_spatial: bool = True,
    spatial_block_km: float | None = OBJECT_SPATIAL_BLOCK_KM,
) -> list[dict]:
    agb = data["agb_pixel"]
    coef1 = data["coef1"]

    values = {
        "AGB": np.sum(agb),
        "BGB (Scenario 1)": np.sum(agb * coef1),
        "BGB (Scenario 2)": np.sum(agb * BGB_RATIO_SCENARIO_2),
        "BGB (Scenario 3)": np.sum(agb * BGB_RATIO_SCENARIO_3),
        "CAGB": np.sum(agb) * CARBON_FACTOR,
        "CBGB (Scenario 1)": np.sum(agb * coef1) * CARBON_FACTOR,
        "CBGB (Scenario 2)": np.sum(agb * BGB_RATIO_SCENARIO_2) * CARBON_FACTOR,
        "CBGB (Scenario 3)": np.sum(agb * BGB_RATIO_SCENARIO_3) * CARBON_FACTOR,
        "Total Carbon (Scenario 1)": np.sum((agb + agb * coef1) * CARBON_FACTOR),
        "Total Carbon (Scenario 2)": np.sum((agb + agb * BGB_RATIO_SCENARIO_2) * CARBON_FACTOR),
        "Total Carbon (Scenario 3)": np.sum((agb + agb * BGB_RATIO_SCENARIO_3) * CARBON_FACTOR),
    }

    s_map = {
        "AGB": data["s_agb"],
        "BGB (Scenario 1)": data["s_bgb1"],
        "BGB (Scenario 2)": data["s_bgb2"],
        "BGB (Scenario 3)": data["s_bgb3"],
        "CAGB": data["s_cagb"],
        "CBGB (Scenario 1)": data["s_cbgb1"],
        "CBGB (Scenario 2)": data["s_cbgb2"],
        "CBGB (Scenario 3)": data["s_cbgb3"],
        "Total Carbon (Scenario 1)": data["s_total_c1"],
        "Total Carbon (Scenario 2)": data["s_total_c2"],
        "Total Carbon (Scenario 3)": data["s_total_c3"],
    }

    rows = []
    coords = data["coords"]
    if compute_spatial:
        spatial_output = block_spatial_sds(coords, s_map, range_m, correlation_scale, object_name, spatial_block_km)
    else:
        spatial_output = {}

    for category, value in values.items():
        s_values = s_map[category]
        sd_ind = independent_sd(s_values)
        if compute_spatial:
            sd_sp = spatial_output[category]
        else:
            sd_sp = np.nan
        rows.append({
            "Object": object_name,
            "Category": category,
            "Value_Mg_or_MgC": float(value),
            "Value_Mt_or_MtC": float(value / 1e6),
            "SD_independent": sd_ind,
            "SD_independent_Mt": sd_ind / 1e6,
            "SD_spatial": sd_sp,
            "SD_spatial_Mt": sd_sp / 1e6 if np.isfinite(sd_sp) else np.nan,
            "inflation_factor": safe_inflation(sd_sp, sd_ind) if np.isfinite(sd_sp) else np.nan,
            "range_used_km": range_km if compute_spatial else np.nan,
            "spatial_block_km": spatial_block_km if compute_spatial else np.nan,
            "n_pixels": int(len(data["s_agb"])),
            "Formatted_independent": f"{value / 1e6:.2f} ± {sd_ind / 1e6:.2f}",
            "Formatted_spatial": f"{value / 1e6:.2f} ± {sd_sp / 1e6:.2f}" if np.isfinite(sd_sp) else "not_calculated",
        })
    return rows


def iter_windows(width: int, height: int, block_pixels: int):
    for row_off in range(0, height, block_pixels):
        win_height = min(block_pixels, height - row_off)
        for col_off in range(0, width, block_pixels):
            win_width = min(block_pixels, width - col_off)
            yield Window(col_off, row_off, win_width, win_height)


def summarize_whole_study_area(src_agb, src_sd, study_mask: np.ndarray, range_m: float, range_km: float, correlation_scale: float) -> list[dict]:
    """Whole-study-area totals using block-level aggregation.

    The raster is read in windows matching the block grid; per-window sums
    (values) and quadrature sums (SDs) become the per-block quantities that
    feed the spatial covariance calculation.
    """
    print("Summarizing whole study area background total...")
    pixel_area_ha = abs(src_agb.transform.a * src_agb.transform.e) / 10000.0
    pixel_size_m = math.sqrt(abs(src_agb.transform.a * src_agb.transform.e))
    block_pixels = max(1, int(round((WHOLE_STUDY_AREA_BLOCK_KM * 1000.0) / pixel_size_m)))
    print(f"Whole study area block aggregation: {WHOLE_STUDY_AREA_BLOCK_KM:.1f} km blocks ({block_pixels} pixels)")

    value_sums = {
        "agb_pixel": [],
        "bgb1_pixel": [],
        "bgb2_pixel": [],
        "bgb3_pixel": [],
    }
    variance_sums = {
        "s_agb": [],
        "s_bgb1": [],
        "s_bgb2": [],
        "s_bgb3": [],
        "s_cagb": [],
        "s_cbgb1": [],
        "s_cbgb2": [],
        "s_cbgb3": [],
        "s_total_c1": [],
        "s_total_c2": [],
        "s_total_c3": [],
    }
    coords = []
    total_valid = 0

    for window in iter_windows(src_agb.width, src_agb.height, block_pixels):
        agb = src_agb.read(1, window=window).astype(np.float32)
        sd = src_sd.read(1, window=window).astype(np.float32)
        window_transform = src_agb.window_transform(window)
        row_slice, col_slice = window.toslices()
        valid = study_mask[row_slice, col_slice] & np.isfinite(agb) & np.isfinite(sd)
        if not np.any(valid):
            continue

        rows_local, cols_local = np.where(valid)
        center_row = float(np.mean(rows_local))
        center_col = float(np.mean(cols_local))
        x, y = rasterio.transform.xy(window_transform, center_row, center_col, offset="center")
        coords.append((float(x), float(y)))

        agb_valid = agb[valid].astype(float)
        sd_valid = sd[valid].astype(float)
        s_agb = sd_valid * pixel_area_ha
        coef1 = np.where(agb_valid > AGB_THRESHOLD, BGB_RATIO_SCENARIO_1_LOW, BGB_RATIO_SCENARIO_1_HIGH)
        agb_pixel = agb_valid * pixel_area_ha

        value_sums["agb_pixel"].append(float(np.sum(agb_pixel)))
        value_sums["bgb1_pixel"].append(float(np.sum(agb_pixel * coef1)))
        value_sums["bgb2_pixel"].append(float(np.sum(agb_pixel * BGB_RATIO_SCENARIO_2)))
        value_sums["bgb3_pixel"].append(float(np.sum(agb_pixel * BGB_RATIO_SCENARIO_3)))

        variance_sums["s_agb"].append(float(np.sum(s_agb ** 2)))
        variance_sums["s_bgb1"].append(float(np.sum((s_agb * coef1) ** 2)))
        variance_sums["s_bgb2"].append(float(np.sum((s_agb * BGB_RATIO_SCENARIO_2) ** 2)))
        variance_sums["s_bgb3"].append(float(np.sum((s_agb * BGB_RATIO_SCENARIO_3) ** 2)))
        variance_sums["s_cagb"].append(float(np.sum((s_agb * CARBON_FACTOR) ** 2)))
        variance_sums["s_cbgb1"].append(float(np.sum((s_agb * coef1 * CARBON_FACTOR) ** 2)))
        variance_sums["s_cbgb2"].append(float(np.sum((s_agb * BGB_RATIO_SCENARIO_2 * CARBON_FACTOR) ** 2)))
        variance_sums["s_cbgb3"].append(float(np.sum((s_agb * BGB_RATIO_SCENARIO_3 * CARBON_FACTOR) ** 2)))
        variance_sums["s_total_c1"].append(float(np.sum((s_agb * (1.0 + coef1) * CARBON_FACTOR) ** 2)))
        variance_sums["s_total_c2"].append(float(np.sum((s_agb * (1.0 + BGB_RATIO_SCENARIO_2) * CARBON_FACTOR) ** 2)))
        variance_sums["s_total_c3"].append(float(np.sum((s_agb * (1.0 + BGB_RATIO_SCENARIO_3) * CARBON_FACTOR) ** 2)))
        total_valid += int(np.sum(valid))

    if not coords:
        return summarize_object(
            "Total biomass or carbon in the whole study area in 2010",
            empty_pixel_data(),
            range_m,
            range_km,
            correlation_scale,
            compute_spatial=False,
        )

    coords_arr = np.asarray(coords, dtype=float)
    print(f"Whole study area valid pixels: {total_valid:,}")
    print(f"Whole study area spatial covariance blocks: {len(coords_arr):,}")

    agb_blocks = np.asarray(value_sums["agb_pixel"], dtype=float)
    bgb1_blocks = np.asarray(value_sums["bgb1_pixel"], dtype=float)
    bgb2_blocks = np.asarray(value_sums["bgb2_pixel"], dtype=float)
    bgb3_blocks = np.asarray(value_sums["bgb3_pixel"], dtype=float)

    fake_data = {
        "coords": coords_arr,
        "agb_pixel": agb_blocks,
        "s_agb": np.sqrt(np.asarray(variance_sums["s_agb"], dtype=float)),
        "s_bgb1": np.sqrt(np.asarray(variance_sums["s_bgb1"], dtype=float)),
        "s_bgb2": np.sqrt(np.asarray(variance_sums["s_bgb2"], dtype=float)),
        "s_bgb3": np.sqrt(np.asarray(variance_sums["s_bgb3"], dtype=float)),
        "s_cagb": np.sqrt(np.asarray(variance_sums["s_cagb"], dtype=float)),
        "s_cbgb1": np.sqrt(np.asarray(variance_sums["s_cbgb1"], dtype=float)),
        "s_cbgb2": np.sqrt(np.asarray(variance_sums["s_cbgb2"], dtype=float)),
        "s_cbgb3": np.sqrt(np.asarray(variance_sums["s_cbgb3"], dtype=float)),
        "s_total_c1": np.sqrt(np.asarray(variance_sums["s_total_c1"], dtype=float)),
        "s_total_c2": np.sqrt(np.asarray(variance_sums["s_total_c2"], dtype=float)),
        "s_total_c3": np.sqrt(np.asarray(variance_sums["s_total_c3"], dtype=float)),
        "coef1": np.divide(bgb1_blocks, agb_blocks, out=np.zeros_like(bgb1_blocks), where=agb_blocks != 0),
    }
    fake_data["agb_pixel"] = agb_blocks
    rows = summarize_object(
        "Total biomass or carbon in the whole study area in 2010",
        fake_data,
        range_m,
        range_km,
        correlation_scale,
        compute_spatial=CALCULATE_WHOLE_STUDY_AREA_SPATIAL,
        spatial_block_km=None,
    )
    for row in rows:
        row["whole_area_spatial_method"] = f"{WHOLE_STUDY_AREA_BLOCK_KM:.1f} km block approximation"
        row["spatial_block_km"] = WHOLE_STUDY_AREA_BLOCK_KM
    return rows

# =========================================================
# MAIN
# =========================================================

def main() -> None:
    print("========================================")
    print("Direct biomass/carbon loss with spatially adjusted SD")
    print("========================================")

    for p in [REFERENCE_TIF, STUDY_AREA_SHP, ROAD_SHP, LOGGING_SHP, AGB2010_TIF, SD2010_TIF, PARAMS_JSON]:
        assert_exists(p)

    range_m, range_km, correlation_scale, params = load_variogram_params()
    print(f"Using shared variogram range: {range_km:.3f} km")
    print(f"Using nugget-adjusted correlation scale: {correlation_scale:.3f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BUFFER_DIR.mkdir(parents=True, exist_ok=True)

    reproject_match_reference(AGB2010_TIF, AGB_PROJECTED, REFERENCE_TIF)
    reproject_match_reference(SD2010_TIF, SD_PROJECTED, REFERENCE_TIF)

    with rasterio.open(REFERENCE_TIF) as ref:
        target_crs = ref.crs

    print("Reading vectors...")
    study_area = read_vector_projected(STUDY_AREA_SHP, target_crs)
    roads = read_vector_projected(ROAD_SHP, target_crs)
    logging = read_vector_projected(LOGGING_SHP, target_crs)
    study_area_mask_gdf = study_area[["geometry"]]
    roads = gpd.clip(roads, study_area_mask_gdf)
    logging = gpd.clip(logging, study_area_mask_gdf)
    roads_sindex = roads.sindex
    print(f"Road features inside study area: {len(roads):,}")
    print(f"Logging features inside study area: {len(logging):,}")

    with rasterio.open(AGB_PROJECTED) as src_agb, rasterio.open(SD_PROJECTED) as src_sd:
        study_mask = build_study_mask(src_agb, study_area)
        road_data = collect_pixels_chunked(roads, src_agb, src_sd, VECTOR_CHUNK_SIZE, "road expansion")
        logging_data = collect_pixels_chunked(logging, src_agb, src_sd, VECTOR_CHUNK_SIZE, "logging activities")

        all_rows = []
        all_rows.extend(summarize_object("Loss due to road expansion", road_data, range_m, range_km, correlation_scale, compute_spatial=True))
        all_rows.extend(summarize_object("Loss due to logging activities", logging_data, range_m, range_km, correlation_scale, compute_spatial=True))
        all_rows.extend(summarize_whole_study_area(src_agb, src_sd, study_mask, range_m, range_km, correlation_scale))

        result_df = pd.DataFrame(all_rows)
        result_df.to_csv(DETAIL_CSV, index=False, encoding="utf-8-sig")
        print(f"\nSaved spatial direct-loss table: {DETAIL_CSV}")

        print("\nComputing state-level road AGB loss with spatial SD...")
        state_name_col = STATE_NAME_COLUMN if STATE_NAME_COLUMN in study_area.columns else study_area.columns[0]
        state_rows = []
        for state_name in study_area[state_name_col].unique():
            print(f"State: {state_name}")
            state_poly = study_area[study_area[state_name_col] == state_name]
            state_geom = unary_union(state_poly.geometry)
            possible_idx = list(roads_sindex.intersection(state_geom.bounds))
            possible_roads = roads.iloc[possible_idx]
            roads_clip = gpd.clip(possible_roads, state_poly[["geometry"]])

            if len(roads_clip) == 0:
                state_rows.append({
                    "State": state_name,
                    "AGB_Mt": 0.0,
                    "AGB_SD_independent_Mt": 0.0,
                    "AGB_SD_spatial_Mt": 0.0,
                    "inflation_factor": np.nan,
                    "range_used_km": range_km,
                    "spatial_block_km": OBJECT_SPATIAL_BLOCK_KM,
                    "n_pixels": 0,
                })
                continue

            state_data = collect_pixels_chunked(roads_clip, src_agb, src_sd, STATE_VECTOR_CHUNK_SIZE, f"state {state_name}")
            agb_value = float(np.sum(state_data["agb_pixel"]))
            sd_ind = independent_sd(state_data["s_agb"])
            sd_sp = spatial_sd(state_data["coords"], state_data["s_agb"], range_m, correlation_scale, f"state {state_name} road AGB")
            state_rows.append({
                "State": state_name,
                "AGB_Mt": agb_value / 1e6,
                "AGB_SD_independent_Mt": sd_ind / 1e6,
                "AGB_SD_spatial_Mt": sd_sp / 1e6,
                "inflation_factor": safe_inflation(sd_sp, sd_ind),
                "range_used_km": range_km,
                "spatial_block_km": OBJECT_SPATIAL_BLOCK_KM,
                "n_pixels": int(len(state_data["s_agb"])),
            })

        state_df = pd.DataFrame(state_rows)
        state_df.to_csv(STATE_CSV, index=False, encoding="utf-8-sig")
        print(f"Saved state-level spatial table: {STATE_CSV}")
        print("\nSTATE-LEVEL ROAD AGB LOSS")
        print("-------------------------")
        print(state_df.to_string(index=False))

    print("\nFINAL RESULTS: DIRECT LOSS, BIOMASS, AND CARBON")
    print("------------------------------------------------")
    display_df = result_df.copy()
    display_df["Independent"] = display_df.apply(
        lambda r: f"{r['Value_Mt_or_MtC']:.2f} ± {r['SD_independent_Mt']:.2f}",
        axis=1,
    )
    display_df["Spatial"] = display_df.apply(
        lambda r: "not_calculated" if not np.isfinite(r["SD_spatial_Mt"]) else f"{r['Value_Mt_or_MtC']:.2f} ± {r['SD_spatial_Mt']:.2f}",
        axis=1,
    )
    display_df["Inflation"] = display_df["inflation_factor"].apply(
        lambda v: "not_calculated" if not np.isfinite(v) else f"{v:.2f}"
    )

    for obj in [
        "Loss due to road expansion",
        "Loss due to logging activities",
        "Total biomass or carbon in the whole study area in 2010",
    ]:
        sub = display_df[display_df["Object"] == obj][[
            "Category",
            "Independent",
            "Spatial",
            "Inflation",
            "range_used_km",
            "spatial_block_km",
        ]]
        if len(sub) == 0:
            continue
        print(f"\n{obj}")
        print(sub.to_string(index=False))

    print("\nAGB QUICK SUMMARY")
    print("-----------------")
    for obj in ["Loss due to road expansion", "Loss due to logging activities"]:
        sub = result_df[(result_df["Object"] == obj) & (result_df["Category"] == "AGB")]
        if len(sub) == 1:
            row = sub.iloc[0]
            print(
                f"{obj}: AGB = {row['Value_Mt_or_MtC']:.2f} Mt, "
                f"SD_ind = {row['SD_independent_Mt']:.4f} Mt, "
                f"SD_spatial = {row['SD_spatial_Mt']:.4f} Mt, "
                f"inflation = {row['inflation_factor']:.2f}, "
                f"range = {row['range_used_km']:.2f} km"
            )
    print("\nDONE")


if __name__ == "__main__":
    main()
