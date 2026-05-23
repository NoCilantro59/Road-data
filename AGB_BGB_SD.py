# =========================================================
# Core functions for biomass and carbon loss estimation
# =========================================================

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from shapely.geometry import mapping
from shapely.ops import unary_union



def reproject_to_target(
        input_raster,
        output_raster,
        target_crs,
        target_transform,
        target_width,
        target_height,
        resampling=Resampling.bilinear):

    with rasterio.open(input_raster) as src:

        kwargs = src.meta.copy()

        kwargs.update({
            'crs': target_crs,
            'transform': target_transform,
            'width': target_width,
            'height': target_height,
            'dtype': 'float32',
            'nodata': np.nan
        })

        with rasterio.open(output_raster, 'w', **kwargs) as dst:

            for i in range(1, src.count + 1):

                dest_array = np.full(
                    (target_height, target_width),
                    np.nan,
                    dtype=np.float32
                )

                reproject(
                    source=rasterio.band(src, i),
                    destination=dest_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                    src_nodata=src.nodata,
                    dst_nodata=np.nan
                )

                dst.write(dest_array, i)



study_area = gpd.read_file(study_area_shp).to_crs(target_crs)

roads = gpd.read_file(road_shp).to_crs(target_crs)

logging = gpd.read_file(logging_shp).to_crs(target_crs)


roads_sindex = roads.sindex



agb_proj = os.path.join(output_folder, "AGB_projected.tif")

sd_proj = os.path.join(output_folder, "SD_projected.tif")

reproject_to_target(
    agb_tif,
    agb_proj,
    target_crs,
    dst_transform,
    dst_width,
    dst_height
)

reproject_to_target(
    sd_tif,
    sd_proj,
    target_crs,
    dst_transform,
    dst_width,
    dst_height
)

src_agb = rasterio.open(agb_proj)

src_sd = rasterio.open(sd_proj)



def calculate_loss_single(vector_gdf):

    geometries = [mapping(geom) for geom in vector_gdf.geometry]

    # -------------------------
    # mask
    # -------------------------

    agb_image, agb_transform = mask(
        src_agb,
        geometries,
        crop=True,
        filled=False
    )

    sd_image, _ = mask(
        src_sd,
        geometries,
        crop=True,
        filled=False
    )

    agb_array = np.ma.filled(
        agb_image[0].astype(np.float32),
        np.nan
    )

    sd_array = np.ma.filled(
        sd_image[0].astype(np.float32),
        np.nan
    )

    # -------------------------
    # pixel area
    # -------------------------

    pixel_width = abs(agb_transform.a)

    pixel_height = abs(agb_transform.e)

    pixel_area_ha = (
            pixel_width * pixel_height
    ) / 10000.0

    # -------------------------
    # valid mask
    # -------------------------

    valid_mask = ~(
            np.isnan(agb_array) |
            np.isnan(sd_array)
    )

    valid_count = np.sum(valid_mask)

    print(f"valid pixels: {valid_count}")

    if valid_count == 0:

        return {
            "AGB": 0.0,
            "AGB_SD": 0.0,

            "BGB1": 0.0,
            "BGB1_SD": 0.0,

            "BGB2": 0.0,
            "BGB2_SD": 0.0,

            "BGB3": 0.0,
            "BGB3_SD": 0.0
        }

    agb_valid = agb_array[valid_mask]

    sd_valid = sd_array[valid_mask]

    # =====================================================
    # AGB
    # =====================================================

    agb_pixel = agb_valid * pixel_area_ha

    sd_pixel = sd_valid * pixel_area_ha

    total_agb = np.sum(agb_pixel)

    total_agb_sd = np.sqrt(
        np.sum(sd_pixel ** 2)
    )

    # =====================================================
    # BGB
    # =====================================================

    coef1 = np.where(
        agb_valid > 75,
        0.24,
        0.39
    )

    bgb1_pixel = agb_pixel * coef1

    bgb2_pixel = agb_pixel * 0.75

    bgb3_pixel = agb_pixel * 0.43

    total_bgb1 = np.sum(bgb1_pixel)

    total_bgb2 = np.sum(bgb2_pixel)

    total_bgb3 = np.sum(bgb3_pixel)

    # =====================================================
    # SD propagation
    # =====================================================

    bgb1_sd = np.sqrt(
        np.sum(
            (sd_pixel * coef1) ** 2
        )
    )

    bgb2_sd = np.sqrt(
        np.sum(
            (sd_pixel * 0.75) ** 2
        )
    )

    bgb3_sd = np.sqrt(
        np.sum(
            (sd_pixel * 0.43) ** 2
        )
    )

    return {

        "AGB": total_agb,
        "AGB_SD": total_agb_sd,

        "BGB1": total_bgb1,
        "BGB1_SD": bgb1_sd,

        "BGB2": total_bgb2,
        "BGB2_SD": bgb2_sd,

        "BGB3": total_bgb3,
        "BGB3_SD": bgb3_sd
    }


def calc_loss_chunked(vector_gdf, chunk_size=3000):

    n_features = len(vector_gdf)

    n_chunks = (n_features + chunk_size - 1) // chunk_size

    total = {

        "AGB": 0.0,
        "AGB_SD_SQ": 0.0,

        "BGB1": 0.0,
        "BGB1_SD_SQ": 0.0,

        "BGB2": 0.0,
        "BGB2_SD_SQ": 0.0,

        "BGB3": 0.0,
        "BGB3_SD_SQ": 0.0
    }

    for i in range(0, n_features, chunk_size):

        chunk = vector_gdf.iloc[i:i + chunk_size]

        result = calculate_loss_single(chunk)

        total["AGB"] += result["AGB"]

        total["AGB_SD_SQ"] += result["AGB_SD"] ** 2

        total["BGB1"] += result["BGB1"]

        total["BGB1_SD_SQ"] += result["BGB1_SD"] ** 2

        total["BGB2"] += result["BGB2"]

        total["BGB2_SD_SQ"] += result["BGB2_SD"] ** 2

        total["BGB3"] += result["BGB3"]

        total["BGB3_SD_SQ"] += result["BGB3_SD"] ** 2

    return {

        "AGB": total["AGB"],

        "AGB_SD": np.sqrt(
            total["AGB_SD_SQ"]
        ),

        "BGB1": total["BGB1"],

        "BGB1_SD": np.sqrt(
            total["BGB1_SD_SQ"]
        ),

        "BGB2": total["BGB2"],

        "BGB2_SD": np.sqrt(
            total["BGB2_SD_SQ"]
        ),

        "BGB3": total["BGB3"],

        "BGB3_SD": np.sqrt(
            total["BGB3_SD_SQ"]
        )
    }



print("\nSTEP 5: road_loss")

road_res = calc_loss_chunked(
    roads,
    chunk_size=3000
)



print("\nSTEP 6: logging_loss")

logging_res = calc_loss_chunked(
    logging,
    chunk_size=3000
)


print("\nSTEP 7: total AGB")

agb_data = src_agb.read(1).astype(np.float32)

sd_data = src_sd.read(1).astype(np.float32)

valid = ~(
        np.isnan(agb_data) |
        np.isnan(sd_data)
)

agb_valid = agb_data[valid]

sd_valid = sd_data[valid]

pixel_area_ha = abs(
    src_agb.transform.a *
    src_agb.transform.e
) / 10000.0

agb_pixel = agb_valid * pixel_area_ha

sd_pixel = sd_valid * pixel_area_ha

# -------------------------
# AGB
# -------------------------

whole_agb = np.sum(agb_pixel)

whole_agb_sd = np.sqrt(
    np.sum(sd_pixel ** 2)
)

# -------------------------
# BGB
# -------------------------

coef1 = np.where(
    agb_valid > 75,
    0.24,
    0.39
)

whole_bgb1 = np.sum(
    agb_pixel * coef1
)

whole_bgb1_sd = np.sqrt(
    np.sum(
        (sd_pixel * coef1) ** 2
    )
)

whole_bgb2 = np.sum(
    agb_pixel * 0.75
)

whole_bgb2_sd = np.sqrt(
    np.sum(
        (sd_pixel * 0.75) ** 2
    )
)

whole_bgb3 = np.sum(
    agb_pixel * 0.43
)

whole_bgb3_sd = np.sqrt(
    np.sum(
        (sd_pixel * 0.43) ** 2
    )
)

whole_res = {

    "AGB": whole_agb,
    "AGB_SD": whole_agb_sd,

    "BGB1": whole_bgb1,
    "BGB1_SD": whole_bgb1_sd,

    "BGB2": whole_bgb2,
    "BGB2_SD": whole_bgb2_sd,

    "BGB3": whole_bgb3,
    "BGB3_SD": whole_bgb3_sd
}



print("\nSTEP 8: road_loss_stateLevel")

state_results = []

for state_name in study_area["name2"].unique():

    print(f"处理州: {state_name}")

    state_poly = study_area[
        study_area["name2"] == state_name
        ]

    state_geom = unary_union(state_poly.geometry)

    possible_matches_index = list(
        roads_sindex.intersection(
            state_geom.bounds
        )
    )

    possible_roads = roads.iloc[
        possible_matches_index
    ]

    roads_clip = possible_roads[
        possible_roads.intersects(state_geom)
    ]

    if len(roads_clip) == 0:

        state_results.append({
            "State": state_name,
            "AGB_Mt": 0,
            "AGB_SD_Mt": 0
        })

        continue

    state_res = calc_loss_chunked(
        roads_clip,
        chunk_size=1000
    )

    state_results.append({

        "State": state_name,

        "AGB_Mt": state_res["AGB"] / 1e6,

        "AGB_SD_Mt": state_res["AGB_SD"] / 1e6
    })

state_df = pd.DataFrame(state_results)

state_csv = os.path.join(
    output_folder,
    "STATE_ROAD_AGB_LOSS.csv"
)

state_df.to_csv(
    state_csv,
    index=False
)



def format_val(val, sd):

    return f"{val/1e6:.2f} ± {sd/1e6:.2f}"

def total_carbon(
        agb,
        agb_sd,
        bgb,
        bgb_sd):

    total = (
            agb + bgb
    ) * CARBON_FACTOR

    total_sd = CARBON_FACTOR * np.sqrt(
        agb_sd ** 2 +
        bgb_sd ** 2
    )

    return total, total_sd

summary_data = {

    "Category": [

        "AGB",

        "BGB (Scenario 1)",
        "BGB (Scenario 2)",
        "BGB (Scenario 3)",

        "CAGB",

        "CBGB (Scenario 1)",
        "CBGB (Scenario 2)",
        "CBGB (Scenario 3)",

        "Total Carbon (Scenario 1)",
        "Total Carbon (Scenario 2)",
        "Total Carbon (Scenario 3)"
    ],

    "Loss due to road expansion": [

        format_val(
            road_res["AGB"],
            road_res["AGB_SD"]
        ),

        format_val(
            road_res["BGB1"],
            road_res["BGB1_SD"]
        ),

        format_val(
            road_res["BGB2"],
            road_res["BGB2_SD"]
        ),

        format_val(
            road_res["BGB3"],
            road_res["BGB3_SD"]
        ),

        format_val(
            road_res["AGB"] * CARBON_FACTOR,
            road_res["AGB_SD"] * CARBON_FACTOR
        ),

        format_val(
            road_res["BGB1"] * CARBON_FACTOR,
            road_res["BGB1_SD"] * CARBON_FACTOR
        ),

        format_val(
            road_res["BGB2"] * CARBON_FACTOR,
            road_res["BGB2_SD"] * CARBON_FACTOR
        ),

        format_val(
            road_res["BGB3"] * CARBON_FACTOR,
            road_res["BGB3_SD"] * CARBON_FACTOR
        ),

        format_val(
            *total_carbon(
                road_res["AGB"],
                road_res["AGB_SD"],
                road_res["BGB1"],
                road_res["BGB1_SD"]
            )
        ),

        format_val(
            *total_carbon(
                road_res["AGB"],
                road_res["AGB_SD"],
                road_res["BGB2"],
                road_res["BGB2_SD"]
            )
        ),

        format_val(
            *total_carbon(
                road_res["AGB"],
                road_res["AGB_SD"],
                road_res["BGB3"],
                road_res["BGB3_SD"]
            )
        )
    ],

    "Loss due to logging activities": [

        format_val(
            logging_res["AGB"],
            logging_res["AGB_SD"]
        ),

        format_val(
            logging_res["BGB1"],
            logging_res["BGB1_SD"]
        ),

        format_val(
            logging_res["BGB2"],
            logging_res["BGB2_SD"]
        ),

        format_val(
            logging_res["BGB3"],
            logging_res["BGB3_SD"]
        ),

        format_val(
            logging_res["AGB"] * CARBON_FACTOR,
            logging_res["AGB_SD"] * CARBON_FACTOR
        ),

        format_val(
            logging_res["BGB1"] * CARBON_FACTOR,
            logging_res["BGB1_SD"] * CARBON_FACTOR
        ),

        format_val(
            logging_res["BGB2"] * CARBON_FACTOR,
            logging_res["BGB2_SD"] * CARBON_FACTOR
        ),

        format_val(
            logging_res["BGB3"] * CARBON_FACTOR,
            logging_res["BGB3_SD"] * CARBON_FACTOR
        ),

        format_val(
            *total_carbon(
                logging_res["AGB"],
                logging_res["AGB_SD"],
                logging_res["BGB1"],
                logging_res["BGB1_SD"]
            )
        ),

        format_val(
            *total_carbon(
                logging_res["AGB"],
                logging_res["AGB_SD"],
                logging_res["BGB2"],
                logging_res["BGB2_SD"]
            )
        ),

        format_val(
            *total_carbon(
                logging_res["AGB"],
                logging_res["AGB_SD"],
                logging_res["BGB3"],
                logging_res["BGB3_SD"]
            )
        )
    ],

    "Total biomass or carbon in the whole study area in 2010": [

        format_val(
            whole_res["AGB"],
            whole_res["AGB_SD"]
        ),

        format_val(
            whole_res["BGB1"],
            whole_res["BGB1_SD"]
        ),

        format_val(
            whole_res["BGB2"],
            whole_res["BGB2_SD"]
        ),

        format_val(
            whole_res["BGB3"],
            whole_res["BGB3_SD"]
        ),

        format_val(
            whole_res["AGB"] * CARBON_FACTOR,
            whole_res["AGB_SD"] * CARBON_FACTOR
        ),

        format_val(
            whole_res["BGB1"] * CARBON_FACTOR,
            whole_res["BGB1_SD"] * CARBON_FACTOR
        ),

        format_val(
            whole_res["BGB2"] * CARBON_FACTOR,
            whole_res["BGB2_SD"] * CARBON_FACTOR
        ),

        format_val(
            whole_res["BGB3"] * CARBON_FACTOR,
            whole_res["BGB3_SD"] * CARBON_FACTOR
        ),

        format_val(
            *total_carbon(
                whole_res["AGB"],
                whole_res["AGB_SD"],
                whole_res["BGB1"],
                whole_res["BGB1_SD"]
            )
        ),

        format_val(
            *total_carbon(
                whole_res["AGB"],
                whole_res["AGB_SD"],
                whole_res["BGB2"],
                whole_res["BGB2_SD"]
            )
        ),

        format_val(
            *total_carbon(
                whole_res["AGB"],
                whole_res["AGB_SD"],
                whole_res["BGB3"],
                whole_res["BGB3_SD"]
            )
        )
    ]
}

summary_df = pd.DataFrame(summary_data)

summary_csv = os.path.join(
    output_folder,
    "TABLE_S8.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False
)

print(f"\nsummary_csv:\n{summary_csv}")
