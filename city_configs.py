"""
City-specific configuration for the building height validation workflow.

Set BASE_DIR to the root of your local data directory before running.
Expected subdirectory structure:
    data/raw/lidar/
    data/raw/gba/
    data/raw/globfp/
    data/processed/
    data/results/
"""

from pathlib import Path

BASE_DIR = Path(r"D:\building-height-validation")
DATA_DIR = BASE_DIR / "data"
LIDAR_DIR = DATA_DIR / "raw" / "lidar"
GBA_DIR = DATA_DIR / "raw" / "gba"
GLOBFP_DIR = DATA_DIR / "raw" / "globfp"
OUTPUT_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CITIES = {
    'amsterdam': {
        'name': 'Amsterdam',
        'crs': 'EPSG:28992',
        'has_ndsm': False,
        'dsm_path': LIDAR_DIR / "amsterdam_dsm.tif",
        'dtm_path': LIDAR_DIR / "amsterdam_dtm.tif",
        'ndsm_path': OUTPUT_DIR / "amsterdam_ndsm.tif",
        'gba_path': GBA_DIR / "amsterdam_gba.gpkg",
        'globfp_path': GLOBFP_DIR / "amsterdam_globfp.gpkg",
        'gba_height_col': 'height',
        'globfp_height_col': 'Height',
        'lidar_resolution': 0.5,
        'lidar_source': 'AHN4',
        'lidar_year': '2020-2022',
        'lidar_accuracy': '0.2-0.5 m RMSE',
    },

    'paris': {
        'name': 'Paris',
        'crs': 'EPSG:2154',
        'has_ndsm': True,
        'ndsm_path': LIDAR_DIR / "paris_mnh.tif",
        'gba_path': GBA_DIR / "paris_gba.gpkg",
        'globfp_path': GLOBFP_DIR / "paris_globfp.gpkg",
        'gba_height_col': 'height',
        'globfp_height_col': 'Height',
        'lidar_resolution': 0.5,
        'lidar_source': 'IGN LiDAR HD',
        'lidar_year': '2021-2023',
        'lidar_accuracy': '0.3-0.8 m RMSE',
    },

    'toronto': {
        'name': 'Toronto',
        'crs': 'EPSG:26917',
        'has_ndsm': False,
        'dsm_path': LIDAR_DIR / "toronto_dsm.tif",
        'dtm_path': LIDAR_DIR / "toronto_dtm.tif",
        'ndsm_path': OUTPUT_DIR / "toronto_ndsm.tif",
        'gba_path': GBA_DIR / "toronto_gba.gpkg",
        'globfp_path': GLOBFP_DIR / "toronto_globfp.gpkg",
        'gba_height_col': 'height',
        'globfp_height_col': 'Height',
        'lidar_resolution': 0.5,
        'lidar_source': 'Ontario GTA LiDAR',
        'lidar_year': '2014-2018',
        'lidar_accuracy': '0.3-0.8 m RMSE',
    },

    'hongkong': {
        'name': 'Hong Kong',
        'crs': 'EPSG:2326',
        'has_ndsm': False,
        'dsm_path': LIDAR_DIR / "hongkong_dsm.tif",
        'dtm_path': LIDAR_DIR / "hongkong_dtm.tif",
        'ndsm_path': OUTPUT_DIR / "hongkong_ndsm.tif",
        'gba_path': GBA_DIR / "hongkong_gba.gpkg",
        'globfp_path': GLOBFP_DIR / "hongkong_globfp.gpkg",
        'gba_height_col': 'height',
        'globfp_height_col': 'Height',
        'lidar_resolution': 0.5,
        'lidar_source': 'CEDD LiDAR',
        'lidar_year': '2019-2020',
        'lidar_accuracy': '0.3-0.8 m RMSE',
    },
}

VALIDATION_PARAMS = {
    'min_lidar_height': 2.0,    # metres
    'min_pixel_count': 10,
    'nodata_value': -9999.0,
    'height_classes': {
        'low_rise':  (0,  12),
        'mid_rise':  (12, 25),
        'high_rise': (25, float('inf')),
    },
}
