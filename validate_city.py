"""
Building height validation: GBA and 3D-GloBFP against LiDAR reference data.

For cities where a pre-computed nDSM is unavailable, the script derives one
from DSM and DTM inputs. Validation metrics are computed overall and stratified
by height class (low-rise / mid-rise / high-rise).

Usage:
    python validate_city.py --city <city_key>

City keys are defined in city_configs.py.

Note: If rasterio cannot locate PROJ data on your system, set the PROJ_LIB
environment variable before running:
    export PROJ_LIB=/path/to/proj/data   (Linux/macOS)
    set PROJ_LIB=C:\path\to\proj\data    (Windows)
"""

import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterstats import zonal_stats

from city_configs import CITIES, VALIDATION_PARAMS, OUTPUT_DIR, RESULTS_DIR

NODATA = VALIDATION_PARAMS['nodata_value']


def create_ndsm(city_config):
    """Derive a normalised DSM by subtracting DTM from DSM.

    Pixels with nDSM <= 0 are masked as NoData. The result is written to
    city_config['ndsm_path'] and that path is returned.
    """
    ndsm_path = city_config['ndsm_path']

    if ndsm_path.exists():
        print(f"nDSM already exists: {ndsm_path.name}")
        return ndsm_path

    print(f"\nCreating nDSM for {city_config['name']} ...")

    with rasterio.open(city_config['dsm_path']) as dsm_src, \
         rasterio.open(city_config['dtm_path']) as dtm_src:

        dsm = dsm_src.read(1).astype("float32")
        dtm = dtm_src.read(1).astype("float32")
        profile = dsm_src.profile.copy()

        dsm_nodata = dsm_src.nodata if dsm_src.nodata is not None else NODATA
        dtm_nodata = dtm_src.nodata if dtm_src.nodata is not None else NODATA

        dsm[dsm == dsm_nodata] = np.nan
        dtm[dtm == dtm_nodata] = np.nan

        ndsm = dsm - dtm
        ndsm[ndsm <= 0] = np.nan
        valid = ~np.isnan(ndsm)

        print(f"  Valid pixels : {valid.sum():,}")
        print(f"  Height range : {np.nanmin(ndsm):.2f} – {np.nanmax(ndsm):.2f} m")
        print(f"  Mean height  : {np.nanmean(ndsm):.2f} m")

        crs_code = int(city_config['crs'].split(':')[1])
        profile.update(dtype="float32", nodata=NODATA, crs=CRS.from_epsg(crs_code))
        ndsm_out = np.where(valid, ndsm, NODATA).astype("float32")

        with rasterio.open(ndsm_path, "w", **profile) as dst:
            dst.write(ndsm_out, 1)

    print(f"  Saved: {ndsm_path.name}")
    return ndsm_path


def calculate_stratified_metrics(gdf_valid, height_column='height'):
    """Compute error metrics for low-rise, mid-rise, and high-rise subsets.

    Height classes are defined in VALIDATION_PARAMS and assigned using the
    LiDAR p95 height as the reference.

    Returns (list of per-class metric dicts, gdf_valid with 'height_class' column).
    """
    height_classes = VALIDATION_PARAMS['height_classes']

    bins = [
        height_classes['low_rise'][0],
        height_classes['low_rise'][1],
        height_classes['mid_rise'][1],
        height_classes['high_rise'][1],
    ]
    labels = ['low_rise', 'mid_rise', 'high_rise']

    gdf_valid['height_class'] = pd.cut(
        gdf_valid['lidar_p95'],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    stratified_results = []

    for height_class in labels:
        subset = gdf_valid[gdf_valid['height_class'] == height_class]
        if len(subset) == 0:
            continue

        median_error = np.median(subset['error'])

        stratified_results.append({
            'height_class': height_class,
            'n': len(subset),
            'lidar_mean':    subset['lidar_p95'].mean(),
            'dataset_mean':  subset[height_column].mean(),
            'mbe':           subset['error'].mean(),
            'mae':           subset['abs_error'].mean(),
            'rmse':          np.sqrt((subset['error'] ** 2).mean()),
            'std':           subset['error'].std(),
            'nmad':          1.4826 * np.median(np.abs(subset['error'] - median_error)),
            'within_3m':     (subset['abs_error'] <= 3).mean(),
            'within_5m':     (subset['abs_error'] <= 5).mean(),
        })

    return stratified_results, gdf_valid


def save_error_distribution(gdf_valid, city_name, dataset_name):
    """Return a dict of error distribution percentiles for downstream plotting."""
    return {
        'city':    city_name,
        'dataset': dataset_name,
        'n':       len(gdf_valid),
        'min':     gdf_valid['error'].min(),
        'p5':      np.percentile(gdf_valid['error'],  5),
        'p10':     np.percentile(gdf_valid['error'], 10),
        'p25':     np.percentile(gdf_valid['error'], 25),
        'p50':     np.percentile(gdf_valid['error'], 50),
        'p75':     np.percentile(gdf_valid['error'], 75),
        'p90':     np.percentile(gdf_valid['error'], 90),
        'p95':     np.percentile(gdf_valid['error'], 95),
        'max':     gdf_valid['error'].max(),
        'mean':    gdf_valid['error'].mean(),
        'std':     gdf_valid['error'].std(),
    }


def save_scatter_sample(gdf_valid, city_name, dataset_name):
    """Write a CSV of up to 10,000 buildings for scatter plot reproducibility."""
    n_sample = min(10_000, len(gdf_valid))
    sample = (gdf_valid.sample(n=n_sample, random_state=42)
              if len(gdf_valid) > 10_000 else gdf_valid.copy())

    scatter_data = sample[
        ['lidar_p95', 'height', 'error', 'abs_error', 'height_class', 'area_m2']
    ].copy()

    scatter_path = OUTPUT_DIR / f'{city_name}_{dataset_name}_scatter_sample.csv'
    scatter_data.to_csv(scatter_path, index=False)
    return scatter_path


def validate_dataset(city_name, dataset_name, dataset_path, height_column, ndsm_path, city_config):
    """Validate one building-height dataset against the LiDAR nDSM.

    Steps:
      1. Load vector dataset and reproject to city CRS.
      2. Clip to nDSM extent.
      3. Extract LiDAR p95, mean, max, and pixel count per building footprint.
      4. Filter buildings: LiDAR p95 >= min_lidar_height, pixel_count >= min_pixel_count,
         dataset height > 0.
      5. Compute per-building errors and aggregate metrics.
      6. Stratify metrics by height class.
      7. Save validated GeoPackage, scatter sample CSV, and error distribution.

    Returns (overall_metrics dict, stratified_results list, error_dist dict),
    or (None, None, None) on failure.
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING: {dataset_name.upper()}  |  {city_name.upper()}")
    print(f"{'='*60}")

    gdf = gpd.read_file(dataset_path)
    print(f"Buildings loaded    : {len(gdf):,}")

    if height_column not in gdf.columns:
        print(f"Height column '{height_column}' not found in {dataset_name}.")
        return None, None, None

    if str(gdf.crs) != city_config['crs']:
        gdf = gdf.to_crs(city_config['crs'])

    with rasterio.open(ndsm_path) as src:
        ndsm_bounds = src.bounds

    gdf = gdf.cx[
        ndsm_bounds.left:ndsm_bounds.right,
        ndsm_bounds.bottom:ndsm_bounds.top,
    ]
    print(f"Within nDSM extent  : {len(gdf):,}")

    stats = zonal_stats(
        gdf.geometry,
        str(ndsm_path),
        stats=["percentile_95", "mean", "max", "count"],
        nodata=NODATA,
        all_touched=True,
    )

    gdf["lidar_p95"]    = [s["percentile_95"] if s else np.nan for s in stats]
    gdf["lidar_mean"]   = [s["mean"]          if s else np.nan for s in stats]
    gdf["lidar_max"]    = [s["max"]            if s else np.nan for s in stats]
    gdf["pixel_count"]  = [s["count"]          if s else 0      for s in stats]

    min_height = VALIDATION_PARAMS['min_lidar_height']
    min_pixels = VALIDATION_PARAMS['min_pixel_count']

    valid_mask = (
        (gdf["lidar_p95"]   >= min_height) &
        (gdf["pixel_count"] >= min_pixels) &
        (gdf[height_column]  > 0)
    )
    gdf_valid = gdf[valid_mask].copy()
    print(f"Valid buildings     : {len(gdf_valid):,} ({len(gdf_valid)/len(gdf)*100:.1f}%)")

    if len(gdf_valid) == 0:
        print("No valid buildings after filtering.")
        return None, None, None

    gdf_valid = gdf_valid.rename(columns={height_column: 'height'})

    gdf_valid['error']     = gdf_valid['height'] - gdf_valid['lidar_p95']
    gdf_valid['abs_error'] = gdf_valid['error'].abs()
    gdf_valid['rel_error'] = gdf_valid['error'] / gdf_valid['lidar_p95']
    gdf_valid['area_m2']   = gdf_valid.geometry.area

    mbe  = gdf_valid['error'].mean()
    mae  = gdf_valid['abs_error'].mean()
    rmse = np.sqrt((gdf_valid['error'] ** 2).mean())
    std  = gdf_valid['error'].std()

    median_error = np.median(gdf_valid['error'])
    nmad = 1.4826 * np.median(np.abs(gdf_valid['error'] - median_error))

    within_3m = (gdf_valid['abs_error'] <= 3).mean()
    within_5m = (gdf_valid['abs_error'] <= 5).mean()

    stratified_results, gdf_valid = calculate_stratified_metrics(gdf_valid, 'height')

    error_dist  = save_error_distribution(gdf_valid, city_name, dataset_name)
    scatter_path = save_scatter_sample(gdf_valid, city_name, dataset_name)

    print(f"\n{'='*60}")
    print(f"RESULTS  {city_name.upper()}  {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"N validated : {len(gdf_valid):,}")
    print(f"\nError metrics:")
    print(f"  MBE  : {mbe:6.2f} m")
    print(f"  MAE  : {mae:6.2f} m")
    print(f"  RMSE : {rmse:6.2f} m")
    print(f"  SD   : {std:6.2f} m")
    print(f"  NMAD : {nmad:6.2f} m")
    print(f"\nWithin thresholds:")
    print(f"  <= 3 m : {within_3m*100:5.1f}%")
    print(f"  <= 5 m : {within_5m*100:5.1f}%")
    print(f"\nMean heights:")
    print(f"  Dataset : {gdf_valid['height'].mean():6.2f} m")
    print(f"  LiDAR   : {gdf_valid['lidar_p95'].mean():6.2f} m")

    print(f"\nStratified by height class:")
    print(f"{'Class':<12} {'N':>8} {'RMSE':>8} {'MBE':>8} {'<=3m':>8}")
    print("-" * 50)
    for s in stratified_results:
        print(f"{s['height_class']:<12} "
              f"{s['n']:>8,} "
              f"{s['rmse']:>8.2f} "
              f"{s['mbe']:>8.2f} "
              f"{s['within_3m']*100:>7.1f}%")

    output_path = OUTPUT_DIR / f"{city_name}_{dataset_name}_validated.gpkg"
    gdf_valid.to_file(output_path, driver='GPKG')
    print(f"\nSaved: {output_path.name}")
    print(f"Saved: {scatter_path.name}")

    overall_metrics = {
        'city':                 city_name,
        'dataset':              dataset_name,
        'n':                    len(gdf_valid),
        'n_raw':                len(gdf),
        'completeness_pct':     len(gdf_valid) / len(gdf) * 100,
        'mbe':                  mbe,
        'mae':                  mae,
        'rmse':                 rmse,
        'std':                  std,
        'nmad':                 nmad,
        'within_3m':            within_3m,
        'within_5m':            within_5m,
        'dataset_mean_height':  gdf_valid['height'].mean(),
        'lidar_mean_height':    gdf_valid['lidar_p95'].mean(),
    }

    for s in stratified_results:
        s['city']    = city_name
        s['dataset'] = dataset_name

    return overall_metrics, stratified_results, error_dist


def run_validation(city_name):
    """Run the full validation pipeline for one city (both GBA and GloBFP)."""
    if city_name not in CITIES:
        print(f"City '{city_name}' not found in city_configs.py.")
        return

    city_config = CITIES[city_name]

    print("=" * 60)
    print(f"{city_config['name'].upper()}  —  DUAL DATASET VALIDATION")
    print("=" * 60)
    print(f"CRS        : {city_config['crs']}")
    print(f"LiDAR      : {city_config['lidar_source']} ({city_config['lidar_year']})")
    print(f"Resolution : {city_config['lidar_resolution']} m")

    if city_config['has_ndsm']:
        ndsm_path = city_config['ndsm_path']
        print(f"nDSM       : {ndsm_path.name} (pre-computed)")
    else:
        ndsm_path = create_ndsm(city_config)

    all_overall_metrics    = []
    all_stratified_metrics = []
    all_error_distributions = []

    for dataset_key, path_key, col_key in [
        ('gba',    'gba_path',    'gba_height_col'),
        ('globfp', 'globfp_path', 'globfp_height_col'),
    ]:
        overall, stratified, error_dist = validate_dataset(
            city_name,
            dataset_key,
            city_config[path_key],
            city_config[col_key],
            ndsm_path,
            city_config,
        )
        if overall:
            all_overall_metrics.append(overall)
            all_stratified_metrics.extend(stratified)
            all_error_distributions.append(error_dist)

    if not all_overall_metrics:
        return

    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    overall_df = pd.DataFrame(all_overall_metrics)
    overall_path = RESULTS_DIR / f'{city_name}_validation_metrics_overall.csv'
    overall_df.to_csv(overall_path, index=False)
    print(f"Saved: {overall_path.name}")

    if all_stratified_metrics:
        stratified_df = pd.DataFrame(all_stratified_metrics)
        stratified_path = RESULTS_DIR / f'{city_name}_validation_metrics_stratified.csv'
        stratified_df.to_csv(stratified_path, index=False)
        print(f"Saved: {stratified_path.name}")

    if all_error_distributions:
        error_dist_df = pd.DataFrame(all_error_distributions)
        error_dist_path = RESULTS_DIR / f'{city_name}_error_distributions.csv'
        error_dist_df.to_csv(error_dist_path, index=False)
        print(f"Saved: {error_dist_path.name}")

    print(f"\n{'='*60}")
    print(f"DATASET COMPARISON  —  {city_config['name'].upper()}")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} {'Raw':>10} {'Valid':>10} {'RMSE':>8} {'MBE':>8} {'<=3m':>8}")
    print("-" * 62)
    for m in all_overall_metrics:
        print(f"{m['dataset'].upper():<15} "
              f"{m['n_raw']:>10,} "
              f"{m['n']:>10,} "
              f"{m['rmse']:>8.2f} "
              f"{m['mbe']:>8.2f} "
              f"{m['within_3m']*100:>7.1f}%")

    print(f"\n{city_config['name']} validation complete.")


def main():
    parser = argparse.ArgumentParser(
        description='Validate building height datasets against LiDAR for one city.'
    )
    parser.add_argument(
        '--city',
        type=str,
        required=True,
        choices=list(CITIES.keys()),
        help='City key as defined in city_configs.py',
    )
    args = parser.parse_args()
    run_validation(args.city)


if __name__ == "__main__":
    main()
