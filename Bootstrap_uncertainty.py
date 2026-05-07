"""
Bootstrap uncertainty analysis for volumetric error quantification.

Resamples validated building footprints (with replacement) to estimate 95%
confidence intervals on volumetric errors when using GBA or 3D-GloBFP heights
relative to a LiDAR reference.

A convergence test determines the minimum number of bootstrap iterations
needed before CI width stabilises (< 0.5% change between iteration levels).
Results are written to RESULTS_DIR as defined in city_configs.py.

Usage:
    python MC_Sim_with_correction_factor.py

Input files (produced by validate_city.py):
    <city>_<dataset>_validated.gpkg  —  one per city/dataset combination
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from city_configs import OUTPUT_DIR as PROCESSED_DIR, RESULTS_DIR

RANDOM_SEED = 42

HEIGHT_MIN = 2.0    # metres — physical lower bound
HEIGHT_MAX = 600.0  # metres — physical upper bound

# Material stock correction factor (0.67)
# Scales gross building volume (footprint × height) to the material-bearing
# fraction by accounting for internal voids, floor slabs, and mechanical
# service spaces. 
# Volume_material = (footprint_area × height) × MATERIAL_STOCK_CORRECTION_FACTOR
MATERIAL_STOCK_CORRECTION_FACTOR = 0.67

CONFIDENCE_LEVEL      = 0.95
CONVERGENCE_THRESHOLD = 0.005   # 0.5% change in CI width
N_REPLICATIONS        = 3       # replications per iteration level
ITERATION_LEVELS      = [500, 1000, 2500, 5000, 10000, 15000]


def apply_height_constraints(heights):
    return np.clip(heights, HEIGHT_MIN, HEIGHT_MAX)


def calculate_volumes(gdf_sample):
    """Return (vol_reference, vol_dataset, vol_error, vol_error_pct) for a building sample.

    Both reference and dataset volumes are scaled by MATERIAL_STOCK_CORRECTION_FACTOR.
    """
    ref_heights     = apply_height_constraints(gdf_sample['lidar_p95'].values)
    dataset_heights = apply_height_constraints(gdf_sample['height'].values)
    areas           = gdf_sample['area_m2'].values

    vol_reference = (areas * ref_heights).sum()     * MATERIAL_STOCK_CORRECTION_FACTOR
    vol_dataset   = (areas * dataset_heights).sum() * MATERIAL_STOCK_CORRECTION_FACTOR
    vol_error     = vol_dataset - vol_reference
    vol_error_pct = (vol_error / vol_reference * 100) if vol_reference > 0 else 0

    return vol_reference, vol_dataset, vol_error, vol_error_pct


def bootstrap_iteration(gdf, rng):
    """One bootstrap iteration: resample buildings with replacement."""
    indices    = rng.choice(len(gdf), size=len(gdf), replace=True)
    gdf_sample = gdf.iloc[indices].copy()
    _, _, vol_error, vol_error_pct = calculate_volumes(gdf_sample)
    return vol_error, vol_error_pct


def convergence_test(gdf, city_name, dataset_name):
    """Run bootstrap at increasing iteration counts and return the first level
    at which CI width changes by less than CONVERGENCE_THRESHOLD.

    Returns (optimal_iterations, convergence_data).
    """
    print(f"\n{'='*60}")
    print(f"CONVERGENCE TEST: {city_name.upper()} — {dataset_name.upper()}")
    print(f"{'='*60}")

    convergence_data   = []
    optimal_iterations = ITERATION_LEVELS[-1]

    for n_iter in ITERATION_LEVELS:
        print(f"\n  {n_iter:,} iterations ...")

        ci_widths_pct = []
        ci_widths_abs = []
        alpha         = 1 - CONFIDENCE_LEVEL

        for rep in range(N_REPLICATIONS):
            rng = np.random.RandomState(RANDOM_SEED + rep * 1_000_000)
            errors, errors_pct = [], []

            for _ in tqdm(range(n_iter), desc=f"  Rep {rep+1}/{N_REPLICATIONS}", leave=False):
                e, ep = bootstrap_iteration(gdf, rng)
                errors.append(e)
                errors_pct.append(ep)

            ci_widths_pct.append(
                np.percentile(errors_pct, (1 - alpha/2) * 100)
                - np.percentile(errors_pct, alpha/2 * 100)
            )
            ci_widths_abs.append(
                np.percentile(errors, (1 - alpha/2) * 100)
                - np.percentile(errors, alpha/2 * 100)
            )

        mean_w_pct = np.mean(ci_widths_pct)
        std_w_pct  = np.std(ci_widths_pct)
        mean_w_abs = np.mean(ci_widths_abs)
        std_w_abs  = np.std(ci_widths_abs)

        convergence_data.append({
            'n_iterations':     n_iter,
            'mean_ci_width_pct': mean_w_pct,
            'std_ci_width_pct':  std_w_pct,
            'cv_pct':            std_w_pct / mean_w_pct if mean_w_pct > 0 else 0,
            'mean_ci_width_abs': mean_w_abs,
            'std_ci_width_abs':  std_w_abs,
            'cv_abs':            std_w_abs / mean_w_abs if mean_w_abs > 0 else 0,
        })

        print(f"  CI width : {mean_w_pct:.3f}% (±{std_w_pct:.3f}%)  |  "
              f"{mean_w_abs:,.0f} m³ (±{std_w_abs:,.0f} m³)")

        if len(convergence_data) >= 2:
            pct_change = abs(mean_w_pct - convergence_data[-2]['mean_ci_width_pct']) \
                         / convergence_data[-2]['mean_ci_width_pct']
            print(f"  Change from previous level : {pct_change*100:.3f}%")

            if pct_change < CONVERGENCE_THRESHOLD:
                print(f"\n  Converged at {n_iter:,} iterations "
                      f"(CI width change < {CONVERGENCE_THRESHOLD*100}%)")
                optimal_iterations = n_iter
                break
    else:
        print(f"\n  Did not meet convergence threshold at any level. "
              f"Using {optimal_iterations:,} iterations.")

    return optimal_iterations, convergence_data


def calculate_height_class_volumes(gdf):
    """Return per-height-class volume statistics (dict keyed by class label)."""
    class_volumes = {}

    for height_class in gdf['height_class'].unique():
        subset           = gdf[gdf['height_class'] == height_class]
        dataset_heights  = apply_height_constraints(subset['height'].values)
        reference_heights = apply_height_constraints(subset['lidar_p95'].values)
        areas            = subset['area_m2'].values

        vol_dataset   = (areas * dataset_heights).sum()  * MATERIAL_STOCK_CORRECTION_FACTOR
        vol_reference = (areas * reference_heights).sum() * MATERIAL_STOCK_CORRECTION_FACTOR

        class_volumes[height_class] = {
            'n':            len(subset),
            'vol_dataset':  vol_dataset,
            'vol_reference': vol_reference,
            'vol_error':    vol_dataset - vol_reference,
            'vol_error_pct': ((vol_dataset - vol_reference) / vol_reference * 100)
                              if vol_reference > 0 else 0,
        }

    return class_volumes


def run_bootstrap_analysis(gdf, city_name, dataset_name, n_bootstrap, run_convergence=True):
    """Bootstrap volumetric error analysis for one city/dataset pair.

    Returns (results dict, bootstrap_data DataFrame, convergence_data list).
    """
    print(f"\n{'='*60}")
    print(f"BOOTSTRAP: {city_name.upper()} — {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Buildings         : {len(gdf):,}")
    print(f"Random seed       : {RANDOM_SEED}")
    print(f"Confidence level  : {CONFIDENCE_LEVEL*100:.0f}%")

    print(f"\nHeight class distribution:")
    for hc, count in gdf['height_class'].value_counts().sort_index().items():
        print(f"  {hc} : {count:,} ({count/len(gdf)*100:.1f}%)")

    vol_ref_obs, vol_ds_obs, vol_err_obs, vol_err_pct_obs = calculate_volumes(gdf)

    n_clip_low_ref  = (gdf['lidar_p95'] < HEIGHT_MIN).sum()
    n_clip_high_ref = (gdf['lidar_p95'] > HEIGHT_MAX).sum()
    n_clip_low_ds   = (gdf['height']    < HEIGHT_MIN).sum()
    n_clip_high_ds  = (gdf['height']    > HEIGHT_MAX).sum()

    if n_clip_low_ref + n_clip_high_ref + n_clip_low_ds + n_clip_high_ds > 0:
        print(f"\nHeight constraint clipping:")
        print(f"  LiDAR reference : {n_clip_low_ref} below {HEIGHT_MIN} m, "
              f"{n_clip_high_ref} above {HEIGHT_MAX} m")
        print(f"  Dataset         : {n_clip_low_ds} below {HEIGHT_MIN} m, "
              f"{n_clip_high_ds} above {HEIGHT_MAX} m")

    print(f"\nObserved volumes (full validation sample):")
    print(f"  Reference (LiDAR) : {vol_ref_obs:,.0f} m³")
    print(f"  Dataset           : {vol_ds_obs:,.0f} m³")
    print(f"  Error             : {vol_err_obs:+,.0f} m³  ({vol_err_pct_obs:+.2f}%)")

    class_results = calculate_height_class_volumes(gdf)
    print(f"\nObserved error by height class:")
    for hc in ['low_rise', 'mid_rise', 'high_rise']:
        if hc in class_results:
            cv = class_results[hc]
            print(f"  {hc} : {cv['vol_error']:+,.0f} m³  ({cv['vol_error_pct']:+.2f}%)  "
                  f"[{cv['n']:,} buildings]")

    if run_convergence:
        optimal_iterations, convergence_data = convergence_test(gdf, city_name, dataset_name)
    else:
        optimal_iterations = n_bootstrap
        convergence_data   = None
        print(f"\nSkipping convergence test. Using {optimal_iterations:,} iterations.")

    print(f"\n{'='*60}")
    print(f"FINAL BOOTSTRAP: {optimal_iterations:,} iterations")
    print(f"{'='*60}")

    rng              = np.random.RandomState(RANDOM_SEED)
    bootstrap_errors     = []
    bootstrap_errors_pct = []

    for _ in tqdm(range(optimal_iterations), desc="Bootstrap"):
        e, ep = bootstrap_iteration(gdf, rng)
        bootstrap_errors.append(e)
        bootstrap_errors_pct.append(ep)

    bootstrap_errors     = np.array(bootstrap_errors)
    bootstrap_errors_pct = np.array(bootstrap_errors_pct)

    alpha        = 1 - CONFIDENCE_LEVEL
    ci_lower_pct = np.percentile(bootstrap_errors_pct, alpha/2 * 100)
    ci_upper_pct = np.percentile(bootstrap_errors_pct, (1 - alpha/2) * 100)
    ci_lower_abs = np.percentile(bootstrap_errors,     alpha/2 * 100)
    ci_upper_abs = np.percentile(bootstrap_errors,     (1 - alpha/2) * 100)

    ci_width_pct = ci_upper_pct - ci_lower_pct
    ci_width_abs = ci_upper_abs - ci_lower_abs

    print(f"\nVolumetric error (percentage):")
    print(f"  Observed       : {vol_err_pct_obs:+.2f}%")
    print(f"  Bootstrap mean : {np.mean(bootstrap_errors_pct):+.2f}%")
    print(f"  Bootstrap SD   : {np.std(bootstrap_errors_pct):.2f}%")
    print(f"  95% CI         : [{ci_lower_pct:+.2f}%, {ci_upper_pct:+.2f}%]")
    print(f"  CI half-width  : ±{ci_width_pct/2:.2f} pp")

    print(f"\nVolumetric error (absolute):")
    print(f"  Observed       : {vol_err_obs:+,.0f} m³")
    print(f"  Bootstrap mean : {np.mean(bootstrap_errors):+,.0f} m³")
    print(f"  Bootstrap SD   : {np.std(bootstrap_errors):,.0f} m³")
    print(f"  95% CI         : [{ci_lower_abs:+,.0f}, {ci_upper_abs:+,.0f}] m³")
    print(f"  CI half-width  : ±{ci_width_abs/2:,.0f} m³")

    results = {
        'city':                         city_name,
        'dataset':                      dataset_name,
        'n_buildings':                  len(gdf),
        'n_bootstrap':                  optimal_iterations,
        'random_seed':                  RANDOM_SEED,
        'confidence_level':             CONFIDENCE_LEVEL,

        'vol_reference':                vol_ref_obs,
        'vol_dataset_observed':         vol_ds_obs,
        'vol_error_observed':           vol_err_obs,
        'vol_error_pct_observed':       vol_err_pct_obs,

        'vol_error_pct_bootstrap_mean':   np.mean(bootstrap_errors_pct),
        'vol_error_pct_bootstrap_median': np.median(bootstrap_errors_pct),
        'vol_error_pct_bootstrap_std':    np.std(bootstrap_errors_pct),
        'vol_error_pct_ci_lower':         ci_lower_pct,
        'vol_error_pct_ci_upper':         ci_upper_pct,
        'vol_error_pct_ci_width':         ci_width_pct,

        'vol_error_bootstrap_mean':       np.mean(bootstrap_errors),
        'vol_error_bootstrap_median':     np.median(bootstrap_errors),
        'vol_error_bootstrap_std':        np.std(bootstrap_errors),
        'vol_error_ci_lower':             ci_lower_abs,
        'vol_error_ci_upper':             ci_upper_abs,
        'vol_error_ci_width':             ci_width_abs,

        'height_min_constraint':          HEIGHT_MIN,
        'height_max_constraint':          HEIGHT_MAX,
        'n_clipped_low_ref':              n_clip_low_ref,
        'n_clipped_high_ref':             n_clip_high_ref,
        'n_clipped_low_dataset':          n_clip_low_ds,
        'n_clipped_high_dataset':         n_clip_high_ds,
    }

    for hc in ['low_rise', 'mid_rise', 'high_rise']:
        if hc in class_results:
            results[f'{hc}_n']            = class_results[hc]['n']
            results[f'{hc}_vol_error']    = class_results[hc]['vol_error']
            results[f'{hc}_vol_error_pct'] = class_results[hc]['vol_error_pct']

    bootstrap_data = pd.DataFrame({
        'iteration':    range(optimal_iterations),
        'vol_error':    bootstrap_errors,
        'vol_error_pct': bootstrap_errors_pct,
        'city':         city_name,
        'dataset':      dataset_name,
    })

    return results, bootstrap_data, convergence_data


def process_city_dataset(city_name, dataset_name, run_convergence=True):
    """Load a validated GeoPackage and run the bootstrap analysis.

    Returns (results, bootstrap_data, convergence_data), or (None, None, None)
    if the input file is missing.
    """
    gpkg_path = PROCESSED_DIR / f'{city_name}_{dataset_name}_validated.gpkg'

    if not gpkg_path.exists():
        print(f"File not found: {gpkg_path.name}")
        return None, None, None

    print(f"\n{'='*80}")
    print(f"PROCESSING: {city_name.upper()} — {dataset_name.upper()}")
    print(f"{'='*80}")

    gdf = gpd.read_file(gpkg_path)
    print(f"Loaded {len(gdf):,} validated buildings")

    results, bootstrap_data, convergence_data = run_bootstrap_analysis(
        gdf, city_name, dataset_name,
        n_bootstrap=15000,
        run_convergence=run_convergence,
    )

    bootstrap_path = RESULTS_DIR / f'{city_name}_{dataset_name}_bootstrap_errors.csv'
    bootstrap_data.to_csv(bootstrap_path, index=False)
    print(f"Saved: {bootstrap_path.name}")

    return results, bootstrap_data, convergence_data


def main():
    cities   = ['amsterdam', 'paris', 'toronto', 'hongkong']
    datasets = ['gba', 'globfp']

    all_results         = []
    all_bootstrap_data  = []
    all_convergence_data = []

    for city in cities:
        for dataset in datasets:
            results, bootstrap_data, convergence_data = process_city_dataset(
                city, dataset, run_convergence=True
            )
            if results is not None:
                all_results.append(results)
                all_bootstrap_data.append(bootstrap_data)

                if convergence_data is not None:
                    for row in convergence_data:
                        row['city']    = city
                        row['dataset'] = dataset
                    all_convergence_data.extend(convergence_data)

    print(f"\n{'='*80}")
    print("SAVING COMBINED RESULTS")
    print(f"{'='*80}")

    results_df   = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / 'bootstrap_volumetric_uncertainty_summary.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Saved: {results_path.name}")

    if all_bootstrap_data:
        bootstrap_df   = pd.concat(all_bootstrap_data, ignore_index=True)
        bootstrap_path = RESULTS_DIR / 'bootstrap_all_iterations.csv'
        bootstrap_df.to_csv(bootstrap_path, index=False)
        print(f"Saved: {bootstrap_path.name}")

    if all_convergence_data:
        conv_df   = pd.DataFrame(all_convergence_data)
        conv_path = RESULTS_DIR / 'bootstrap_convergence_test.csv'
        conv_df.to_csv(conv_path, index=False)
        print(f"Saved: {conv_path.name}")

    print(f"\n{'='*80}")
    print("VOLUMETRIC ERROR UNCERTAINTY SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'City':<12} {'Dataset':<8} {'N':>10} {'Obs.Err%':>11} {'95% CI (%)':>20} {'±Half-width':>12} {'Iter':>8}")
    print("-" * 90)

    for _, row in results_df.iterrows():
        ci_range = f"[{row['vol_error_pct_ci_lower']:+.2f}, {row['vol_error_pct_ci_upper']:+.2f}]"
        print(f"{row['city']:<12} "
              f"{row['dataset']:<8} "
              f"{row['n_buildings']:>10,} "
              f"{row['vol_error_pct_observed']:>+10.2f} "
              f"{ci_range:>20} "
              f"±{row['vol_error_pct_ci_width']/2:>10.2f} "
              f"{row['n_bootstrap']:>8,}")

    print(f"\n{'='*80}")
    print("CONVERGENCE SUMMARY")
    print(f"{'='*80}")
    if all_convergence_data:
        print(f"\n{'City':<12} {'Dataset':<8} {'Converged':>12} {'Final iter':>12} {'CI half-width (%)':>18}")
        print("-" * 68)
        max_iter = ITERATION_LEVELS[-1]
        for _, row in results_df.iterrows():
            status = "Yes" if row['n_bootstrap'] < max_iter else "At max"
            print(f"{row['city']:<12} "
                  f"{row['dataset']:<8} "
                  f"{status:>12} "
                  f"{row['n_bootstrap']:>12,} "
                  f"±{row['vol_error_pct_ci_width']/2:>17.3f}")

    print(f"\n{'='*80}")
    print("VOLUMETRIC ERROR UNCERTAINTY (ABSOLUTE)")
    print(f"{'='*80}")
    print(f"\n{'City':<12} {'Dataset':<8} {'Obs.Err (m³)':>15} {'95% CI (m³)':>35} {'±Half-width (m³)':>17}")
    print("-" * 95)
    for _, row in results_df.iterrows():
        ci_abs = f"[{row['vol_error_ci_lower']:+,.0f}, {row['vol_error_ci_upper']:+,.0f}]"
        print(f"{row['city']:<12} "
              f"{row['dataset']:<8} "
              f"{row['vol_error_observed']:>+15,.0f} "
              f"{ci_abs:>35} "
              f"±{row['vol_error_ci_width']/2:>15,.0f}")

    print(f"\nBootstrap analysis complete. Results written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
