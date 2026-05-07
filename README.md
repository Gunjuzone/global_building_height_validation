# Building Height Validation and Volumetric Uncertainty Analysis

Supplementary code for:

> **How Reliable Are Global Building Height Datasets for Urban Volumetric Stock Estimation? LiDAR Evidence from Four Cities.**
> Sakiru Olarewaju Olagunju, Huseyin Atakan Varol, Ferhat Karaca

---

## Repository contents

```
├── city_configs.py                   # Path configuration and per-city parameters
├── validate_city.py                  # Building height validation against LiDAR (single city)
├── run_all_validations.py            # Runs validate_city.py sequentially for all cities
├── Bootstrap uncertainty.py          # Bootstrap uncertainty analysis for volumetric error
├── Supplemenatry Material 1          # Validation statistics for global building height datasets
└── Supplementary Materia 2           # Bootstrap-derived volumetric stock error propagation results
```

### Script overview

**`city_configs.py`**
Central configuration file. Set `BASE_DIR` to your local data root before running anything else. Defines LiDAR metadata, dataset paths, CRS, and validation parameters for Amsterdam, Paris, Toronto, and Hong Kong.

**`validate_city.py`**
Validates GBA and 3D-GloBFP building heights against airborne LiDAR for one city. Where a pre-computed nDSM is unavailable, it is derived from DSM − DTM. Outputs per-building error metrics (MBE, MAE, RMSE, NMAD), stratified by height class (low-rise / mid-rise / high-rise), and writes validated GeoPackages used by the bootstrap script.

```bash
python validate_city.py --city amsterdam   # or paris, toronto, hongkong
```

**`run_all_validations.py`**
Convenience wrapper that calls `validate_city.py` for all four cities in sequence.

```bash
python run_all_validations.py
```

**`Bootstrap uncertainty.py`**
Bootstrap resampling analysis (up to 15,000 iterations with convergence testing) that propagates building-level height errors into 95% confidence intervals on city-scale volumetric stock estimates. Applies a material stock correction factor of 0.67 to convert gross building volume to material-bearing volume (Heeren & Fishman, 2019). Reads the validated GeoPackages produced by `validate_city.py`.

```bash
python MC_Sim_with_correction_factor.py
```

---

## Dependencies

```
geopandas
rasterio
rasterstats
numpy
pandas
tqdm
```

Install with:

```bash
pip install geopandas rasterio rasterstats numpy pandas tqdm
```

---

## Data

Input data (LiDAR, GBA, GloBFP) are not included in this repository. All primary sources are cited in the manuscript. Set the paths in `city_configs.py` to match your local directory structure.

Expected input layout:

```
data/
├── raw/
│   ├── lidar/       # DSM/DTM or pre-computed nDSM rasters (.tif)
│   ├── gba/         # GBA building footprints (.gpkg)
│   └── globfp/      # 3D-GloBFP building footprints (.gpkg)
├── processed/       # Intermediate outputs (nDSM, validated .gpkg, scatter CSVs)
└── results/         # Final CSV outputs (metrics, bootstrap distributions)
```

---

## Citation

*To be updated upon publication.*
