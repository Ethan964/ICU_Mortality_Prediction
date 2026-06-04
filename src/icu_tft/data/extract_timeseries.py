"""
extract_timeseries.py

==========================
Extracted hourly-binned vital signs and lab values for a 24-hour ICU mortality prediction task.
Data taken from MIMIC-IV v3.1

Pipeline
---------
1. Query chartevents / labevents for each stay_id in the cohort
2. Compute hours-since-admission offset, bin into 1-hour slots
3. Within each (stay_id, time_step) bin, take the first valid value.
4. Build a complete 24-row skeleton (time_steps 0-23) per patient.
5. Forward-fill gaps up to MAX_FORWARD_FILL_HOURS (6); leave longer gaps as NaN.
6. CLips all values to CLIP_BOUNDS (physiologically plausible regions).
7. Add binary missingness indicator columns (*_missing).
8. Return a Pandas DataFrame index by (stay_id, time_step).


The heavy aggregation is done entirely in polars where the conversion to pandas happens
only at the final step to keep memory allocation efficient on MIMIC's larger tables.


Usage
--------
    import duckdb
    import pandas as pd
    from icu_tft.data.extract_timeseries import build_timeseries, validate_cohort

    con = duckdb.connect('mimic.duckdb')
    cohort = pd.read_sql('SELECT * FROM cohort', con)       # not reading parquet, feather, or csv, files to large
    ts = build_timeseries(cohort, con)
    validate_report(ts)
"""


from __future__ import annotations


import logging
from typing import Final


import duckdb
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

# -----------------------
#       constants 
# -----------------------

# taking vital signas from mimic_icu.chartevents.
# key  = standard feature name used throughout this module
# value = list of mimic-iv item ids that map to that feature

VITAL_ITEMIDS: Final[dict[str, list[int]]] = {
    'heart_rate'        :             [220045],
    'sbp'               :             [220179],
    'dbp'               :             [229180],
    'mbp'               :             [220052],
    'spo2'              :             [220277],
    'resp_rate'         :             [220210],
    'tempurature_c'     :             [223761],
    'gcs_verbal'        :             [223900],
    'gcs_motor'         :             [223901],
    'gcs_eyes'          :             [220739],
}

# lab value: label mapping from mimic_hosp.labevents
# Maps standard feature name to list of itemids found in d_labitems.
# itemids taken from MIMIC-IV v3.1 d_labitems 

LAB_ITEMIDS: Final[dict[str, list[int]]] = {
    'creatinine'    :   [50912],
    'bun'           :   [51006],
    'sodium'        :   [50971],
    'bicarbonate'   :   [50882],
    'lactate'       :   [50813],
    'wbc'           :   [51301],
    'hemoglobin'    :   [51222],
    'platelets'     :   [51265],
    'inr'           :   [51237],       
}   

# components sums up all gcs features into one single gcs_total feature
GCS_COMPONENTS : Final[list[str]] = ['gcs_verbal', 'gcs_motar', 'gcs_eyes']

# max. number of consecutive hours to forward fill a missing value.
# gaps longer than this are left as NaN to avoid erronious data imputation.
MAX_FORWARD_FILL_HOURS: Final[int] = 6

# target observation window: hours 0 through 23 (inclusive)
N_HOURS: Final[int] = 24


# ----------------------------------------------
# Physiologically plausible clip bounds
# ----------------------------------------------

# format: feature_name -> (plausible lower bound, plausible upper bound)
# Values outside these bounds are treared as measurement or transciption errors.
# and will be replaced with NaNs before forward-filling so they do not forward propagate.
# Source: MIMIC-Extract paper (Wang et al. 2020), clincial reference regions, 
#         common ICU-data quality filters used in relevant literature.

CLIPS_BOUNDS: Final[dict[str, tuple[float, float]]] = {\
    'heart_rate'        :             (0.0, 300.0),
    'sbp'               :             (0.0, 300.0),
    'dbp'               :             (0.0, 200.0),
    'mbp'               :             (0.0, 250.0),
    'spo2'              :             (0.0, 100.0),
    'resp_rate'         :             (0.0, 80.0),
    
    # temperature: becareful with degree conversion
    'tempurature_c'     :             (25.0, 45.0),
    
    # gcs components
    'gcs_verbal'        :             (1.0, 5.0),
    'gcs_motor'         :             (1.0, 6.0),
    'gcs_eyes'          :             (1.0, 4.0),
    'gcs_total'         :             (3.0, 15.0),

    # lab
    'creatinine'        :             (0.0, 30.0),
    'bun'               :             (0.0, 200.0),
    'sodium'            :             (80.0, 200.0),
    'potassium'         :             (1.0, 15.0),
    'bicarbonate'       :             (0.0, 60.0),
    'lactate'           :             (0.0, 30.0),
    'wbc'               :             (0.0, 500.0),
    'hemoglobin'        :             (0.0, 25.0),
    'platelets'         :             (0.0, 3000.0),
    'inr'               :             (0.5, 20.0), 
}

# all features names in final output order (gcs components replaced by total).
ALL_FEATURES: Final[list[str]] = [
    'heart_rate', 'sbp', 'dbp', 'mbp', 'spo2', 'resp_rate', 'temperature_c',
    'gcs_total', 
    'creatinine', 'bun', 'sodium', 'potassium', 'bicarbonate', 'lactate', 
    'wbc', 'hemoglobin', 'platelets', 'inr'
]

# INTERAL DATA CONEVRSION HELPERS 

def _fahrenheit_to_celsius(series: pl.Expr) -> pl.Expr:
    '''coverts polars expression from F to C.'''
    return (series - 32.0) * (5.0 / 9.0)

def _itemid_filter_expr(itemid_map: dict[str, list[int]]) -> list[int]:
    '''flattens every itemid from a mapping tool into a single deduplicated list.'''
    ids: list[int] = []
    for id_list in itemid_map.values():
        ids.extend(id_list)
    return list(set(ids))       # removes dupes.

def _build_id_to_feature(itemid_map: dict[str, list[int]]) -> dict[int, str]:
    '''inverts itemid_map from itemid -> feature_name for faster label lookup.'''
    mapping: dict[int, str] = {}
    for feature, ids in itemid_map.items():
        for iid in ids:
            mapping[iid] = feature
    return mapping


def _clip_values(df: pl.DataFrame, bounds: dict[str, tuple[float, float]]) -> pl.DataFrame:
    '''
    replaces values outside physiologically possible ranges with Nulls
    this is done to not clamp the values inoder to have them participate in
    missingness indicators and pre-existing forward-fill logic, rather than
    simply biasing sample means.

    params: 
    -----------

    df: 
        Polars DataFrame withb one column per feature.

    bounds:
        Mapping of feature_name -> (lower_bound, upper_bound)

    
    Returns
    ----------
    pl.DataFrame with out-of-range values replaced by NULL.
    '''

    exprs: list[pl.Expr] = []
    for col in df.columns:
        if col in bounds:
            lo, hi = bounds[col]
            exprs.append(
                pl.when((pl.col(col) < lo) | (pl.col(col) > hi))
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
        else:
            exprs.append(pl.col(col))
    return df.select(exprs)


def _forward_fill_with_limit(df: pl.DataFrame, features: list[str], limit: int) -> pl.DataFrame:
    '''
    Forward-fill null values within each individual stay_id group, up to a specified value
    of limit steps.

    Polar's native forward_fill accepts a limit param (number of consecutive nulls to fill). 
    We sort by time_step within each group before filling.

    Params
    ---------
    df: Polars DataFrame with columns [stay_id, time_step, *features].

    features: feature column names to be forward-filled.

    limit: Max. number of consecutive null steps to fill.

    Returns:
    ---------
    pl.DataFrame with forward-filled values.
    '''

    fill_exprs_exprs = [
        pl.col(f).forward_fill(limit=limit).over('stay_id')
        for f in features:
    ]
    non_feature_cols = [c for c in df.columns if c not in features]
    return df.sort(['stay_id', 'time_step']).with_columns(fill_exprs)

def _add_missingness_indicators(df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    '''
    Add a binary indicator *_missing column for each feature

    The indicator is set to 1 where the value is null after forward-filling 
    for example if the gap exceeded MAX_FOWARD_FILL_HOURS or the feature was
    never observed for patient x.

    Params
    ------------
    df: 
        Polar DataFrame containing feature columns

    features: 
        list of feature names to create indicators for.

    Returns
    ---------
    pl.DataFrame with additional f'feature'_missing columns (int, 0 | 1).

    '''
    indicator_exprs = [
        pl.col(f).is_null.cast(pl.Int8).alias(f'{f}_missing')
        for f in features
    ]
    return df.with_columns(indicator_exprs)