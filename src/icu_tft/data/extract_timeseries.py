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
    validate_cohort(ts)
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

# taking vital signals from mimic_icu.chartevents.
# key  = standard feature name used throughout this module
# value = list of mimic-iv item ids that map to that feature

VITAL_ITEMIDS: Final[dict[str, list[int]]] = {
    'heart_rate'        :             [220045],
    'sbp'               :             [220179],
    'dbp'               :             [229180],
    'mbp'               :             [220052],
    'spo2'              :             [220277],
    'resp_rate'         :             [220210],
    'temperature_c'     :             [223761],
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
GCS_COMPONENTS : Final[list[str]] = ['gcs_verbal', 'gcs_motor', 'gcs_eyes']

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
    'temperature_c'     :             (25.0, 45.0),
    
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

    fill_exprs = [
        pl.col(f).forward_fill(limit=limit).over('stay_id')
        for f in features
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
        pl.col(f).is_null().cast(pl.Int8).alias(f'{f}_missing')
        for f in features
    ]
    return df.with_columns(indicator_exprs)


# ----------------------------------------------
# Query Builders 
# ----------------------------------------------

def _query_chartevents(stay_ids: list[int]) -> str:
    '''
    builds a parameterized SQL query to pull relevant cchartevent rows.
    
    Returns only rows within the first 24 hours of ICU admission, joined 
    to icustays to compute the hours-since-admission offset. 
    
    Params
    --------
    SQL string compatable with DuckDb and PostgreSQL.
    '''
    all_vital_signs = _itemid_filter_expr(VITAL_ITEMIDS)
    ids_sql = ",".join(str(i) for i in all_vital_signs)
    stay_ids_sql = ', '.join(str(i) for i in stay_ids)

    return f'''
    SELECT
        ce.stay_id, 
        ce.itemid,
        ce.charttime, 
        ce.valuenum,
        ie.intime,
        -- Hours since ICU admission, floored to int. bin index.
        FLOOR(
            EXTRACT(EPOCH FROM (ce.charttime - ie.intime)) / 3600.0
            )::INTEGER          AS time_step
        FROM mimic_icu.chartevents AS ce
        INNER JOIN mimic_icu.icustays AS ie
            ON ce.stay_id = ie.stay_id
        WHERE 
            ce.stay_id      IN ({stay_ids_sql})
            AND ce.itemid   IN ({ids_sql})
            -- Only Values recorded during the 24-hour observation window
            AND ce.charttime >= ie.intime
            AND ce.charttime < ie.intime + INTERVAL '24 hours'
            -- Exclude null values common in MIMIC
            AND ce.valuenum IS NOT NULL
        ORDER BY ce.stay_id, time_step, ce.charttime
'''

def _query_labevents(hadm_ids: list[int]) -> str:
    '''
    SQL query that pulls lab values within the 24-hour ICU window.
    
    labevents links to hadm_id, not stay_id, so we join back to icustays to recover intime for the correct first ICU stay.

    Params
    ---------
    hadm_ids:
        hopsital admission ids corresponding to the cohort.

    stay_ids: 
        Mapping of stay_id -> intime ISO string used to push the time filter as close to the source as possible. 

    Returns
    ---------
    SQL string
    '''
    all_lab_ids = _itemid_filter_expr(LAB_ITEMIDS)
    ids_sql = ', '.join(str(i) for i in all_lab_ids)
    hadm_ids_sql = ', '.join(str(i) for i in hadm_ids)

    return f'''
    SELECT
        ie.stay_id,
        le.itemid,
        le.charttime,
        le.valuenum,
        ie.intime,
        FLOOR(
            EXTRACT(EPOCH FROM (le.charttime - ie.intime)) / 3600.0
            )::INTEGER          AS time_step
    FROM mimic_hosp.labevents AS le
    -- Join to icustays to get the first ICU stay's intime for the admission
    INNER JOIN mimic_icu.icustays AS ie
        ON le.hadm_id = ie.hadm_id
    WHERE 
        le.hadm_id IN ({hadm_ids_sql})
        AND le.itemid IN ({ids_sql})
        AND le.charttime >= ie.intime
        AND le.charttime < ie.intime + INTERVAL '24 hours'
        AND le.valuenum IS NOT NULL
        -- quick status flag, 'D' marks deleted results in some MIMIC versions BEWARE
        AND (le.flag IS NOT NULL OR le.flag != 'delta')
    ORDER BY ie.stay_id, time_step, le.charttime
    '''

# ----------------------------------------------
# Transformations 
# ----------------------------------------------

def _raw_to_feature_frame(
        raw: pl.DataFrame,
        ids_to_features: dict[int, str],
        is_vitals: bool,
        ) -> pl.DataFrame:
    '''
    Maps itemids to feature names, apply unit conversions, and bins into hour intervals.

    Steps
    -------- 
    1. Map itemid -> feature_name via id_to_feature func.
    2. Apply temperature F to C conversion for temperature_c feature
    3. Group by (stay_id, time_step, feature_name), take first non-null value.
    4. Pivot to wide format" one column per feature

    Params
    --------- 
    Wide polars: DataFrame with columns [stay_id, time_step, feature_1, ....]
    '''

    if raw.is_empty():
        return pl.DataFrame(schema={'stay_id' : pl.Int64, 'time_step': pl.Int32})
    
    feature_map_series = raw['itemid'].map_elements(
        lambda x: ids_to_features.get(x, '__drop__'), return_dtype=pl.String
    )

    df = raw.with_columns(feature_map_series.alias('feature_name'))
    df = df.filter(pl.col('feature_name') != '__drop__')

    # temp conversion
    if is_vitals:
        df = df.with_columns(
            pl.when(pl.col('feature_name') == 'temperature_c')
            .then(_fahrenheit_to_celsius(pl.col('valuenum')))
            .otherwise(pl.col('valuenum'))
            .alias('valuenum')
        )
        # first valid value per (stay_id, time_step, feature_name)
        # sort is already done in SQL then the group by perserves first row order in Polars
        # when maintain_order=True
    binned = (
        df.sort(['stay_id', 'time_step', 'charttime'])
        .group_by(['stay_id', 'time_step', 'feature_name'], maintain_order=True)
        .agg(pl.col('valuenum').first().alias('value'))
    )
    wide = binned.pivot(
        values='value',
        index=['stay_id', 'time_step'],
        columns='feature_name',
        aggregate_function='first'
    )

    return wide

def _build_skeleton(stay_ids: list[int]) -> pl.DataFrame:
    '''
    Creates a complete stay_id by time_step grid for hours 0-23

    This ensures every stay gets exactly 24 rows regardless of whether observations exist,
    ensuring the final tensor is dense and uniformly shaped.
    
    Params
    --------
    stay_ids: 
        All stay_ids in the cohort
    
    Returns
    --------
    Polars dataframe with columns [stay_id (Int64), time_step (Int 32)].
    '''

    stays = pl.DataFrame({'stay_id' : stay_ids}).cast({'stay_id': pl.Int64})
    hours = pl.DataFrame({'time_step' : list(range(N_HOURS))}).cast({'time_step' : pl.Int32})

    # return a cross join to produce all cocmbinations of stay_id and time_steps
    return stays.join(hours, how='cross')


def _combine_gcs(df: pl.DataFrame) -> pl.DataFrame:
    '''
    Sum GCS component columns into gcs_total and drop the component columns.

    gcs_total = gcs_eyes + gcs_motor + gcs_verbal (range 3-15).

    If any component is null for a given bin, gcs_total is null for that bin
    (sum of nulls is null in Polars, use fill_null(0) only if all three components are
    expeceted to coincide, which they can in MIMIC-IV)
    
    Handling Missingness: if ALL three components are null = null total score.
    If some are null, we treat gcs_total partial_null as null total to be conservative.

    Params
    -------
    df: 
        wide DataFrame potentially containing gcs_verbal, gcs_motor, gcs_eyes

    Returns
    -------
    DataFrame with gcs_total column replacing individual GCS components.
    '''
    present = [c for c in GCS_COMPONENTS if c in df.columns]
    if not present:
        if 'gcs_total' not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias('gcs_total'))
        return df
    
    if len(present) == 3:
        df = df.with_columns(
            (pl.col('gcs_eyes') + pl.col('gcs_motor') + pl.col('gcs_verbal'))
            .alias('gcs_total')
        )
    else:
        df = df.with_columns(
            sum(pl.col(c) for c in present).alias('gcs_total')
        )
    df = df.drop([c for c in GCS_COMPONENTS if c in df.columns])
    return df

def _ensure_all_feature_columns(df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    '''
    Add any missing features columns as null float64s.

    This func gusrentees the DataFrame has exactly the expected column set regardless of
    which features had observations in this cohort/batch of patients.

    Params
    -------
    df:
        polars DataFrame that may be missing some feature columns.

    features: 
        Full list of features that will be checked for missing some feature columns.

    Returns
    -------
    DataFrame with all feature columns present (where nulls absent).
    '''

    for feat in features:
        if feat not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(feat))
    return df


# ----------------------------------------------
# Public APIs 
# ----------------------------------------------

def build_timeseries(
        cohort: pd.DataFrame,
        con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    '''
    Extract and preprocesses a 24-hour time-series for every patient's icu stay

    This will serve as the main entry point for this module. It orchestrates SQL extraction,
    Polars-based (aggregation, labelling, foward-filling, filtering, and missingness flagging)

    Params
    -------
    cohort:
        Pandas DF produced by the cohort SQl script : ../src/icu_tft/data/sql/mimic_iv_24h_icu_mortality_cohort.sql
        Must contain at minumum: stay_id (int), hadm_id (int).
    
    con:
        Active DuckDB connection pointing at the database with mimic_icu and mimic_hosp schemas attached.

    Returns
    -------
    pandas.DataFrame
    Indexed by (stay_id, time_step). Columns:
        - One float32 column per feature in ALL_FEATURES
        - One Into binary column per feature named <feature>_missing.
        Shape: (n_patients x 24) rows x (n_features x 2) columns.

    Raises
    -------
    ValueError: 
        If cohort is empty or missing required columns
    '''
    required_cols = {'stay_id', 'hadm_id'}
    missing_cols = required_cols - set(cohort.columns)
    if missing_cols:
        return ValueError(f'Cohort DataFrame missing required columns: {missing_cols}')
    if cohort.empty:
        raise ValueError(f'Cohort DataFrame is empty, extracting nothing')
    
    stay_ids: list[int] = cohort['stay_id'].tolist()
    hadm_ids: list[int] = cohort['hadm_id'].tolist()

    logger.info('Extracting chartevents for %d ICU stays.', len(stay_ids))
    chart_sql = _query_chartevents(stay_ids)
    chart_raw_pd = con.execute(chart_sql).df()
    chart_raw = pl.from_pandas(chart_raw_pd)

    logger.info('Extracting labevents for %d ICU stays.', len(hadm_ids))
    lab_sql = _query_labevents(hadm_ids)
    lab_raw_pd = con.execute(lab_sql).df()
    lab_raw = pl.from_pandas(lab_raw_pd)

    # Map items to feature names and bin into hours
    vital_id_map = _build_id_to_feature(VITAL_ITEMIDS)
    lab_id_map = _build_id_to_feature(LAB_ITEMIDS)

    logger.info('Aggregating vitals into hourly bins')
    vitals_wide = _raw_to_feature_frame(chart_raw, vital_id_map, is_vitals=True)

    logger.info('Aggregating labs into hourly bins')
    labs_wide = _raw_to_feature_frame(lab_raw, lab_id_map, is_vitals=False)

    skeleton = _build_skeleton(stay_ids)

    skeleton = skeleton.with_columns([
        pl.col('stay_id').cast(pl.Int64),
        pl.col('time_step').cast(pl.Int32),
    ])

    def _safe_join(base: pl.DataFrame, other: pl.DataFrame) -> pl.DataFrame:
        '''Left join other onto base, returns base unchanged if other is empty '''
        if other.is_empty() or 'stay_id' not in other.columns:
            return base
        other = other.with_columns([
            pl.col('stay_id').cast(pl.Int64),
            pl.col('time_step').cast(pl.Int32),
        ])
        return base.join(other, on=['stay_id', 'time_step'], how='left')
    df = _safe_join(skeleton, vitals_wide)
    df = _safe_join(df, labs_wide)


    # Collapse GCS components -> gcs_total
    df = _combine_gcs(df)

    # Ensure all features are present
    df = _ensure_all_feature_columns(df, ALL_FEATURES)

    # Clip out of range results
    logger.info('Clipping out-of-range results')
    df = _clip_values(df, CLIPS_BOUNDS)

    # forward fill within each stay
    logger.info('Foward Filling gaps up to %d hours', MAX_FORWARD_FILL_HOURS)
    df = _forward_fill_with_limit(df, ALL_FEATURES, limit=MAX_FORWARD_FILL_HOURS)

    # Add binary missingness indicator
    df = _add_missingness_indicators(df, ALL_FEATURES)

    # Cast features column to float32 to halve memory allocation
    float_casts = {f: pl.Float32 for f in ALL_FEATURES}
    df = df.cast(float_casts)

    # final column ordering
    missing_cols_order = [f'{f}_missing' for f in ALL_FEATURES]
    final_cols = ['stay_id', 'time_step'] + ALL_FEATURES + missing_cols_order
    df = df.select([c for c in final_cols if c in df.columns])

    # covert to pandas and set MultiIndex
    logger.info('Converting DF to Pandas')
    pdf = df.to_pandas()
    pdf = pdf.sort_values(['stay_id', 'time_step']).reset_index(drop=True)
    pdf = pdf.set_index(['stay_id', 'time_step'])

    logger.info('Extraction complete. Shape: %s | Memory: %.1f MB',
                pdf.shape,
                pdf.memory_usage(deep=True).sum() / 1e6,
    )
    return pdf


def validate_cohort(ts: pd.DataFrame) -> None:
    '''
    Prints a diagnostic summary of the extracted time-series cohort.
    
    Report includes: 
    -----------------
    - n_patients : totoal unique stay_ids
    - n_with_complete_24h : stays with all 24 time steps present
    - mean missingness : per feature fraction of null values (which is computed after forward fill, reflecting true data missingness).
    
    Params
    -------
    ts: 
        DataFrame returned by build_timeseries(), indexed by (stay_id, time_step).
    '''
    
    stay_ids = ts.index.get_level_values('stay_id').unique()
    n_patients = len(stay_ids)
    
    # calculation of complete stays
    row_counts = ts.groupby(level='stay_id').size()
    n_complete = int((row_counts == N_HOURS).sum())
    
    # missingness measure per feature
    missing_cols = [c for c in ts.columns if c.endswith('_missing')]
    if missing_cols:
        miss_rates = ts[missing_cols].mean().rename(
            lambda c: c.replace('_missing', '')
        )
    else:
        feature_cols = [c for c in ts.columns if c in ALL_FEATURES]
        miss_rates = ts[feature_cols].isnull().mean()
    
    # function output section
    print('=' * 50)
    print(' COHORT TIME-SERIES VALIDATION REPORT')
    print('=' * 50)
    print(f' Total patients (stay_ids) : {n_patients:>8,}')
    print(f'  Patients with complete 24 h  : {n_complete:>8,}  '
          f'({100 * n_complete / max(n_patients, 1):.1f}%)')
    print(f' Dataset Shape : {ts.shape}')
    print()
    print(' Mean Missingness per feature (after foward fill): ')
    print(' ' + '-' * 40)
    for feature, rate in miss_rates.sort_values(ascending=False).items():
        bar = '█' * int(rate * 20)
        print(f'  {feature:<20s}  {rate:5.1%}  {bar}')
    print('=' * 60)        
    