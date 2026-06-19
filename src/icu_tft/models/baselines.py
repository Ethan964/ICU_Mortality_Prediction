'''
baselines.py


This file is a prereq. from the tft. We want to be able to import, compute, and analyze our sofa (sequential organ failure assessment) scores
for all icu-stays in our cohort. A threshold of SOFA > 11 has shown to be associated with 
a predicted mortality of greater than 50% in large enough cohorts.

Establishing the baselines before the TFT allows us to:

    1. Demostrate that a learning model can add value beyond what clinicians already know
    2. Isolate the contribution of certain unknown temporal dynamics: 
    if a simple linear model on timed-averaged features nearly matches the TFT, the 
    recurrent architecture adds little, but if the TFT is substantially better, the temporal 
    structure of vitals truely matters.
    3. Produce calibrated probability estimates so that net-benefit / decision-curbe analysis will be meaningful.

will store all metrics in a schema 'compare_baselines()' which extends with TFT results for a unified table.
'''

# env setup and general warnings
from __future__ import annotations
import warnings
import logging
from pathlib import Path
from typing import Any

# viz + data wrangling
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sklearn portion 
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

# ------------------
#       Paths 
# ------------------

MODELS_DIR = Path('models/baselines')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

STATIC_FEATURES: list[str] = [
    # demographics
    'age_years',
    'gender_male',
    'race_White',
    'race_Black',
    'race_Hispanic',
    'race_Asian',
    'first_careunit_encoded',   
    'los_icu_days_prewindow',   
    'gcs_min_6h',
    'gcs_mean_6h',
    'sbp_min_6h',
    'map_min_6h',
    'hr_max_6h',
    'rr_max_6h',
    'temp_min_6h',
    'spo2_min_6h',
    'elix_chf',
    'elix_cardiac_arrhythmias',
    'elix_valvular_disease',
    'elix_pulmonary_circulation',
    'elix_peripheral_vascular',
    'elix_hypertension_uncomplicated',
    'elix_hypertension_complicated',
    'elix_paralysis',
    'elix_other_neurological',
    'elix_chronic_pulmonary',
    'elix_diabetes_uncomplicated',
    'elix_diabetes_complicated',
    'elix_hypothyroidism',
    'elix_renal_failure',
    'elix_liver_disease',
    'elix_peptic_ulcer',
    'elix_aids',
    'elix_lymphoma',
    'elix_metastatic_cancer',
    'elix_solid_tumor',
    'elix_rheumatoid_arthritis',
    'elix_coagulopathy',
    'elix_obesity',
    'elix_weight_loss',
    'elix_fluid_electrolyte',
    'elix_blood_loss_anemia',
    'elix_deficiency_anemias',
    'elix_alcohol_abuse',
    'elix_drug_abuse',
    'elix_psychoses',
    'elix_depression',
    'van_walraven_score',
]

TIMESERIES_MEAN_FEATURES: list[str] = [
    'heart_rate__mean',
    'sbp__mean',
    'dbp__mean',
    'map__mean',
    'resp_rate__mean',
    'temperature__mean',
    'spo2__mean',
    'glucose__mean',
    'wbc__mean',
    'hemoglobin__mean',
    'hematocrit__mean',
    'platelet__mean',
    'sodium__mean',
    'potassium__mean',
    'chloride__mean',
    'bicarbonate__mean',
    'bun__mean',
    'creatinine__mean',
    'bilirubin__mean',
    'lactate__mean',
    'pao2__mean',
    'fio2__mean',
    'paco2__mean',
    'ph__mean',
    'inr__mean',
    'pt__mean',
    'ptt__mean',
    'albumin__mean',
    'alt__mean',
    'ast__mean',
    'alkaline_phosphatase__mean',
]

SOFA_COLUMNS: dict[str, str] = {
    'resp_pao2_fio2': 'pao2_fio2_ratio__min',   
    'coag_platelets':  'platelet__min',         
    'liver_bilirubin': 'bilirubin__max',        
    'cardio_map':      'map__min',              
    'cardio_vaso':     'vasopressor_flag',      
    'cns_gcs':         'gcs_total__min',        
    'renal_creatinine':'creatinine__max',       
    'renal_urine':     'urine_output_24h',      
}
 
TARGET = 'mortality_24h'


def _expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() ==0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


# ============================================
# BASELINE 1 – Sofa Score Ruling
# ============================================
def compute_sofa_score(df: pd.DataFrame) -> pd.Series:
    '''
    Computing the SOFA score for each row using the worst or most abnormal values over 24-h observation window.
    
    params: 
    --------
    df: pd.DataFrame
        Must contain the columns listed in SOFA_COLUMNS.
        + Any missing columns are imputed with 0 for the corresponding sub-score and warning given.
        
    Returns:
    ---------
    pd.Series
        Int SOFA Score (0-24) for each row, indexed like *df.
    '''
    missing = [c for c in SOFA_COLUMNS.values() if c not in df.columns]
    if missing: 
        logger.warning(
            'SOFA columns not found, sub-scores set to 0: %s', missing)
        
    def _get(col: str, fill: float = np.nan) -> pd.Series:
        return df[col].fillna(fill) if col in df.columns else pd.Series(fill, index=df.index)
    
    # ----Respiratory---
    pf = _get('pao2_fio2_ratio__min', fill=400.0)
    resp = pd.cut(
        pf, 
        bins=[-np.inf, 100, 200, 300, 400, np.inf],
        labels=[4,3,2,1,0],
        right=False,
    ).astype(int)
    
    # ----Coagulation---
    plt_ = _get('platelet__min', fill=150.0)
    coag = pd.cut(
        plt_,
        bins=[-np.inf, 20, 50, 100, 150, np.inf],
        labels=[4,3,2,1,0],
        right=False,
    ).astype(int)
    
    # ----Liver---
    bili = _get('bilirubin__max', fill=0.0)
    liver = pd.cut(
        bili,
        bins=[-np.inf, 1.2, 2.0, 6.0, 12.0, np.inf],
        labels=[0, 1, 2, 3, 4],
        right=False,
    ).astype(int)
    
    # ----Cardiovascular---
    map_ = _get('map__min', fill=70.0)
    vaso = _get('vasopressor_flag', fill=0.0).astype(bool)
    cardio = pd.Series(0, index=df.index, dtype=int)
    cardio[map_ < 70] = 1
    cardio[(map_ < 70) & vaso] = 2 
    
    # ---CNS---
    gcs = _get('gcs_total__min', fill=15.0)
    cns = pd.cut(
        gcs,
        bins=[-np.inf, 6, 10, 13, 15, np.inf],
        labels=[4, 3, 2, 1, 0],
        right=False
    ).astype(int)
    
    # ----Renal
    cr = _get('creatinine__max', fill=0.0)
    uo = _get('urine_output_24h', fill=1000.0)
    renal = pd.cut(
        cr, 
        bins=[-np.inf, 1.2, 2.0, 3.5, 5.0, np.inf],
        labels=[0, 1, 2, 3, 4],
        right=False,
    ).astype(int)
    
    renal[uo < 200] = np.maximum(renal[uo < 200], 4)
    renal[uo < 500] = np.maximum(renal[uo < 500], 3)
    
    sofa = resp + coag + liver + cardio + cns + renal
    sofa.name = 'sofa_score'
    return sofa.clip(0, 24)


    
def evaluate_sofa_baseline(
    df: pd.DataFrame,
    threshold: int = 11,
    verbose: bool = True,
) -> dict[str, Any]: 
    '''
    Evaluates morality based on SOFA >= threshold as a binary predictor
    
    Params
    ------
    df = pd.DataFrame
        must contain SOFA component columns and morality_24h
    threshold: int
        SOFA score cut-off for a positive prediction (default 11).
    verbose: bool
        If true, print a formatted metrics table.
    
    Returns
    ------
    dict with keys: sofa_score (Series), metrics (dict), threshold (int)
    '''
    
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found in DataFrame.")
 
    y_true = df[TARGET].astype(int).values
    sofa = compute_sofa_score(df)
    y_pred = (sofa >= threshold).astype(int).values
 
    auroc = roc_auc_score(y_true, sofa.values)
    auprc = average_precision_score(y_true, sofa.values)
 
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
 
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
 
    metrics: dict[str, Any] = {
        'auroc':       round(auroc, 4),
        'auprc':       round(auprc, 4),
        'sensitivity': round(sensitivity, 4),
        'specificity': round(specificity, 4),
        'tp':          tp,
        'tn':          tn,
        'fp':          fp,
        'fn':          fn,
        'threshold':   threshold,
        'n':           len(y_true),
        'prevalence':  round(y_true.mean(), 4),
    }
 
    if verbose:
        print('\n' + '=' * 60)
        print('BASELINE 1 — SOFA Score Rule')
        print(f'  Threshold  : SOFA ≥ {threshold}')
        print(f'  N          : {metrics['n']:,}')
        print(f'  Prevalence : {metrics['prevalence']:.1%}')
        print('-' * 60)
        print(f'  AUROC      : {metrics['auroc']:.4f}')
        print(f'  AUPRC      : {metrics['auprc']:.4f}')
        print(f'  Sensitivity: {metrics['sensitivity']:.4f}  (recall / TPR)')
        print(f'  Specificity: {metrics['specificity']:.4f}  (TNR)')
        print(f'  TP / TN / FP / FN: {tp} / {tn} / {fp} / {fn}')
        print('=' * 60 + '\n')
 
    return {'sofa_score': sofa, 'metrics': metrics, 'threshold': threshold}



# ============================================
# BASELINE 2 – Penalized LogReg 
# ============================================

def _build_logreg_pipeline(
    feature_cols: list[str],
) -> Pipeline:
    '''
    constructs a fully encapsulated sklearn pipeline.
    
    Internals
    ------------
    ColumnTransformer
        -------- numeric branch: SimpleImputer (median) -> StandardScaler
    LogisticRegression(penalty='elasticnet), solver='saga', max_iter=2000)
    
    Learning alpha and penalty coeffcs are tuned in func. below.
    '''
    
    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, feature_cols),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )
    clf = LogisticRegression(
        solver='saga',
        l1_ratio=0.5,
        class_weight='balanced',
        max_iter=2000,
        random_state=617
    )
    # notes: l1_ratio=0 -> Ridge, l1_ratio=1 -> Lasso
    
    return Pipeline(steps=[('preprocessor', preprocessor), ('clf', clf)])

def fit_logreg_baseline(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    calibrate: bool = True,
    cv_folds: int = 5,
    verbose: bool = True,
    plot: bool = True,
) -> dict[str, Any]:
    '''
    Fits a calibrated ElasticNet logistic regression model and evaluates it.
    
    Params
    --------
    df: pd.DataFrame
        Must contain feature_cols and mortality_24h
    feature_cols : list[str] or None
        Columns to use as features/covariates. Defaults to all recognised static + timesereis mean columns that are present in df
    calibrate: bool
        If true, wrap the tuned estimator in 'CalibratedClassifierCV
    cv_folds : int
        Number of stratified CV folds for hyper-parameter search
    verbose: bool
        If true, prints metrics and coefficients table.
    plot: bool
        Saves a calibration curve figure to 'models/baselines/'
    
    Returns
    --------
    dict wt keys: pipeline, calibrated_pipeline (if calibrate), metrics, feature_cols, coef_df    
    '''
    
    if TARGET not in df.columns:
        raise KeyError(f"Target Column '{TARGET}' not found in DataFrame")
    
    # time to determine feature cols
    
    if feature_cols is None:
        all_candidates = STATIC_FEATURES + TIMESERIES_MEAN_FEATURES
        feature_cols = [c for c in all_candidates if c in df.columns]
        if not feature_cols:
            raise ValueError(
                'No recognised feature columns found in DataFrame.'
                'Pass `feature_cols` explicitly'
            )
        logger.info('Auto-selected %d feature columns.', len(feature_cols))
        
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning(
            'REquested feature columns missing from df - dropping: %s', missing_cols
        )
        feature_cols = [c for c in feature_cols if c in df.columns]
        
    X = df[feature_cols].copy()
    y= df[TARGET].astype(int).values
    
    param_grid = {
        'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0],
        'clf__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    }            
    base_pipeline = _build_logreg_pipeline(feature_cols)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=617)
    if verbose:
        print(f'\nRunning {cv_folds}-fold stratified CV over '
              f'{len(param_grid['clf__C']) * len(param_grid['clf__l1_ratio'])} '
              'hyper-parameter combinations …')
        
    gs = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    gs.fit(X, y)
    best_pipeline: Pipeline = gs.best_estimator_
    best_params = gs.best_params_
    
    if verbose:
        print(f' Best Params: {best_params}')
        print(f' CV AUROC: {gs.best_score_:.4f}')
        
        
    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    if calibrate:
        calibrated = CalibratedClassifierCV(
            estimator=best_pipeline,
            method="isotonic",
            cv=cv_folds,
        )
        calibrated.fit(X, y)
        y_prob = calibrated.predict_proba(X)[:, 1]
        final_estimator = calibrated
    else:
        y_prob = best_pipeline.predict_proba(X)[:, 1]
        final_estimator = best_pipeline
        
        
    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    auroc  = roc_auc_score(y, y_prob)
    auprc  = average_precision_score(y, y_prob)
    brier  = brier_score_loss(y, y_prob)
    ece    = _expected_calibration_error(y, y_prob)

    metrics: dict[str, Any] = {
        'auroc':       round(auroc,  4),
        'auprc':       round(auprc,  4),
        'brier_score': round(brier,  4),
        'ece':         round(ece,    4),
        'best_C':      best_params['clf__C'],
        'best_l1_ratio': best_params['clf__l1_ratio'],
        'n_features':  len(feature_cols),
        'n':           len(y),
        'prevalence':  round(y.mean(), 4),
    }   
    if verbose:
        print('\n' + '=' * 60)
        print('BASELINE 2 — ElasticNet Logistic Regression')
        print(f'  N          : {metrics['n']:,}')
        print(f'  Prevalence : {metrics['prevalence']:.1%}')
        print(f'  Features   : {metrics['n_features']}')
        print(f'  Best C     : {metrics['best_C']}')
        print(f'  Best l1_ratio: {metrics['best_l1_ratio']}')
        print('-' * 60)
        print(f'  AUROC      : {metrics['auroc']:.4f}')
        print(f'  AUPRC      : {metrics['auprc']:.4f}')
        print(f'  Brier score: {metrics['brier_score']:.4f}  (lower = better)')
        print(f'  ECE        : {metrics['ece']:.4f}  (lower = better)')
        print('=' * 60)
        
    coef_df = _extract_coefficients(best_pipeline, feature_cols)
    if verbose: 
        _print_coef_table(coef_df)
    if plot:
        _plot_calibration_curve(y, y_prob, save_dir=MODELS_DIR)
        
        
    joblib.dump(best_pipeline, MODELS_DIR / 'logreg_elasticnet.joblib')
    if calibrate:
        joblib.dump(calibrated, MODELS_DIR / 'logreg_elasticnet_calibrated.joblib')
    logger.info('Pipelines saved to %s', MODELS_DIR)

    return {
        'pipeline':            best_pipeline,
        'calibrated_pipeline': final_estimator if calibrate else None,
        'metrics':             metrics,
        'feature_cols':        feature_cols,
        'coef_df':             coef_df,
        'best_params':         best_params,
    }

def _extract_coefficients(
    pipeline: Pipeline,
    feature_cols: list[str],
) -> pd.DataFrame:
    '''
    Extracts logistic regression coefficients mapped to original feature names.
    
    Returns a Df with columns: feature, coefficient, abs_coeff.
    Sorted by descending abs. value
    '''
    
    clf = pipeline.named_steps['clf']
    
    coef = clf.coef_[0]
    
    preprocessor = pipeline.named_steps['preprocessor']
    try:
        transformed_names = list(
            preprocessor.get_feature_names_out()
        )
    except AttributeError:
        transformed_names = feature_cols
        
    coef_df = pd.DataFrame(
        {
            'feature' : transformed_names[: len(coef)],
            'coefficient' : coef,
            'abs_coefficient' : np.abs(coef),
        }
    ).sort_values('abs_coefficient', ascending=False)
    
    return coef_df.reset_index(drop=True)

def _print_coef_table(coef_df: pd.DataFrame, top_n: int =15) -> None:
    
    '''
    Print top-N positive and top_N negative coeffficients.
    '''
    
    positive = (
        coef_df[coef_df['coefficient'] > 0].nlargest(top_n, 'coefficient')[['feature', 'coefficient']]
    )
    negative = (
        coef_df[coef_df['coefficient'] < 0].nsmallest(top_n, 'coefficient')[['feature', 'coefficient']]
    )
    
    print(f'\n{'-' * 55}')
    print(f' Top {top_n} POSITIVE coefficients (higher mortality risk)')
    print(f'\n{'-' * 55}')
    for _, row in positive.iterrows():
        bar = "█" * max(1, int(abs(row["coefficient"]) * 8))
        print(f' {row['feature']:<38s} {row['coefficient']:+.4f} {bar}')
    print()
    
    
    # ------------------------------------------------------------------
    # Plot Calibration Curve
    # ------------------------------------------------------------------

def _plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_dir: Path = MODELS_DIR,
) -> Path:
    '''
    Saves a calibration curve to `save_dir`
    
    Returns the path of the saved figure.
    '''
    
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    fig, axes = plt.subplots(1, 2, figsize=(12,10))
    ax = axes[0]
    ax.plot([0,1], [0,1], linestyle='--', color='grey', label='Perfect Calibration') 
    ax.plot(mean_pred, fraction_pos, marker='o', color='#2563EB', label='ElasticNet LR')
    ax.set_xlabel('Mean predicted probabiltiy')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration curve (reliability diagram)')
    ax.legend()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.grid(alpha=0.3)
    
    ax2 = axes[1]
    ax2.hist(
        y_prob[y_true == 0], bins=40, alpha=0.6, color='#64748B', label='Survived (y=0)'
    )
    ax2.hist(
        y_prob[y_true == 1], bins=40, alpha=0.7, color='#DC2626', label='Died (y=1)'
    )
    ax2.set_xlabel('Predicted Probability of 24-h mortality')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    fig.suptitle('ElasticNet LogisticRegression - Calibration', fontsize=13, y=1.01)
    fig.tight_layout()
    
    out_path = save_dir / 'logreg_calibration_curve.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Calibration curve saved to %s', out_path)
    return out_path


    # ------------------------------------------------------------------
    # compare_baselines()
    # ------------------------------------------------------------------

def compare_baselines(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    sofa_threshold: int = 11,
    calibrate_logreg: bool = True,
    cv_folds: int = 5,
    verbose: bool = True,
    plot: bool = True,
) -> pd.DataFrame:
    '''
    Fits both baselines on `df` and returns a clean comparison dataframe.
    
    This is the canonical entry point for the baseline module. The returned DataFrame schema is designed to accepet a TFT row later.
     
    schema: 
        model | auroc | auprc | brier_score | ece | sensitivity | specificity | notes
        
    Params
    ----------------
    df: pd.DataFrame
        the combined dataframe containing static features, time-averaged vitals and signs, SOFA component columns and `mortality_24h`.
    feature_Cols: list[str]
        Explicit feature list for the logistic regression. Auto-detected from recognised column names if None.
    sofa_threshold: int
        SOFA >= threshold for binary predictor modeling (default = 11).
    calibrate_logreg: bool
        whether or not to calibrate the logistic regression clf
    cv_folds: int
        folds for stratified/k-fold cross-validation
    verbose: bool
        deciding whether to print results/metrics/coefficients or not.
    
    Returns
    ----------------
    pd.DataFrame with one row per model and columns containing:
        model | auroc | auprc | brier_score | ece | sensitivity | specificity | notes
    '''
    sofa_result = evaluate_sofa_baseline(
        df, threshold=sofa_threshold, verbose=verbose
    )
    sm = sofa_result['metrics']
    
    logreg_result = fit_logreg_baseline(
        df, 
        feature_cols=feature_cols,
        calibrate=calibrate_logreg,
        cv_folds=cv_folds,
        verbose=verbose,
        plot=plot
    )
    lm = logreg_result['metrics']
    
    rows = [
        {
            'model' : 'SOFA Binary Predictor',
            'auroc' : sm['auroc'],
            'auprc' : sm['auprc'],
            'brier_score' : np.nan,
            'ece' : np.nan,
            'sensitivity' : sm['sensitivity'],
            'specificity' : sm['specificity'],
            'notes' : f'threshold≥{sofa_threshold}; current clinical standard.',
        },
        {
            'model' : 'ElasticNet LR',
            'auroc' : lm['auroc'],
            'auprc' : lm['auprc'],
            'brier_score' : lm['brier_score'],
            'ece' : lm['ece'],
            'sensitivity' : np.nan,
            'specificity' : np.nan,
            'notes' : (
                f'C={lm['best_C']}, l1_ratio={lm['best_l1_ratio']}; '
                f'Isotonic calibration; {lm['n_features']} features.' 
            ), 
        },
    ]
    
    compare_df = pd.DataFrame(rows).set_index('model')
    
    if verbose:
        print('\n' + '=' * 60)
        print('BASELINE COMPARISON TABLE')
        print('=' * 60)
        print(compare_df.to_string())
        print(
            '\n(TFT results row to be appended after model training.)\n'
        )
    out_csv = MODELS_DIR / 'baseline_comparison.csv'
    compare_df.to_csv(out_csv)
    logger.info('comparison table save to %s', out_csv)
    
    return compare_df


    # ------------------------------------------------------------------
    # synthetic dataset creator
    # ------------------------------------------------------------------
    
def _make_syntehtic_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_pos = max(1, int(n * 0.15)) 
    # we can impute any kind of mortality rate 
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    rng.shuffle(y)
    
    data: dict[str, Any] = {TARGET: y}

    # We have our static, personal demographic identifiers
    data["age_years"]            = rng.integers(18, 90, n).astype(float)
    data["gender_male"]          = rng.integers(0, 2, n).astype(float)
    data["van_walraven_score"]   = rng.integers(-7, 30, n).astype(float)
    data["gcs_min_6h"]           = rng.integers(3, 15, n).astype(float)
    data["gcs_mean_6h"]          = rng.uniform(3, 15, n)
    data["sbp_min_6h"]           = rng.uniform(60, 180, n)
    data["map_min_6h"]           = rng.uniform(40, 110, n)
    data["first_careunit_encoded"] = rng.integers(0, 6, n).astype(float)
    data["los_icu_days_prewindow"] = rng.exponential(2, n)

    for flag in [
        "race_White", "race_Black", "race_Hispanic", "race_Asian",
        "elix_chf", "elix_renal_failure", "elix_liver_disease",
        "elix_metastatic_cancer", "elix_coagulopathy",
    ]:
        data[flag] = rng.integers(0, 2, n).astype(float)

    # Timeseries mean features
    data["heart_rate__mean"]   = rng.uniform(50, 150, n)
    data["sbp__mean"]          = rng.uniform(70, 180, n)
    data["map__mean"]          = rng.uniform(50, 120, n)
    data["resp_rate__mean"]    = rng.uniform(8, 40, n)
    data["spo2__mean"]         = rng.uniform(85, 100, n)
    data["temperature__mean"]  = rng.uniform(35, 40, n)
    data["creatinine__max"]    = rng.exponential(1.5, n)
    data["bilirubin__max"]     = rng.exponential(1.0, n)
    data["platelet__min"]      = rng.uniform(20, 400, n)
    data["lactate__mean"]      = rng.exponential(1.5, n)
    data["bun__mean"]          = rng.uniform(5, 100, n)
    data["glucose__mean"]      = rng.uniform(60, 300, n)
    data["inr__mean"]          = rng.uniform(0.8, 5, n)
    data["pao2__mean"]         = rng.uniform(60, 400, n)
    data["fio2__mean"]         = rng.uniform(0.21, 1.0, n)

    # SOFA components
    data["pao2_fio2_ratio__min"] = data["pao2__mean"] / np.maximum(data["fio2__mean"], 0.21)
    data["gcs_total__min"]       = rng.integers(3, 15, n).astype(float)
    data["vasopressor_flag"]     = rng.integers(0, 2, n).astype(float)
    data["urine_output_24h"]     = rng.exponential(1500, n)

    # give slight noise 
    data["creatinine__max"] += y * rng.uniform(0.5, 2.0, n)
    data["lactate__mean"]   += y * rng.uniform(1.0, 3.0, n)
    data["gcs_total__min"]  -= y * rng.uniform(1.0, 4.0, n)

    return pd.DataFrame(data)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
    )
    print('Running baselines on synthetic test, slight sanity check.')
    synthetic = _make_syntehtic_df(n=600, seed=617)
    results = compare_baselines(
        synthetic,
        sofa_threshold=11, 
        calibrate_logreg=True,
        cv_folds=3,
        verbose=True,
        plot=True,
)
    print(f'Final Comparison DataFrame:')
    print(results)