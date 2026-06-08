'''
static_features.py

============================================
Extracts and processes time-invariant statis features within the ICU 24-hour mortality 
prediction task on MIMIC-IV v3.1.

Pipeline 
----------
1. Demographic info: Age, race, Gender (all binned).
2. Case-by-case Context: Admission Type, First-careunit, Insurance.
3. Comorbidities: Elixhauer Index (specifically defined weights from van Walraven) from diagnoses_icd.
4. Severity: Min MAP, Max Lactate, Min GCS from the first 6 hours of ICU stay.

'''

from __future__ import annotations
import logging
from typing import Final
import duckdb
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# feature groupings
# will export for future shap explainer plots and DL attention visualizations

FEATURE_GROUPS: Final[dict[str, str]] = {
    'age' : 'Demographics',
    'age_binned' : 'Demographics',
    'gender_M' : 'Demographics',
    
    # Race
    'race_White' : 'Demographics',
    'race_Black' : 'Demographics',
    'race_Hispanic' : 'Demographics',
    'race_Asian' : 'Demographics',
    'race_Other' : 'Demographics',
    'race_unknown' : 'Demographics',
    
    # icu admission context
    
    'admission_type_Elective' : 'Admission Context',
    'admission_type_Emergency' : 'Admission Context',
    'admission_type_Urgent' : 'Admission Context',
    'insurance_Medicare' : 'Admission Context',
    'insurance_Medicaid' : 'Admission Context',
    'insurance_Private' : 'Admission Context',
    'insurance_Other' : 'Admission Context',
    'insurance_Self_pay' : 'Admission Context',
    'careunit_MICU' : 'Admission Context',
    'careunit_SICU' : 'Admission Context',
    'careunit_CCU' : 'Admission Context',
    'careunit_CSRU' :  'Admission Context',
    'careunit_Other' : 'Admission Context',
    
    # Comorbidities
    
    'elixhauser_val_walraven_score' : 'Comorbodities',
    'elix_chf' : 'Comorbodities',
    'elix_arrhythmia' : 'Comorbodities',
    'elix_valvular' : 'Comorbodities',
    'elix_pcd' : 'Comorbodities',
    'elix_pvd' : 'Comorbodities',
    'elix_htn_uncomp' : 'Comorbodities',
    'elix_htn_comp' : 'Comorbodities', 
    'elix_paralysis' : 'Comorbodities',
    'elix_neuro_other' : 'Comorbidities',
    'elix_cpd': 'Comorbidities',
    'elix_diabetes_uncomp': 'Comorbidities',
    'elix_diabetes_comp': 'Comorbidities',
    'elix_hypothyroid': 'Comorbidities',
    'elix_renal_failure': 'Comorbidities',
    'elix_liver_disease': 'Comorbidities',
    'elix_pud': 'Comorbidities',
    'elix_hiv': 'Comorbidities',
    'elix_lymphoma': 'Comorbidities',
    'elix_mets': 'Comorbidities',
    'elix_tumor': 'Comorbidities',
    'elix_rheum': 'Comorbidities',
    'elix_coag': 'Comorbidities',
    'elix_obesity': 'Comorbidities',
    'elix_weightloss': 'Comorbidities',
    'elix_fluid_lytes': 'Comorbidities',
    'elix_blood_loss': 'Comorbidities',
    'elix_anemia': 'Comorbidities',
    'elix_alcohol': 'Comorbidities',
    'elix_drugs': 'Comorbidities',
    'elix_psych': 'Comorbidities',
    'elix_depression': 'Comorbidities',
    
    # severity
    'severity_min_map_6h' : 'Severity Proxies',
    'severity_max_lactate_6h' : 'Severity Proxies',
    'severity_min_gcs_6h' : 'Severity Proxies'
}
    # weights derived from walraven (2009)
VAL_WALRAVEN_WEIGHTS: Final[dict[str, int]] = {
    'elix_chf': 7, 'elix_arrhythmia': 5, 'elix_valvular': -1, 
    'elix_pcd': 4, 'elix_pvd': 2, 'elix_htn_uncomp': 0, 
    'elix_htn_comp': 0, 'elix_paralysis': 7, 'elix_neuro_other': 6, 
    'elix_cpd': 3, 'elix_diabetes_uncomp': 0, 'elix_diabetes_comp': 0, 
    'elix_hypothyroid': 0, 'elix_renal_failure': 5, 'elix_liver_disease': 11, 
    'elix_pud': 0, 'elix_hiv': 0, 'elix_lymphoma': 9, 
    'elix_mets': 12, 'elix_tumor': 4, 'elix_rheum': 0, 
    'elix_coag': 3, 'elix_obesity': -4, 'elix_weightloss': 6, 
    'elix_fluid_lytes': 5, 'elix_blood_loss': -2, 'elix_anemia': -2, 
    'elix_alcohol': 0, 'elix_drugs': -7, 'elix_psych': -5, 
    'elix_depression': -3
}

# ============================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================

def extract_demographics_and_context(cohort: pd.DataFrame) -> pd.DataFrame:
    '''
    Extracts demographic and contextual info from cohort table
    '''
    df = cohort.copy()
    
    
    # binning cohort based on age
    df['age'] = df['anchor_age'].astype(float)
    df['age_binned'] = pd.cut(
        df['age'],
        bins=[0, 40, 60, 75, 100],
        labels=[0, 1, 2, 3],
        right=False
    ).astype(int)
    
    # gender binning
    df['gender_M'] = df['gender'].map(({'M' : 1, 'F' : 0}).astype(int))
    
    # race (being categorical, one-hot encoding)
    race_map = {
        'WHITE': 'White', 'WHITE - OTHER EUROPEAN': 'White', 'WHITE - RUSSIAN': 'White', 'WHITE - BRAZILIAN': 'White',
        'BLACK/AFRICAN AMERICAN': 'Black', 'BLACK/AFRICAN': 'Black', 'BLACK/CAPE VERDEAN': 'Black', 'BLACK/CARIBBEAN ISLAND': 'Black',
        'HISPANIC/LATINO - PUERTO RICAN': 'Hispanic', 'HISPANIC OR LATINO': 'Hispanic', 'HISPANIC/LATINO - DOMINICAN': 'Hispanic', 'HISPANIC/LATINO - GUATEMALAN': 'Hispanic', 'HISPANIC/LATINO - CUBAN': 'Hispanic', 'HISPANIC/LATINO - SALVADORAN': 'Hispanic', 'HISPANIC/LATINO - MEXICAN': 'Hispanic', 'HISPANIC/LATINO - CENTRAL AMERICAN': 'Hispanic', 'HISPANIC/LATINO - COLUMBIAN': 'Hispanic', 'HISPANIC/LATINO - HONDURAN': 'Hispanic',
        'ASIAN': 'Asian', 'ASIAN - CHINESE': 'Asian', 'ASIAN - ASIAN INDIAN': 'Asian', 'ASIAN - VIETNAMESE': 'Asian', 'ASIAN - KOREAN': 'Asian', 'ASIAN - SOUTH EAST ASIAN': 'Asian', 'ASIAN - FILIPINO': 'Asian', 'ASIAN - JAPANESE': 'Asian',
        'UNKNOWN': 'Unknown', 'UNABLE TO OBTAIN': 'Unknown', 'PATIENT DECLINED TO ANSWER': 'Unknown', 'OTHER': 'Other', 'MULTIPLE RACE/ETHNICITY': 'Other', 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'Other', 'AMERICAN INDIAN/ALASKA NATIVE': 'Other'
    }

    df['race_group'] = df['race'].map(race_map).fillna('Unknown')
    df = pd.get_dummies(df, columns=['race_group'], prefix='race', dtype=int)
    
    # admission type
    adm_map = {'ELECTIVE' : 'Elective', 'URGENT' : 'Urgent', 'EMERGENCY' : 'Emergency', 'EU OBSERVATION' : 'EMERGENCY', 'DIRECT EMER.' : 'Emergency', 'DIRECT OBSERVATION' : 'Emergency'}
    df['adm_group'] = df['admission_type'].map(adm_map).fillna('Emergency')
    df = pd.get_dummies(df, columns=['adm_group'], prefix='admission_type', dtype=int)
    
    # first care unit identifier
    cu_map = {'MICU' : 'MICU', 'SICU' : 'SICU', 'CCU' : 'CCU', 'CRSU' : 'CRSU'}
    df['cu_group'] = df['first_careunit'].map(lambda x: cu_map.get(x, 'Other'))    
    df = pd.get_dummies(df, columns=['cu_group'], prefix='careunit', dtype=int)
    
    # insurance type
    df['ins_group'] = df['insurance'].map(lambda x: x if x in ['Medicare', 'Medicaid', 'Other'] else ('Private' if x != 'Self Pay' else 'Self_pay'))
    df = pd.get_dummies(df, columns=['ins_group'], prefix='insurance', dtype=int)
    
    # drop all non-engineered feature cols
    cols_to_keep = ['stay_id'] + [col for col in df.columns if col in FEATURE_GROUPS]
    return df[cols_to_keep]

def extract_elixhauser(cohort: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    '''
    Returns Elixhauser comorbidity index for cohort using ICD-9 and ICD-10 definintations 
    sourced from various literature: val Walraven (2009) & Quan et al (2005).
    '''
    
    hadm_ids = tuple(cohort['hadm_id'].unique())
    query = '''
    SELECT hadm_id, icd_code, icd_version
    FROM mimic_hosp.diagnoses_icd
    WHERE hadm_id IN {hadm_ids}
    '''
    
    icd_df = con.execute(query).df()
    
    # import index calculation algos form Quan et al (2005) for defining comorbidities in ICD-9 & 10.
    regex_map = {
        'elix_chf': r'^(39891|428|402[012]1|404[012][13]|I099|I110|I13[02]|I255|I420|I42[56789]|I43|I50)',
        'elix_arrhythmia': r'^(426[01379]|427[012346789]|427[1-9]|I44[123]|I45[69]|I47|I48|I49)',
        'elix_valvular': r'^(0932|394|395|396|397|424|746[3456]|V422|V433|I05|I06|I07|I08|I091|I098|I34|I35|I36|I37|I38|I39|Q23[0123]|Z95[234])',
        'elix_pcd': r'^(416|4179|I26|I27|I28[089])',
        'elix_pvd': r'^(440|441|442|443[1-9]|444|4471|449|7472|V434|I70|I71|I73[189]|I77[10]|I79[02]|K55[189]|Z95[89])',
        'elix_htn_uncomp': r'^(401|I10)',
        'elix_htn_comp': r'^(402[012]0|403[012][01]|404[012][012]|405|I119|I12[09]|I13[0129]|I15)',
        'elix_paralysis': r'^(342|343|344[0-6]|3449|G81|G82|G83)',
        'elix_neuro_other': r'^(3319|3320|3334|3335|334|335|340|341|345|3481|3483|7803|7843|G10|G11|G12|G20|G21|G22|G25[45]|G31[289]|G32|G35|G36|G37|G40|G41|G93[14]|R470|R56)',
        'elix_cpd': r'^(490|491|492|493|494|495|496|500|501|502|503|504|505|5064|J40|J41|J42|J43|J44|J45|J46|J47|J60|J61|J62|J63|J64|J65|J66|J67|J684|J70[13])',
        'elix_diabetes_uncomp': r'^(250[0123]0|250[0123]1|E10[019]|E11[019]|E12[019]|E13[019]|E14[019])',
        'elix_diabetes_comp': r'^(250[456789]|E10[2-8]|E11[2-8]|E12[2-8]|E13[2-8]|E14[2-8])',
        'elix_hypothyroid': r'^(243|244|E00|E01|E02|E03|E890)',
        'elix_renal_failure': r'^(584|585|586|V420|V451|V56|N18|N19|N25|Z49|Z940|Z992)',
        'elix_liver_disease': r'^(0702|0703|0704|0705|0706|0709|570|571|572[1-8]|573[123]|V427|B18|I85|I864|I982|K70|K71[13457]|K72[19]|K73|K74|K76[023489]|Z944)',
        'elix_pud': r'^(531[79]|532[79]|533[79]|534[79]|K25[79]|K26[79]|K27[79]|K28[79])',
        'elix_hiv': r'^(042|043|044|V08|B20|B21|B22|B24|Z21)',
        'elix_lymphoma': r'^(200|201|202|2030|2386|C81|C82|C83|C84|C85|C88|C90[02]|C96)',
        'elix_mets': r'^(196|197|198|199|C77|C78|C79|C80)',
        'elix_tumor': r'^(14[0-9]|15[0-9]|16[0-9]|17[0-9]|18[0-9]|19[0-5]|C[0-1]|C2|C3|C4|C5|C6|C7[0-6]|C97)',
        'elix_rheum': r'^(446|710[0-4]|714[0-2]|7148|725|M05|M06|M315|M32|M33|M34|M351|M353|M360)',
        'elix_coag': r'^(286|287[1-5]|D65|D66|D67|D68|D69[13456])',
        'elix_obesity': r'^(2780|V853|V854|E66|Z683|Z684)',
        'elix_weightloss': r'^(26[0-9]|7994|E4[0-6]|R634)',
        'elix_fluid_lytes': r'^(276|E86|E87)',
        'elix_blood_loss': r'^(2800|D500)',
        'elix_anemia': r'^(280[1-9]|281|D50[189]|D51|D52|D53)',
        'elix_alcohol': r'^(291[0-9]|303[09]|3050|V113|F10|Z502)',
        'elix_drugs': r'^(292[0-9]|304[0-9]|305[2-9]|V6542|F11|F12|F13|F14|F15|F16|F18|F19|Z503)',
        'elix_psych': r'^(295[0-9]|296[045]|297[0-9]|298[0-9]|F20|F22|F23|F24|F25|F28|F29|F302|F312|F315)',
        'elix_depression': r'^(296[23]|3004|311|F32|F33|F341)'
    }
    res_dfs = []
    for col_name, pattern in regex_map.items():
        matched = icd_df['icd_code'].str.match(pattern)
        res_dfs.append(
            icd_df[matched][['hadm_id']].drop_duplicates().assign(**{col_name: 1})
        )
    if res_dfs:
        merged_elix = res_dfs[0]
        for idx in range(1, len(res_dfs)):
            merged_elix = pd.merge(merged_elix, res_dfs[idx], on='hadm_id', how='outer')
        merged_elix = merged_elix.fillna(0).astype(int)
    else:
        merged_elix = pd.DataFrame({'hadm_id': pd.Series(dtype=int)})
        for col_name in regex_map:
            merged_elix[col_name] = 0
            
    # calc. each walraven score
    score = pd.Series(np.zeros(len(merged_elix)), index=merged_elix.index)
    for col_name, weight in VAL_WALRAVEN_WEIGHTS.items():
        if col_name in merged_elix.columns:
            score += merged_elix[col_name] * weight
    
    final_df = cohort[['stay_id', 'hadm_id']].merge(merged_elix, on='hadm_id', how='left').drop(columns=['hadm_id'])
    return final_df.fillna(0)

def extract_severity_proxies(cohort: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    '''
    Exracts min Map, max lactate, and min GCS score from the first 6 hours of the ICU stay.
    This function serves as a early-warning static baseline feature.
    
    '''
    stay_ids = tuple(cohort['stay_id'].unique())
    
    # ids for min_map : 220052, max_latacte : 50813, gcs : {223900, 223901, 220739}
    
    query = '''
        SELECT 
            ce.stay_id, 
            MIN(CASE WHEN ce.itemid = 220052 THEN ce.valuenum END) AS severity_min_map_6h,
            MAX(CASE WHEN ce.itemid = 50813 THEN ce.valuenum END) AS severity_max_lactate_6h
        FROM mimic_icu.chartevents ce
        INNER JOIN mimic_icu.icustays ie ON ce.stay_id = ie.stay_id
        WHERE ce.stay_id IN {stay_ids}
            AND ce.itemid IN (220052, 50813)
            AND ce.charttime >= ie.intime
            AND ce.charttime <= ie.intime + INTERVAL '6 hours
            AND ce.valuenum IS NOT NULL
        GROUP BY ce.stay_id
    '''
    proxies_df = con.execute(query).df()
    
    # need second query with same premise, substitue gcs component ids 
    gcs_query = f'''
        WITH gcs_components AS (
            SELECT 
                ce.stay_id, 
                ce.charttime,
                SUM(ce.valuenum) as gcs_total
            FROM mimic_icu.chartevents ce
            INNER JOIN mimic_icu.icustays ie ON ce.stay_id = ie.stay_id
            WHERE ce.stay_id IN {stay_ids}
                AND ce.itemid IN (223900, 223901, 220739)
                AND ce.charttime >= ie.intime 
                AND ce.charttime <= ie.intime + INTERVAL '6 hours'
            GROUP BY ce.stay_id, ce.charttime
            HAVING COUNT(DISTINCT ce.itemid) = 3
        )
        SELECT 
            stay_id, 
            MIN(gcs_total) AS severity_min_gcs_6h
        FROM gcs_components
        GROUP BY stay_id
    '''
    
    gcs_df = con.execute(query).df()
    
    merged_proxies = pd.merge(cohort[['stay_id']], proxies_df, on='stay_id', how='left')
    merged_proxies = pd.merge(merged_proxies, gcs_df, on='stay_id', how='left')
    return merged_proxies

def build_static_features(cohort: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    '''
    Engine of script: 
    Executes all preexisting functions and complies on stay_id
    '''
    logger.info('Extracting demographic, context, and admission context.')
    df_demo = extract_demographics_and_context(cohort)
    
    logger.info('Extracting Elixhauser Comorbodity Scores.')
    df_elix = extract_elixhauser(cohort, con)
    
    logger.info('Extracting early severity proxies from 6 hours of ICU stay.')
    df_proxy = extract_severity_proxies(cohort, con)
    
    static_df = df_demo.merge(df_elix, on='stay_id', how='left')
    static_df = static_df.merge(df_proxy, on='stay_id', how='left')
    return static_df 

def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    '''
    outputs cohort data integrity, missingness, and feature distributions
    '''
    
    print('=' * 50)
    print('STATIC FEATURES SUMMARY REPORT')
    print('=' * 50)
    
    for col in df.columns:
        if col == 'stay_id':
            continue
        n_unique = df[col].nunique()
        missing_rate = df[col].isnull().mean()
        
        print(f'\nFeature: {col} | Unique : {n_unique} | Missing : {missing_rate:.1%}')
        print('=' * 50)
        
        if n_unique <= 10:
            dist = df[col].value_counts(dropna=False, normalize=True).sort_index()
            for val, pct in dist.items():
                print(f' {val} : {pct:.1%}')
        else:
            desc= df[col].describe()
            print(f' Mean: {desc['mean']:.2f} | Std Dev : {desc['std']:.2f}')
            print(f' Min: {desc['min']:.2f} | Median: {desc['50%']:.2f} | Max: {desc['max']:.2f}')
            
    print('=' * 50)
    