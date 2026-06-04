/*
------------------------------------------------------------------------
MIMIC-IV 3.1 24-Hour Mortality Predicction Cohort
------------------------------------------------------------------------

********************************
        VERSION  CONTROL
********************************
Compatable with: DuckDB ≥ 0.9 & PostgreSQL ≥ 13

Schema prefixes: mimc_hosp.* (previously mimicc_ccore.*) for v3.x of MIMIC-IV

------------------------------------
        MORTALITY LABEL DEF
------------------------------------

1. mortality_24h (primary task label)
    Source: mimic_icu.icustays.outtime
            mimic_hosp.patients.dod         (date of death, temporal precision = day)
    
    Logic: Patient X is labeled 1 if their date of death (dod) falls on the 
            same calendar day as, or beofre, the date that is exactly 24 hours 
            after ICU admission (intime + INTERVAL '24h'). Beccause dod has day-level
            precision in MIMIC-IV (time-of-death is not recorded), we treated dod <= DATE(intime + INTERVAL '24h')
            as the conservative criteria. This means a patient who died at 23:59 on 
            day 1 and one who died at 00:01 on day 2 are both labeled as 1, which 
            slightly inflates the positive class by at most one boundary day. An
            alternative sharper definition (dod < DATE(intime) + 1) would under-count.
            Thus we chose to proceed with the inclusive form and document it here so users may
            adjust as necessary.

2. mortality_inhospital (secondary label)
    Source: mimic_hosp.admissions.hospital_expire_flag

    Logic: This is the standard MIMIC-IV in-hospital mortality flag. Initially set to 1
            by PhysioNet if the patient has died during the admission (deathtime IS NOT NULL
            in admissions). It is included as a supportive target and a cross-check against 'mortality_24h'.


------------------------------------
        GENERAL SCHEMA NOTES
------------------------------------ 
+ mimic_core was renamed to mimic_hosp in v2.0+. Use mimic_hosp.admissions,
mimc_hosp.patients, mimic_hosp.transfers.
+ mimic_icu.icustays retains its name across future versions.
+ patients.anchor_age is age at anchor_year, not at each admission.
    Actual age at ICU admission is computed as:
        anchor_age + (EXTRACT(YEAR FROM intime) - anchor_year)
    This is an appropriate approximation as it ignores sub-year birthday shifts,
    but is consistent with PhysioNet guidance/documentation and avoids re-identification risks.

------------------------------------------------------------------------
*/

-- ________________________________________________________________________________________
-- CTE 1: base_icu - pulls every ICU stay with its core timing columns
-- ________________________________________________________________________________________

WITH base_icu AS (
    SELECT
        ie.subject_id,
        ie.hadm_id,
        ie.stay_id,
        ie.intime           AS icu_intime,
        ie.outtime          AS icu_outtime,
        ie.first_careunit,
        ie.last_careunit,
        -- ICU LOS in fractional hours (DuckDB and PG both can support EXTRACT on interval)
        EXTRACT(EPOCH FROM (ie.outtime - ie.intime)) / 3600.00 AS icu_los_hours,
        -- row number within each hospital admission, ordered by ICU admission time
        ROW_NUMBER() OVER (
            PARTITION BY ie.hadm_id
            ORDER BY ie.intime ASC
        )                               AS icu_seq_num 
        FROM mimic_icu.icustays as ie
),

-- ________________________________________________________________________________________
-- CTE 2: first_icu - keeps only the FIRST ICU syat per hospital admission
-- ________________________________________________________________________________________

first_icu AS (
    SELECT * 
    FROM base_icu
    WHERE icu_seq_num = 1       -- this exlcudes re-admissions within the same hospitalization occassion
),



-- ________________________________________________________________________________________
-- CTE 3: patient_info - patient demographic info w/ computed ICU-admission age
-- ________________________________________________________________________________________


patient_info AS(
    SELECT
    p.subject_id,
    p.gender,
    p.anchor_age,
    p.anchor_year,
    p.dod,                      -- date of death (day precision)
    -- Approx. age at first ICU admission (info in header)
    p.anchor_age
        + (EXTRACT(YEAR FROM fi.icu_intime)::INT - p.anchor_year)
                                                    AS age_at_icu_admission,
    fi.stay_id                                                -- carrying stay_id for future table joins
    FROM mimic_hosp.patients AS p
    INNER JOIN first_icu as fi
        ON p.subject_id = fi.subject_id
),


-- ________________________________________________________________________________________
-- CTE 4: admission_info - hospital admission metadata and LOS
-- ________________________________________________________________________________________

admission_info AS (
    SELECT 
    a.hadm_id,
    a.admittime,
    a.dischtime,
    a.admissions_type,
    a.insurance,
    a.language,
    a.marital_status,
    a.race,
    a.hospital_expire_flag,                     -- PhysioNet in-hospital mortality flag (0/1)
    a.deathtime                                 -- timestamp of in-hospital death
    -- Hospital LOS in fractional hours
    EXTRACT(EPOCH FROM (a.dischtime - a.admittime)) / 3600.0
                                            AS hospital_los_hours
    FROM mimic_hosp.admissions AS a
),


-- ________________________________________________________________________________________
-- CTE 5: joined - merge ICU stats, patient demographic info, and admissions info
-- ________________________________________________________________________________________


joined AS (
    SELECT 
        fi.subject_id,
        fi.hadm_id,
        fi.stay_id,
        fi.icu_intime,
        fi.icu_outtime,
        fi.icu_los_hours,
        fi.first_careunit,
        pi.gender,
        pi.anchor_age,
        pi.age_at_icu_admission,
        pi.dod,
        ai.admissions_type,
        ai.insurance,
        ai.marital_status,
        ai.race,
        ai.hospital_los_hours,
        ai.hospital_expire_flag,
        ai.deathtime
    FROM first_icu      AS fi
    INNER JOIN patient_info AS pi ON fi.stay_id = pi.stay_id
    INNER JOIN admission_info AS ai on fi.hadm_id_id = ai.hadm_id
),


-- ________________________________________________________________________________________
-- CTE 6: filtered - apply all four cohort inclusion / exclusion criteria
-- ________________________________________________________________________________________

filtered AS (
    SELECT 
        *
        FROM JOINED 
        WHERE 
            -- Criteria A: Adult patients ( age ≥ 18 at ICU admission)
            age_at_icu_admission >= 18

            -- Criteria B: First ICU stay only - already enforced by first_icu CTE 2
            -- (icu_seq_num = 1), no additional filter needed

            -- Criteria C: ICU LOS ≥ 24 hours (need full observztion window available)
            AND icu_los_hours >= 24.0

            -- Criteria D: Exclude patients who died wihin the first hour of ICU stay.
            --                        This is where dod precision is needed, only when
            --                        when deathtime IS NOT NULL, it gives hour-level 
            --                        precision.
            AND NOT (
                -- Sub-case D1: deathtime is recorded (hour-level precision available to us)
                deathtime IS NOT NULL
                AND EXTRACT(EPOCH FROM (deathtime - icu_intime)) / 3600.0 < 1.0 
            )

            AND NOT (
                -- Sub-case D2: deathtime is NULL but dod equals the admission day,
                --             meaning death was recorded on 'day 0' which is likely 
                --             a dataset artefact. We use dod only when deathtime is NULL
                --             when deathtime IS NOT NULL, it gives hour-level precision.
                deathtime IS NULL 
                AND dod = DATE(icu_intime)
                AND icu_los_hours < 1.0
            )
),


-- ________________________________________________________________________________________
-- CTE 7: labelled - compute two-way binary labels 
-- ________________________________________________________________________________________

labelled AS(
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        gender, 
        anchor_age,
        admission_type,
        first_careunit,
        icu_los_hours,
        hospital_los_hours,
        insurance,
        race,
        marital_status

            -- mortality_24h ______________________________________________
            -- 1 if the patient's date of death is on or before the calendar
            -- date corresponding to icu_intime + 24 hours
            -- Logic: dod has day-only precision which casts the 24h threshold to
            --        a date avoids erronius mismatches from time-of-day.
            --        Patients with no dod (survived or died elsewhere) recieve
            --        a 0. 
        CASE 
            WHEN dod IS NOT NULL
                AND dod <= DATE(icu_intime + INTERVAL '24 hours')
            THEN 1
            ELSE 0
        END                 AS mortality_24h

        -- mortality_inhospital
        -- Directly from mimic_hosp.admissions.hospital_expire_flag (0/1).
        -- PhysioNet sets this to 1 when deathtime is not NULL for the admission.
        CAST(hospital_expire_flag AS SMALLINT) AS mortality_inhospital
    FROM filtered

)

-- __________________________________________________________________________________________________________________________________________
-- FINAL OUTPUT: cleaned, filtered, labelled cohort table ( run as CREATE TABLE or SELECT INTO as needed)
-- __________________________________________________________________________________________________________________________________________

SELECT
    subject_id,
    hadm_id,
    stay_id,
    gender,
    anchor_age,
    admission_type,
    first_careunit,
    ROUND(icu_los_hours::NUMERIC, 2) AS icu_los_hours,
    ROUND(hospital_los_hours::NUMERIC, 2) AS hospital_los_hours,
    mortality_24h,
    mortality_inhospital,
    insurance,
    race,
    marital_status
FROM labelled
ORDER BY subject_id, stay_id
;

-- ______________________________________________
-- SANITY CHECKS
-- ______________________________________________

*/ 
-- 24-hour mortality class balance
SELECT 
    mortality_24h AS label_value,
    COUNT (*) AS n_patients,
    ROUND(100.0 * COUNT (*) / SUM(COUNT(*)) OVER (), 2) as pct
FROM cohort 
GROUP BY mortality_24h
ORDER BY mortality_24h
;

-- In-hospital mortality class imbalance

SELECT
    mortality_inhospital AS label_value,
    COUNT(*) AS n_patients,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct
FROM cohort 
GROUP BY mortality_inhospital
ORDER BY mortality_inhospital
;

-- ______________________________________________
-- contingency table - 24-h vs in-hospital 
-- ______________________________________________

SELECT 
    mortality_24h,
    mortality_inhospital,
    COUNT(*) AS n
FROM cohort 
GROUP BY mortality_24h, mortality_inhospital
ORDER BY mortality_24h, mortality_inhospital

/*

______________________________________________
INDEX RECS. 
Apply these three sanity checks after the
cohort tabler is populated.
Logic: any downstream feature_extraction queries
       always joins on (stay_id) or (subject_id/hadm_id),
       and a filter/sort on the label columns.
______________________________________________

-- Primary join key for ICU time-series tables (chartevents, labevents, etc.)
CREATE INDEX IF NOT EXISTS idx_cohort_stay_id
    ON cohort(stay_id);

-- Secondary join key when linking back to hosp-level tables
CREATE INDEX IF NOT EXISTS idx_cohort_hadm_id
    ON cohort (hadm_id);

-- Subject-level lookups (notes etc)
CREATE INDEX IF NOT EXISTS idx_cohort_subject_id
    ON cohort (subject_id);

-- Label columns - used in GROUP BY and WHERE for model evaluation splits
CREATE INDEX IF NOT EXISTS idx_cohort_mortality_24h
    ON cohort (mortality_24h);

--CREATE INDEX IF NOT EXISTS idx_cohort_mortality_inhospital
    ON cohort (mortality_inhospital);

-- COMPOSITE: subject + admission - covers most MIMIC join patterns in one index
CREATE INDEX IF NOT EXISTS idx_cohort_subject_hadm
    ON cohort (subject_id, hadm_id);

    