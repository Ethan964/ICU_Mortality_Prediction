'''
connect.py


====================================

Reads and registers all MIMIC-IV v3.1 .csv.gz files as 
duckdb views under mimic_hosp.* and mimic_icu.* schemas.

Pipeline
----------
    formr icu_tft.data.connect import get_connection, list_tables, probe_tables
    
    con = get_connection()
    
    print(list_tables(con))
    
    print(list_tables(con))
    proble_tables(con)

Env Requirements
--------------------
Needs MIMIC_DATA_DIR to be set into poetry .env file 

'''

from __future__ import annotations
import os
from pathlib import Path
import duckdb
from dotenv import load_dotenv

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]
load_dotenv(_PROJECT_ROOT / '.env')

_MIMIC_TABLES: list[tuple[str, str, str]] = [
    ('mimic_hosp', 'admissions',          'hosp/admissions.csv.gz'),
    ('mimic_hosp', 'patients',            'hosp/patients.csv.gz'),
    ('mimic_hosp', 'transfers',           'hosp/transfers.csv.gz'),
    ('mimic_hosp', 'diagnoses_icd',       'hosp/diagnoses_icd.csv.gz'),
    ('mimic_hosp', 'd_icd_diagnoses',     'hosp/d_icd_diagnoses.csv.gz'),
    ('mimic_hosp', 'procedures_icd',      'hosp/procedures_icd.csv.gz'),
    ('mimic_hosp', 'd_icd_procedures',    'hosp/d_icd_procedures.csv.gz'),
    ('mimic_hosp', 'hcpcsevents',         'hosp/hcpcsevents.csv.gz'),
    ('mimic_hosp', 'd_hcpcs',             'hosp/d_hcpcs.csv.gz'),
    ('mimic_hosp', 'labevents',           'hosp/labevents.csv.gz'),
    ('mimic_hosp', 'd_labitems',          'hosp/d_labitems.csv.gz'),
    ('mimic_hosp', 'microbiologyevents',  'hosp/microbiologyevents.csv.gz'),
    ('mimic_hosp', 'antimicrobial',       'hosp/antimicrobial.csv.gz'),
    ('mimic_hosp', 'pharmacy',            'hosp/pharmacy.csv.gz'),
    ('mimic_hosp', 'prescriptions',       'hosp/prescriptions.csv.gz'),
    ('mimic_hosp', 'poe',                 'hosp/poe.csv.gz'),
    ('mimic_hosp', 'poe_detail',          'hosp/poe_detail.csv.gz'),
    ('mimic_hosp', 'emar',                'hosp/emar.csv.gz'),
    ('mimic_hosp', 'emar_detail',         'hosp/emar_detail.csv.gz'),
    ('mimic_hosp', 'drgcodes',            'hosp/drgcodes.csv.gz'),
    ('mimic_hosp', 'services',            'hosp/services.csv.gz'),
    ('mimic_hosp', 'omr',                 'hosp/omr.csv.gz'),
    # ── mimic_icu (9 tables) ────────────────────────────────────────────────
    ('mimic_icu',  'icustays',            'icu/icustays.csv.gz'),
    ('mimic_icu',  'chartevents',         'icu/chartevents.csv.gz'),
    ('mimic_icu',  'datetimeevents',      'icu/datetimeevents.csv.gz'),
    ('mimic_icu',  'd_items',             'icu/d_items.csv.gz'),
    ('mimic_icu',  'inputevents',         'icu/inputevents.csv.gz'),
    ('mimic_icu',  'outputevents',        'icu/outputevents.csv.gz'),
    ('mimic_icu',  'procedureevents',     'icu/procedureevents.csv.gz'),
    ('mimic_icu',  'ingredientevents',    'icu/ingredientevents.csv.gz'),
    ('mimic_icu',  'caregiver',           'icu/caregiver.csv.gz'),
]

def get_connection(
    db_path: str | Path | None = None,
    mimic_data_dir: str | Path | None = None,
    *,
    read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    if db_path is None:
        db_path = _PROJECT_ROOT / 'data' / 'mimic.duckdb'
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    if mimic_data_dir is None:
        mimic_data_dir = os.environ.get('MIMIC_DATA_DIR')
    if mimic_data_dir is None:
        raise EnvironmentError(
            'MIMIC_DATA_DIR is not set. Add it to .env file'
        )
    mimic_data_dir = Path(mimic_data_dir)
    if not mimic_data_dir.exists():
        raise FileNotFoundError(
            f'MIMIC_DATA_DIR does not exist: {mimic_data_dir}'
        )
    
    con = duckdb.connect(str(db_path), read_only=read_only)
    con.execute('CREATE SCHEMA IF NOT EXISTS mimic_hosp')
    con.execute('CREATE SCHEMA IF NOT EXISTS mimic_icu')
    
    missing: list[str] = []
    registered: list[str] = []
    
    for schema, table, rel_path in _MIMIC_TABLES:
        full_path = mimic_data_dir / rel_path
        if not full_path.exists():
            missing.append(str(full_path))
            continue
        
        con.execute(f'''
                    CREATE OR REPLACE VIEW {schema}.{table} AS 
                    SELECT * FROM read_csv_auto('{full_path}', compression='gzip')
                    ''')  
        registered.append(f'{schema}.{table}')
        
        
    print(f'[connect.py] Registered {len(registered)} views against {db_path.name}')
    if missing:
        print(
            f"[connect.py] WARNING — {len(missing)} file(s) not found "
            f"(views skipped):\n  " + "\n  ".join(missing)
        )
    return con

def list_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    ''''
    Returns a sorted llist of all registered view names
    '''
    
    rows = con.execute('''
                       SELECT table_schema || '.' || table_name AS full_name 
                       FROM information_schema.tables
                       WHERE table_schema IN ('mimic_hosp', 'mimic_icu')
                       ORDER BY table_schema, table_name
                       ''').fetchall()
    return[r[0] for r in rows]

def probe_tables(
    con: duckdb.DuckDBPyConnection,
    n: int = 2,
) -> None:
    '''
    Fast and efficient view on all tables for quick sanity checks and data integrity checks
    '''
    tables = list_tables(con)
    if not tables:
        print('[proble_tables] No tables found')
        return
    print(f"\n{'Table':<40} {'Columns':>7}  Sample columns")
    print("-" * 75)
    for full_name in tables:
        try:
            info = con.execute(
                f"SELECT * FROM {full_name} LIMIT {n}"
            ).description                                  
            col_names = [d[0] for d in info]
            print(f"{full_name:<40} {len(col_names):>7}  {', '.join(col_names[:4])} …")
        except Exception as exc:
            print(f"{full_name:<40}  ERROR: {exc}")
            
if __name__ == '__main__':
    connection = get_connection()
    tables = list_tables(connection)
    print(f'\nRegistered tables ({len(tables)}): ')
    for t in tables:
        print(f'{t}')
    print()
    probe_tables(connection)