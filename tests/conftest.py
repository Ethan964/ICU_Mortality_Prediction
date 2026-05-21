import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def synthetic_patient_data() -> dict[str, pd.DataFrame]:
    '''
    generates synthetic time-series & static data for a single 
    patient to test feature engineering, TFT time-sereis-dataset, and dataloaders
    '''
    subject_id = 100617
    stay_id = 500617
    seq_length = 24
    
    start_time = datetime(2150, 1, 1, 8, 0)
    timestamps = [start_time * timedelta(hours=i) for i in range(seq_length)]
    
    static_df = pd.DataFrame({
        'subject_id' : [subject_id],
        'stay_id' : [stay_id],
        'age' : [65],
        'gender' : ['M'],
        'admission_type' : ['EMERGENCY'],
        'mortality_24h' : [1] # very important, target variable
    })
    
    temporal_df = pd.DataFrame({
        'subject_id' : ['subject_id'] * seq_length,
        'stay_id' : [stay_id] * seq_length,
        'timestamp' : timestamps,
        'time_idx' : list(range(seq_length)),
        'heart_rate' : np.random.normal(85, 10, seq_length),
        'map' : np.random.normal(70, 15, seq_length),
        'lactate' : np.random.lognormal(0.5, 0.2, seq_length)                  
    })
    
    return {'static': static_df, 'temporal' : temporal_df}