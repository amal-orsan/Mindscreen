import pandas as pd
import os

TARGET_COL   = 'Depression_Type'
LEAKAGE_COLS = ['Symptoms']

_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Mental Health Classification.csv')

df = pd.read_csv(_data_path)
X  = df.drop(columns=[TARGET_COL] + LEAKAGE_COLS)
y  = df[TARGET_COL]
