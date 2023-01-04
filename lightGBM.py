import pandas as pd
import lightgbm as lgb

X = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='History', index_col=0, parse_dates=True)
y = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='KRTN', index_col=0, parse_dates=True)

a = 1