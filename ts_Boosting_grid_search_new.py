import pickle
from glob import glob

import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from joblib import dump, load

#from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pandas._libs.tslibs.offsets import MonthEnd
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries import random_search_forecaster_multiseries
#from catboost import CatBoostRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import plotly.express as px

import warnings

from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

def labeling(x):
    if x > 0.1:
        return 1
    else:
        return 0

def model_selection(x):
    data = x[:6]
    #return str(np.argwhere(data.values == data.max()).flatten().tolist())
    return np.argmax(data.values)

def prediction_categ(x):
    if x=='Bull':
        return 2
    elif x=='Neutral':
        return 1
    else:
        return 0

def data_split(data, date):
    dt_date = dt.datetime.strptime(date,'%Y-%m-%d')
    train_total = data[data.index <= dt_date]
    test = data[data.index > dt_date]

    train = train_total
    #valid = train_total.iloc[int(len(train_total) * 0.8) :, :]
    #train = train_total.iloc[: int(len(train_total) * 0.8), :]

    return train, test

var_path = glob('C:/Users/NHWM/PycharmProjects/KospiModel/pickle_var/variables/*')
var_path = [x for x in var_path if '2023-02-17' in x]
for var_path_each in tqdm(var_path):
    d_rtn = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kdrtn', index_col=0, parse_dates=True)
    d_idx = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kidx', index_col=0, parse_dates=True)
    m_rtn = d_idx.pct_change(periods=7).dropna().rename(columns={'IDX':'y'})
    m_rtn.index = m_rtn.index - pd.Timedelta(1,'w')

    with open(var_path_each, 'rb') as f:
        data = pickle.load(f)  # 단 한줄씩 읽어옴

    data = data.drop('KospiDeT', axis=1)
    # add y label to data, and then resample data in to weekly format
    X = pd.concat([data, m_rtn], axis=1) \
        .resample('W-Fri').last().dropna()

    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = RandomForestRegressor(random_state=123, criterion='absolute_error', n_jobs=-1),
                     level              = 'y',
                     lags               = 52,
                     steps              = 4,
                     transformer_exog   = None,
                     weight_func        = None,
                 )

    lags_grid = [13, 26, 52]                                #[12, 26, 52]                       #Best:26

    param_distributions = {'n_estimators': np.arange(start=100, stop=1000, step=100, dtype=int),
                           'max_depth': np.arange(start=2, stop=16, step=1, dtype=int),
                           'min_samples_leaf': np.arange(start=2, stop=32, step=1, dtype=int),
                           'max_features': [1.0, "sqrt", "log2"]
                           }

    results_grid = random_search_forecaster_multiseries(
                    forecaster          = forecaster,
                    series              = X,
                    param_distributions = param_distributions,
                    lags_grid           = lags_grid,
                    steps               = 4,
                    refit               = True,
                    metric              = 'mean_absolute_error',
                    initial_train_size  = 52 + 4,
                    fixed_train_size    = False,
                    return_best         = True,
                    verbose             = False,
                    random_state        = 123
                    )

    print(var_path_each)
    print(results_grid)