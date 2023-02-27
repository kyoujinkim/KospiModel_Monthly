import multiprocessing
import pickle
from glob import glob

import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
from joblib import dump, load

from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries

from skforecast.plot import plot_residuals

import matplotlib.pyplot as plt

import warnings

from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

def ensemble_learning(var_path_each):

    results = []
    level = 'y'
    lags = 26
    steps = 4

    with open(var_path_each, 'rb') as f:
        data = pickle.load(f)  # 단 한줄씩 읽어옴

    data = data.drop('KospiDeT', axis=1)
    # add y label to data, and then resample data in to weekly format
    X = pd.concat([data, m_rtn], axis=1) \
        .resample('W-Fri').last().dropna()

    forecaster = ForecasterAutoregMultiVariate(
        regressor=Ridge(alpha=.5),
        level=level,
        lags=lags,  # 26
        steps=steps
    )
    forecaster.fit(series=X)
    prediction = forecaster.predict(steps=steps)
    # print(forecaster.predict(steps=5))
    result = ((1 + prediction).product() - 1).values * 100
    results.extend(result)

    forecaster = ForecasterAutoregMultiVariate(
        regressor=LGBMRegressor(random_state=123,
                                boosting_type='dart', #dart
                                colsample_bytree=0.9, #0.8
                                learning_rate=0.1, #0.1
                                max_depth=8, #2
                                min_child_samples=2, #4,
                                objective='regression'),
        level=level,
        lags=lags,  # 26
        steps=steps
    )
    forecaster.fit(series=X)
    prediction = forecaster.predict(steps=steps)
    # print(forecaster.predict(steps=5))
    result = ((1 + prediction).product() - 1).values * 100
    results.extend(result)

    forecaster = ForecasterAutoregMultiVariate(
        regressor=RandomForestRegressor(random_state=123,
                                        max_depth=8,
                                        min_samples_leaf=2),
        level=level,
        lags=lags,  # 26
        steps=steps
    )
    forecaster.fit(series=X)
    prediction = forecaster.predict(steps=steps)
    result = ((1 + prediction).product() - 1).values * 100
    results.extend(result)

    return results


var_path = glob('C:/Users/NHWM/PycharmProjects/KospiModel/pickle_var/variables/*')

d_rtn = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kdrtn', index_col=0, parse_dates=True)
d_idx = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kidx', index_col=0, parse_dates=True)
m_rtn = d_idx.pct_change(periods=7).dropna().rename(columns={'IDX':'y'})
m_rtn.index = m_rtn.index - pd.Timedelta(1,'w')

#to loop through all date, get array of Unique date
date_arr = list(sorted(set([x.split('\\')[1].replace('.pkl','')[-10:] for x in var_path])))

if __name__ == "__main__":
    p = multiprocessing.Pool(6)
    total_results = pd.DataFrame()
    for date_each in tqdm(date_arr[5:]):
        var_path_by_date = [x for x in var_path if (date_each in x) and ('All' not in x)]

        results = [date_each]
        result = p.map(ensemble_learning, var_path_by_date)
        results.extend(np.array(result).flatten())

        total_results = pd.concat([total_results, pd.DataFrame(results).T])

    #forecaster = load(filename='./model/forecaster.py')
    total_results_df = total_results.set_index(keys=0)
    total_results_df.to_csv('./result.csv')