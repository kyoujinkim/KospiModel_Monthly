import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from joblib import dump, load

#from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pandas._libs.tslibs.offsets import MonthEnd
#from catboost import CatBoostRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from skforecast_kjCustom.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast_kjCustom.model_selection_multiseries import backtesting_forecaster_multiseries
from skforecast_kjCustom.model_selection_multiseries import grid_search_forecaster_multiseries
from skforecast_kjCustom.model_selection_multiseries import random_search_forecaster_multiseries
from skforecast_kjCustom.plot import plot_residuals

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import plotly.express as px

import warnings
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

data = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='History', index_col=0, parse_dates=True)
data_LEI = pd.read_csv('./data/LEI_TOTL.csv', index_col=0, parse_dates=True)
m_rtn = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kmrtn', index_col=0, parse_dates=True)
d_rtn = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kdrtn', index_col=0, parse_dates=True)
d_idx = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kidx', index_col=0, parse_dates=True)

data[['3m Precision', 'Precision']] = data[['3m Precision', 'Precision']].shift(1)
data = data.drop(
    labels=['Kospi Rtn'] + list(data.filter(regex='Neutral')), #'3m Precision',  'Precision'
    axis=1
    )
data['model'] = data.apply(model_selection, axis=1)
data.Forecast = data.Forecast.apply(prediction_categ)
data.model = data.model.astype('int')
data.Forecast = data.Forecast.astype('int')

data_LEI.index = data_LEI.index + MonthEnd(1)
data = pd.merge_asof(data, data_LEI, left_index=True, right_index=True, direction='backward')

rtn_df = pd.DataFrame(columns=['y_Rtn', 'MonthlyRtn'])

for startdate in data.index:
    enddate = startdate + relativedelta(months=1)
    pre_startdate = startdate - relativedelta(months=1)
    preweek_startdate = startdate - relativedelta(weeks=1)

    windowed = d_rtn[(d_rtn.index>startdate) & (d_rtn.index<=enddate)]
    pre_windowed = d_rtn[(d_rtn.index>pre_startdate) & (d_rtn.index<=startdate)]
    preweek_windowed = d_rtn[(d_rtn.index>preweek_startdate) & (d_rtn.index<=startdate)]

    y_Rtn = ((windowed/100 + 1).prod()-1)*100# / (windowed.std() * np.sqrt(len(windowed)))
    MonthlyRtn = ((pre_windowed / 100 + 1).prod() - 1) * 100# / (pre_windowed.std() * np.sqrt(len(pre_windowed)))
    WeeklyRtn = ((preweek_windowed / 100 + 1).prod() - 1) * 100
    #SHARPE = (windowed.std() * np.sqrt(len(windowed)))
    #pre_SHARPE = (pre_windowed.std() * np.sqrt(len(pre_windowed)))

    if data.index[-1] < enddate:
        rtn_df = rtn_df.append(pd.DataFrame({'y_Rtn': np.nan,
                                             'MonthlyRtn': MonthlyRtn.values,
                                             'WeeklyRtn': WeeklyRtn.values}
                                            , index=[startdate]))
    else:
        rtn_df = rtn_df.append(pd.DataFrame({'y_Rtn':y_Rtn.values,
                                             'MonthlyRtn':MonthlyRtn.values,
                                             'WeeklyRtn':WeeklyRtn.values}
                                            , index=[startdate]))

data['MonthlyRtn'] = rtn_df.MonthlyRtn
data['WeeklyRtn'] = rtn_df.WeeklyRtn
rtn_df['label'] = rtn_df.y_Rtn.apply(labeling)
print(rtn_df.groupby(by='label').count())

data = pd.concat([data, rtn_df.y_Rtn], axis=1)
#add frequency to data index
data.index.freq = pd.infer_freq(data.index)

data = data.bfill().dropna()

end_val = '2015-04-30'
end_val_idx = data.index.get_loc(end_val, method='ffill')

forecaster = ForecasterAutoregMultiVariate(
                 regressor          = Ridge(),
                 level              = 'y_Rtn',
                 lags               = 52,
                 steps              = 1,
                 transformer_exog   = None,
                 weight_func        = None,
             )

lags_grid = [26]                                #[12, 26, 52]                       #Best:26

param_distributions = {'alpha': np.arange(start=0.1, stop=1.0, step=0.01, dtype=float),
                       'fit_intercept': [False, True]}

'''param_grid = {
            'alpha': [0.1, ],          #['gbdt', 'dart']                   #Best:dart
            'colsample_bytree': [0.9],          #[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]     #Best:0.9
            'learning_rate': [0.1],             #[0.1, 0.01, 0.001]                 #Best 0.1
            'max_depth': [2],                   #[2, 3, 4, 5, 6, 7, 8, 9, 10]       #Best 2
            'min_child_samples': [4],           #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]    #Best 4
            'objective': ['regression'],        # or binary
            }'''

results_grid = random_search_forecaster_multiseries(
                forecaster          = forecaster,
                series              = data,
                param_distributions = param_distributions,
                lags_grid           = lags_grid,
                steps               = 1,
                refit               = True,
                metric              = 'mean_absolute_error',
                initial_train_size  = end_val_idx,
                fixed_train_size    = False,
                return_best         = True,
                verbose             = False,
                random_state        = 123
                )
'''
forecaster = ForecasterAutoregMultiVariate(
                 regressor          = LGBMRegressor(random_state=123,
                                                    boosting_type='gbdt', #gbdt
                                                    colsample_bytree=0.8, #0.8
                                                    learning_rate=0.1, #0.1
                                                    max_depth=6, #6
                                                    min_child_samples=5, #5
                                                    objective='regression'),
                 level              = 'y_Rtn',
                 lags               = 52, #52
                 steps              = 1,
                 transformer_exog   = None,
                 weight_func        = None,
             )
'''
dump(forecaster, filename='./model/forecaster.py')