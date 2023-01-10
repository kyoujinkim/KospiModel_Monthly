import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from joblib import dump, load

#from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
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
m_rtn = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kmrtn', index_col=0, parse_dates=True)
d_rtn = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kdrtn', index_col=0, parse_dates=True)

data = data.drop(labels=['Kospi Rtn', '3m Precision', 'Precision'], axis=1)
data['model'] = data.apply(model_selection, axis=1)
data.Forecast = data.Forecast.apply(prediction_categ)
data.model = data.model.astype('int')
data.Forecast = data.Forecast.astype('int')

rtn_df = pd.DataFrame(columns=['y_Rtn', 'MonthlyRtn'])

for startdate in data.index:
    enddate = startdate + relativedelta(months=1)
    pre_startdate = startdate - relativedelta(months=1)

    if data.index[-1] < enddate:
        break

    windowed = d_rtn[(d_rtn.index>startdate) & (d_rtn.index<=enddate)]
    pre_windowed = d_rtn[(d_rtn.index>pre_startdate) & (d_rtn.index<=startdate)]

    y_Rtn = ((windowed/100 + 1).prod()-1)*100# / (windowed.std() * np.sqrt(len(windowed)))
    MonthlyRtn = ((pre_windowed / 100 + 1).prod() - 1) * 100# / (pre_windowed.std() * np.sqrt(len(pre_windowed)))

    rtn_df = rtn_df.append(pd.DataFrame({'y_Rtn':y_Rtn.values,
                                               'MonthlyRtn':MonthlyRtn.values}
                                              , index=[startdate]))

data['MonthlyRtn'] = rtn_df.MonthlyRtn
rtn_df['label'] = rtn_df.y_Rtn.apply(labeling)
print(rtn_df.groupby(by='label').count())

data = pd.concat([data, rtn_df.y_Rtn], axis=1).dropna()
end_val = '2015-03-31'
train, test = data_split(data, end_val)

forecaster = ForecasterAutoregMultiVariate(
                 regressor          = LGBMRegressor(random_state=123),
                 level              = 'y_Rtn',
                 lags               = 52,
                 steps              = 1,
                 transformer_exog   = None,
                 weight_func        = None,
             )

lags_grid = [52]                                #[12, 26, 52]           #Best:52

param_grid = {
            'boosting_type': ['gbdt'],          #['gbdt', 'dart']       #Best:gbdt
            'colsample_bytree': [0.8],          #[0.7, 0.8, 0.9]        #Best:0.8
            'learning_rate': [0.1, 0.01],       #[0.1, 0.01, 0.001]     #Best 0.1
            'max_depth': [6, 7, 8, 9, 10],      #[2, 3, 4, 5, 6]        #Best 6
            'min_child_samples': [3, 5, 10, 20],                        #Best 5
            'objective': ['regression'],        # or binary
            }

results_grid = grid_search_forecaster_multiseries(
                forecaster          = forecaster,
                series              = data,
                param_grid          = param_grid,
                lags_grid           = lags_grid,
                steps               = 1,
                refit               = True,
                metric              = 'mean_squared_error',
                initial_train_size  = len(train),
                fixed_train_size    = False,
                return_best         = True,
                verbose             = False
                )

'''forecaster = ForecasterAutoregMultiVariate(
                 regressor          = LGBMRegressor(random_state=123,
                                                    boosting_type='gbdt',
                                                    colsample_bytree=0.8,
                                                    learning_rate=0.1,
                                                    max_depth=6,
                                                    min_child_samples=5,
                                                    objective='regression'),
                 level              = 'rtn',
                 lags               = 52,
                 steps              = 1,
                 transformer_exog   = None,
                 weight_func        = None,
             )
'''
dump(forecaster, filename='./model/forecaster.py')