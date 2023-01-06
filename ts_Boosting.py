import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta

#from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
#from catboost import CatBoostRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from skforecast.ForecasterAutoregMultiVariate.ForecasterAutoregMultiVariate_KJCustom import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries
from skforecast.model_selection_multiseries import grid_search_forecaster_multiseries
from skforecast.model_selection_multiseries import random_search_forecaster_multiseries
from skforecast.plot import plot_residuals

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

SHARPE_df = pd.DataFrame(columns=['Sharpe', 'pre_Sharpe'])

for startdate in data.index:
    enddate = startdate + relativedelta(months=1)
    pre_startdate = startdate - relativedelta(months=1)

    if data.index[-1] < enddate:
        break

    windowed = d_rtn[(d_rtn.index>startdate) & (d_rtn.index<=enddate)]
    pre_windowed = d_rtn[(d_rtn.index>pre_startdate) & (d_rtn.index<=startdate)]

    SHARPE = ((windowed/100 + 1).prod()-1)*100# / (windowed.std() * np.sqrt(len(windowed)))
    pre_SHARPE = ((pre_windowed / 100 + 1).prod() - 1) * 100# / (pre_windowed.std() * np.sqrt(len(pre_windowed)))
    #SHARPE = (windowed.std() * np.sqrt(len(windowed)))
    #pre_SHARPE = (pre_windowed.std() * np.sqrt(len(pre_windowed)))

    SHARPE_df = SHARPE_df.append(pd.DataFrame({'Sharpe':SHARPE.values,
                                               'pre_Sharpe':pre_SHARPE.values}
                                              , index=[startdate]))

data['m_rtn'] = SHARPE_df.pre_Sharpe
SHARPE_df['label'] = SHARPE_df.Sharpe.apply(labeling)
print(SHARPE_df.groupby(by='label').count())

data = pd.concat([data,SHARPE_df.Sharpe], axis=1).dropna()
train, test = data_split(data, '2015-03-31')

forecaster = ForecasterAutoregMultiVariate(
                 regressor          = LGBMRegressor(random_state=123,
                                                    max_depth='3'),
                 level              = 'Sharpe',
                 lags               = 52,
                 steps              = 1,
                 transformer_exog   = None,
                 weight_func        = None,
             )
forecaster.fit(series=data)

metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                           forecaster         = forecaster,
                                           series             = data,
                                           steps              = 1,
                                           metric             = 'mean_absolute_error',
                                           initial_train_size = len(train),
                                           refit              = True,
                                           fixed_train_size   = False,
                                           verbose            = False
                                       )

print(metrics_levels)
pd.set_option('display.max_rows', len(forecaster.get_feature_importance(step=1)))
print(forecaster.get_feature_importance(step=1).sort_values(by='importance'))
'''fig, ax = plt.subplots()
backtest_predictions.plot(ax=ax)
test.Sharpe.plot(ax=ax)
ax.legend(['Prediction','Actual'])
plt.show()'''
residuals = backtest_predictions['Sharpe'] - test.Sharpe
plot_residuals(residuals=residuals)
plt.show()

a = 1