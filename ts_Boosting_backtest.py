import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
from joblib import dump, load

from lightgbm import LGBMRegressor

from skforecast_kjCustom.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast_kjCustom.model_selection_multiseries import backtesting_forecaster_multiseries
from skforecast_kjCustom.plot import plot_residuals

import matplotlib.pyplot as plt

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

    return train, test

def interval_test(x):
    if x.y_Rtn_lower_bound < x.Act_Rtn and x.y_Rtn_upper_bound > x.Act_Rtn:
        return 1
    else:
        return 0

def interval_area(x):
    return (x.Act_Rtn - x.y_Rtn_lower_bound) / (x.y_Rtn_upper_bound - x.y_Rtn_lower_bound) * 100

data = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='History', index_col=0, parse_dates=True)
data_LEI = pd.read_csv('./data/LEI_TOTL.csv', index_col=0, parse_dates=True)
d_rtn = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kdrtn', index_col=0, parse_dates=True)
d_idx = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kidx', index_col=0, parse_dates=True)

data[['3m Precision', 'Precision']] = data[['3m Precision', 'Precision']].shift(1)
data = data.drop(
    labels=list(data.filter(regex='Neutral'))+list(data.filter(regex='Kospi Rtn')), #'3m Precision',  'Precision'
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

    if d_rtn.index[-1] <= enddate:
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

data = data.bfill()

end_val = '2015-04-30'
end_val_idx = data.index.get_loc(end_val, method='ffill')

#forecaster = load(filename='./model/forecaster.py')
forecaster = ForecasterAutoregMultiVariate(
                 regressor          = LGBMRegressor(random_state=123,
                                                    boosting_type='dart', #dart
                                                    colsample_bytree=0.9, #0.8
                                                    learning_rate=0.1, #0.1
                                                    max_depth=2, #2
                                                    min_child_samples=4, #4
                                                    objective='regression'),
                 level              = 'y_Rtn',
                 lags               = 26, #26
                 steps              = 1
             )

interval = [15, 85]#[[15, 85], [10, 90], [5, 95]]
interval_std_window = 13#   26    #[13, 26, 52, 104, None]

metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
                                           forecaster         = forecaster,
                                           series             = data.dropna(),
                                           steps              = 1,
                                           metric             = 'mean_absolute_error',
                                           initial_train_size = end_val_idx,
                                           refit              = True,
                                           fixed_train_size   = False,
                                           verbose            = False,
                                           interval           = interval,
                                           interval_std_window = interval_std_window,
                                           tqdm_on             = True
                                       )

forecaster.fit(series=data.dropna())
for idx_step in range(len(data.dropna())-len(data),0):
    confidence_interval = backtest_predictions.iloc[-1].y_Rtn - backtest_predictions.iloc[-1].y_Rtn_lower_bound
    y_Rtn = forecaster.predict(steps=1, last_window=data.iloc[:idx_step])
    prediction_step = pd.DataFrame({'y_Rtn':y_Rtn.values[0][0],
                                    'y_Rtn_lower_bound':y_Rtn.values[0][0] - confidence_interval,
                                    'y_Rtn_upper_bound':y_Rtn.values[0][0] + confidence_interval},
                                   index=y_Rtn.index)
    backtest_predictions = pd.concat([backtest_predictions, prediction_step]).drop_duplicates().sort_index()

print(metrics_levels)

#forecaster.fit(series=data.dropna())
#pd.set_option('display.max_rows', len(forecaster.get_feature_importance(step=1)))
#print(forecaster.get_feature_importance(step=1).sort_values(by='importance'))

#plot Time Series of prediction and actual
temp_data = data.y_Rtn.copy()
temp_data.name = 'Act_Rtn'
backtest_predictions_total = pd.merge(backtest_predictions, d_idx, left_index=True, right_index=True, how='left').ffill()
backtest_predictions_total = pd.merge(backtest_predictions_total, temp_data, left_index=True, right_index=True, how='left')
backtest_predictions_rtn = backtest_predictions_total.loc[:,['y_Rtn','y_Rtn_lower_bound','y_Rtn_upper_bound','Act_Rtn']]
backtest_predictions_idx = (backtest_predictions_rtn / 100 + 1).multiply(backtest_predictions_total.IDX, axis=0)
#backtest_predictions_idx = (backtest_predictions_rtn.drop(labels='Act_Rtn', axis=1) / 100 + 1)\
#                            .multiply(backtest_predictions_total.IDX.rolling(4).mean(), axis=0)
#backtest_predictions_idx['Act_Rtn'] = (backtest_predictions_rtn.Act_Rtn / 100 + 1).multiply(backtest_predictions_total.IDX, axis=0)

backtest_predictions_rtn.to_csv('./interval_result.csv')

#plot statistics of interval accuracy
print(f'Prediction Accuracy : '
      f'interval: {interval}, interval_std_window: {interval_std_window}')
print(f'Prediction Accuracy : '
      f'{round((backtest_predictions_total.y_Rtn.apply(lambda x: 1 if x>0 else 0) == backtest_predictions_total.Act_Rtn.apply(lambda x: 1 if x>0 else 0)).mean()*100, 1)}%')
print(f'Prob for Actual Value within Interval Area: '
      f'{(round(backtest_predictions_idx.dropna().apply(interval_test, axis=1).mean()*100,1))}%')
print(f'Position for Actual Value within Interval Area: '
      f'{(round(backtest_predictions_total.dropna().apply(interval_area, axis=1).mean(),1))}%')
print(f'Description of Interval Area:\n',
    str((backtest_predictions_idx.y_Rtn_upper_bound - backtest_predictions_idx.y_Rtn_lower_bound).dropna().describe()))

#plot Graph of Actual, Prediction and Interval
fig, ax = plt.subplots()
backtest_predictions_idx.y_Rtn.plot(ax=ax, label='KOSPI Prediction')
backtest_predictions_idx.Act_Rtn.plot(ax=ax, label='KOSPI Actual')
ax.fill_between(
    backtest_predictions_idx.index,
    backtest_predictions_idx.y_Rtn_upper_bound,
    backtest_predictions_idx.y_Rtn_lower_bound,
    color = 'deepskyblue',
    alpha = 0.3,
    label = '70% interval'
)
ax.legend()
plt.show()

#plot statistics of prediction residual
residuals = backtest_predictions['y_Rtn'] - data.y_Rtn
plot_residuals(residuals=residuals)
plt.show()

a = 1