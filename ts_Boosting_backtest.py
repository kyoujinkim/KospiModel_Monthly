import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
from joblib import dump, load
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

from skforecast_kjCustom.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast_kjCustom.model_selection_multiseries import backtesting_forecaster_multiseries
from skforecast_kjCustom.plot import plot_residuals

import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def labeling(x):
    if x > 0:
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
    return (x.Act_Rtn - x.y_Rtn_lower_bound) / (x.y_Rtn_upper_bound - x.y_Rtn_lower_bound)

def process_rtn_std(data, d_rtn, drift_window):
    rtn_df = pd.DataFrame()
    for startdate in data.index:
        enddate = startdate + relativedelta(months=1)
        pre_startdate = startdate - relativedelta(months=1)
        preweek_startdate = startdate - relativedelta(weeks=1)

        windowed = d_rtn[(d_rtn.index > startdate) & (d_rtn.index <= enddate)]
        pre_windowed = d_rtn[(d_rtn.index > pre_startdate) & (d_rtn.index <= startdate)]
        preweek_windowed = d_rtn[(d_rtn.index > preweek_startdate) & (d_rtn.index <= startdate)]

        avgRtn = (np.power((d_rtn[d_rtn.index <= startdate].iloc[-drift_window:] / 100 + 1).prod(), 20 / drift_window) - 1) * 100
        y_Rtn = ((windowed / 100 + 1).prod() - 1) * 100 - avgRtn
        MonthlyRtn = ((pre_windowed / 100 + 1).prod() - 1) * 100
        WeeklyRtn = ((preweek_windowed / 100 + 1).prod() - 1) * 100

        MonthlyStd = pre_windowed.std() * np.sqrt(252)
        WeeklyStd = preweek_windowed.std() * np.sqrt(252)

        value_windowed = (windowed/ 100 + 1).cumprod()
        RangeIdx = (value_windowed.max() - value_windowed.min()) / value_windowed.iloc[0] * 100

        if d_rtn.index[-1] <= enddate:
            rtn_df = rtn_df.append(pd.DataFrame({'y_Rtn': np.nan,
                                                 'avgRtn': avgRtn.values,
                                                 'MonthlyRtn': MonthlyRtn.values,
                                                 'WeeklyRtn': WeeklyRtn.values,
                                                 'MonthlyStd': MonthlyStd.values,
                                                 'WeeklyStd': WeeklyStd.values,
                                                 'RangeIdx': RangeIdx.values}
                                                , index=[startdate]))
        else:
            rtn_df = rtn_df.append(pd.DataFrame({'y_Rtn': y_Rtn.values,
                                                 'avgRtn': avgRtn.values,
                                                 'MonthlyRtn': MonthlyRtn.values,
                                                 'WeeklyRtn': WeeklyRtn.values,
                                                 'MonthlyStd': MonthlyStd.values,
                                                 'WeeklyStd': WeeklyStd.values,
                                                 'RangeIdx': RangeIdx.values}
                                                , index=[startdate]))

    return rtn_df

def print_backtest_info(interval, interval_std_window, backtest_predictions_total, backtest_predictions_idx):
    # plot statistics of interval accuracy
    print(f'Prediction Accuracy : '
          f'interval: {interval}, interval_std_window: {interval_std_window}')
    print(f'Prediction Accuracy : '
          f'{round((backtest_predictions_total.y_Rtn.apply(lambda x: 1 if x > 0 else 0)==backtest_predictions_total.Act_Rtn.apply(lambda x: 1 if x > 0 else 0)).mean() * 100, 1)}%')
    print(f'Prob for Actual Value within Interval Area: '
          f'{(round(backtest_predictions_idx.dropna().apply(interval_test, axis=1).mean() * 100, 1))}%')
    print(f'Position for Actual Value within Interval Area: '
          f'{(round(backtest_predictions_total.dropna().apply(interval_area, axis=1).mean() * 100, 1))}%')
    print(f'Description of Interval Area:\n',
          str((backtest_predictions_idx.y_Rtn_upper_bound - backtest_predictions_idx.y_Rtn_lower_bound).dropna().describe()))

def plot_backtest_result(backtest_predictions_idx):
    fig, ax = plt.subplots()
    backtest_predictions_idx.y_Rtn.plot(ax=ax, label='KOSPI Prediction')
    backtest_predictions_idx.Act_Rtn.plot(ax=ax, label='KOSPI Actual')
    backtest_predictions_idx.Curr_Idx.plot(ax=ax, label='KOSPI Current')
    ax.fill_between(
        backtest_predictions_idx.index,
        backtest_predictions_idx.y_Rtn_upper_bound,
        backtest_predictions_idx.y_Rtn_lower_bound,
        color='deepskyblue',
        alpha=0.3,
        label='70% interval'
    )
    ax.legend()
    plt.show()

def tanh(x):
    e_x = np.exp(x)
    e_mx = np.exp(-x)

    return (e_x - e_mx) / (e_x + e_mx)
def custom_loss(data, preds):
    y_true = data
    grad = tanh(y_true - preds)
    hess = preds * (1.0 - preds)
    return grad, hess

def custom_loss_eval(preds, data):
    y_true = data
    loss = tanh(y_true - preds)
    return "custom_loss", np.mean(loss), False

data = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='History', index_col=0, parse_dates=True)
data_LEI = pd.read_csv('./data/LEI_TOTL.csv', index_col=0, parse_dates=True)
d_rtn = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kdrtn', index_col=0, parse_dates=True)
d_idx = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kidx', index_col=0, parse_dates=True)

data[['3m Precision', 'Precision']] = data[['3m Precision', 'Precision']].shift(1)
data.Forecast = data.Forecast.apply(prediction_categ)
data = data.drop(
    labels=list(data.filter(regex='Neutral'))
           +list(data.filter(regex='Bear'))
           +['Kospi Rtn'],#, 'Precision', 'Forecast'], #'3m Precision',  'Precision'
    axis=1
    )

data_LEI.index = data_LEI.index + MonthEnd(1)
data = pd.merge_asof(data, data_LEI, left_index=True, right_index=True, direction='backward')

rtn_df = process_rtn_std(data, d_rtn, 252)

data['avgRtn'] = rtn_df.avgRtn
data['MonthlyRtn'] = rtn_df.MonthlyRtn
data['WeeklyRtn'] = rtn_df.WeeklyRtn
data['MonthlyStd'] = rtn_df.MonthlyStd
data['WeeklyStd'] = rtn_df.WeeklyStd
data['y_Rtn'] = rtn_df.y_Rtn

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
                                                    max_depth=8, #2
                                                    min_child_samples=2, #4
                                                    #objective='regression'),
                                                    objective=custom_loss),
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
                                           metric             = 'mean_absolute_error', #mean_absolute_error
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

forecaster.fit(series=data.dropna())
pd.set_option('display.max_rows', len(forecaster.get_feature_importance(step=1)))
print(forecaster.get_feature_importance(step=1).sort_values(by='importance'))

#plot Time Series of prediction and actual
temp_data = data.y_Rtn.copy()
temp_data.name = 'Act_Rtn'
backtest_predictions_total = pd.merge(backtest_predictions, d_idx, left_index=True, right_index=True, how='left').ffill()
backtest_predictions_total = pd.merge(backtest_predictions_total, temp_data, left_index=True, right_index=True, how='left')
backtest_predictions_rtn = backtest_predictions_total.loc[:,['y_Rtn','y_Rtn_lower_bound','y_Rtn_upper_bound','Act_Rtn']]
backtest_predictions_idx = (backtest_predictions_rtn / 100 + 1).multiply(backtest_predictions_total.IDX, axis=0)
backtest_predictions_idx['Curr_Idx'] = backtest_predictions_total.IDX

backtest_predictions_rtn = pd.concat([backtest_predictions_rtn, rtn_df.RangeIdx, rtn_df.avgRtn], axis=1)
backtest_predictions_rtn.to_csv('./interval_result.csv')

# plot statistics of interval accuracy
print_backtest_info(interval, interval_std_window, backtest_predictions_total, backtest_predictions_idx)

#plot Graph of Actual, Prediction and Interval
plot_backtest_result(backtest_predictions_idx)

#plot statistics of prediction residual
residuals = backtest_predictions['y_Rtn'] - data.y_Rtn
plot_residuals(residuals=residuals)
plt.show()