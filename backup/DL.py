import tensorflow as tf
import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np
import datetime as dt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def labeling(x):
    if x > 0.:
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

SHARPE_df = pd.DataFrame(columns=['Sharpe'])
for startdate in data.index:
    enddate = startdate + relativedelta(months=1)
    if data.index[-1] < enddate:
        break
    windowed = d_rtn[(d_rtn.index>startdate) & (d_rtn.index<=enddate)]
    SHARPE = ((windowed/100 + 1).prod()-1)*100 / (windowed.std() * np.sqrt(len(windowed)))
    SHARPE_df = SHARPE_df.append(pd.DataFrame({'Sharpe':SHARPE.values}, index=[startdate]))
SHARPE_df['label'] = SHARPE_df.Sharpe.apply(labeling)
print(SHARPE_df.groupby(by='label').count())

data = pd.concat([data,SHARPE_df.label], axis=1).dropna()
train, test = data_split(data, '2021-01-01')
