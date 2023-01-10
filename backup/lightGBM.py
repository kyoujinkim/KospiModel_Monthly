import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import lightgbm as lgb
from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta
from sklearn import tree, ensemble
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

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

    #train = lgb.Dataset(train.drop('label', axis=1), label=train.label, categorical_feature=['Forecast', 'model'], free_raw_data=False)
    #valid = lgb.Dataset(valid.drop('label', axis=1), label=valid.label, categorical_feature=['Forecast', 'model'], free_raw_data=False)
    #test = lgb.Dataset(test.drop('label', axis=1), label=test.label, categorical_feature=['Forecast', 'model'], free_raw_data=False)

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

'''params={}
params['learning_rate']=0.001
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='multiclass' #Multi-class target feature
params['metric']='multi_logloss' #metric for multi-class
params['max_depth']=5
params['num_class']=3
bst = lgb.train(params, train_set=train, valid_sets=[valid])'''
#plt.show()

clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(train.drop('label', axis=1), train.label)
clf.predict(test.drop('label', axis=1))
print(confusion_matrix(test.label, clf.predict(test.drop('label', axis=1))))
print(classification_report(test.label, clf.predict(test.drop('label', axis=1))))

rfc = ensemble.RandomForestClassifier(max_features=2)
rfc = rfc.fit(train.drop('label', axis=1), train.label)
rfc.predict(test.drop('label', axis=1))
print(confusion_matrix(test.label, rfc.predict(test.drop('label', axis=1))))
print(classification_report(test.label, rfc.predict(test.drop('label', axis=1))))

params ={
    'max_depth':[2,4,6],
    'min_samples_leaf':[2,4,6]
}
rf = ensemble.RandomForestClassifier(random_state=0, n_jobs=-1)
grid_cv = GridSearchCV(rf, param_grid=params, cv=2, n_jobs=-1)
grid_cv.fit(train.drop('label', axis=1), train.label)

print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))
a = 1