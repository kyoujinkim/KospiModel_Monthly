import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt


def getCumProd(m_arr, d_arr):

    main_arr = pd.DataFrame()
    for idx in range(len(m_arr)):
        if idx < len(m_arr) - 1:
            currdate = m_arr.index[idx]
            nextdate = m_arr.index[idx + 1]
        else:
            currdate = m_arr.index[idx]
            nextdate = currdate + relativedelta(months=1)

        cumprod_arr = ((1 + d_arr[currdate:nextdate] / 100).cumprod() - 1) * 100
        main_arr = pd.concat([main_arr, cumprod_arr])

    main_arr = main_arr[~main_arr.index.duplicated(keep='last')]

    return main_arr

def calc_holding(x):
    if x.signal == 0 :
        new_holding =  1 - x.percentile
    else:
        new_holding =  x.percentile

    return new_holding

def backtest(testbed):
    '''
    :param testbed: Backdata for backtesting
    :return: Backtested Portfolio Details
    '''

    method = 'SimpleRebalance' #DynamicRebalance, SimpleRebalance
    port = pd.DataFrame({'equVal': .5,
                         'cashVal': .5},
                        index=[testbed.index[0] - relativedelta(days=1)])
    testbed['dsignal'] = (0.5 - testbed.holding).abs().apply(lambda x: 1 if x>=0.3 else 0)
    for date, row in testbed.iterrows():
        if row.signal == 1:
            currPort = port.iloc[-1].equVal * (1 + row.RTN / 100)
            totalVal = currPort + port.iloc[-1].cashVal
        else:
            currPort = port.iloc[-1].equVal * (1 - row.RTN / 100)
            totalVal = currPort + port.iloc[-1].cashVal

            if (row.rebal_flag == 1):
                port_row = pd.DataFrame({'equVal': totalVal * row.holding,
                                         'cashVal': totalVal * (1 - row.holding)},
                                        index=[date])
                port = pd.concat([port, port_row])

        if method == 'DynamicRebalance':
            if (row.holding == 0) or (row.holding == 1):
                port_row = pd.DataFrame({'equVal': totalVal * row.holding,
                                         'cashVal': totalVal * (1 - row.holding)},
                                        index=[date])
                port = pd.concat([port, port_row])
            else:
                port_row = pd.DataFrame({'equVal': currPort,
                                         'cashVal': totalVal - currPort},
                                        index=[date])
                port = pd.concat([port, port_row])

        elif method == 'SimpleRebalance':
            if (row.holding <= 0.2):
                port_row = pd.DataFrame({'equVal': totalVal * row.holding,
                                         'cashVal': totalVal * (1 - row.holding)},
                                        index=[date])
                port = pd.concat([port, port_row])
            elif (row.holding >= 0.8):
                port_row = pd.DataFrame({'equVal': totalVal * row.holding,
                                         'cashVal': totalVal * (1 - row.holding)},
                                        index=[date])
                port = pd.concat([port, port_row])
            else:
                port_row = pd.DataFrame({'equVal': currPort,
                                         'cashVal': totalVal - currPort},
                                        index=[date])
                port = pd.concat([port, port_row])

    port['totalPort'] = port.apply(lambda x: x.sum(), axis=1)
    port['idx'] = (testbed.RTN / 100 + 1).cumprod()

    return port

interval_pred = pd.read_csv('interval_result.csv', parse_dates=True).dropna()
interval_pred.columns = ['Date']+interval_pred.columns[1:].to_list()
interval_pred.Date = pd.to_datetime(interval_pred.Date)

d_rtn = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kdrtn', index_col=0, parse_dates=True)

interval_pred_Monthly = interval_pred.groupby([interval_pred.Date.dt.year, interval_pred.Date.dt.month]).max()
interval_pred_Monthly.set_index(keys='Date', drop=True, inplace=True)
rebal_flag = pd.DataFrame(np.ones(len(interval_pred_Monthly)), index= interval_pred_Monthly.index, columns=['rebal_flag'])

enddate = interval_pred_Monthly.index[-1] + relativedelta(months=1)
d_rtn = d_rtn[interval_pred_Monthly.index[0]:enddate]

cumprod = getCumProd(interval_pred_Monthly, d_rtn)
cumprod.columns = ['CumRtn']

testbed = pd.concat([d_rtn, cumprod, interval_pred_Monthly], axis=1).ffill()
testbed = pd.concat([testbed, rebal_flag], axis=1).fillna(0)

testbed['signal'] = testbed.y_Rtn.apply(lambda x: 1 if x>0 else 0)
testbed['percentile'] = (testbed.RTN - testbed.y_Rtn_lower_bound) / (testbed.y_Rtn_upper_bound - testbed.y_Rtn_lower_bound)
testbed['holding'] = testbed.apply(calc_holding, axis=1)

port = backtest(testbed)

port.totalPort.plot()
port.idx.plot()
plt.show()
a = 1
