import pandas as pd
from dateutil.relativedelta import relativedelta

def averageByVariable(x):
    result = []
    for idx in range(3, len(x)+1, 3):
        result.append(x[idx - 3 : idx].mean())

    return result

def averageByVariable2(x):
    result = []
    for idx in range(0, 3):
        result.append((x[idx] + x[idx+3] + x[idx+6] + x[idx+9] + x[idx+12] + x[idx+15]) / 6)

    return result

def getLabelArray(prdData, dIdx):
    '''
    Get Label data for evaluate prediction results
    :param prdData: Array of Prediction Results
    :param dIdx: Array of Price, used for calculating Label
    :return: Label Data
    '''
    # make actual value array(Label Value)
    yActDf = pd.DataFrame()
    for rowName, row in prdData.iterrows():
        # designate Range for calculate return
        startDate = rowName
        endDate = rowName + relativedelta(weeks=4)

        dRange = dIdx.loc[startDate: endDate]

        # Calculate Return
        yAct = (dRange.iloc[-1] / dRange.iloc[0] - 1) * 100
        # designate date of label
        yAct.index = [startDate]

        # Append to total Array
        yActDf = pd.concat([yActDf, yAct], axis=0)

    # set label array name as y
    yActDf.columns = ['y']

    return yActDf

def getMae(df, rollingWindow):
    '''

    :param df: dataframe with label data
    :param rollingWindow: MAE average rolling window
    :return: mae dataframe
    '''
    #get MAE value
    mae = df.apply(getMaeApply, axis=1)
    #drop label value
    mae = mae.drop(labels='y', axis=1)
    #calculate 52 weeks rolling average MAE
    mae_rolling = mae.rolling(rollingWindow, min_periods=1).mean()

    return mae_rolling

def getMaeApply(x):
    return (x - x['y']).abs()

prdData = pd.read_csv('./result_ensemble.csv', index_col=0, parse_dates=True)
dIdx = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kidx', index_col=0, parse_dates=True)

#average each model's prediction value by type of variables
prdData = prdData.apply(averageByVariable2, axis=1, result_type='expand')

#get label array
yActDf = getLabelArray(prdData, dIdx)

#append label to prediction array
data = pd.concat([prdData, yActDf], axis=1)

#get 52 weeks rolling window average MAE of each Models
mae = getMae(data, 52)

#choose BestModel
bestModel = mae.apply(lambda x: mae.columns[x.argmax()], axis=1).shift(4).dropna() #4
bestModel = pd.DataFrame(bestModel)

#append prediction value of BestModel
bestModel['prd'] = bestModel.apply(lambda x: data.loc[x.name][x].values[0], axis=1)

#append actual value of corresponding Date
bestModel = pd.concat([bestModel, data.y], axis=1).dropna()

a=1