'''
By:shenqiti
2019/9/4
分段预测
'''

import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math

import statsmodels.api as sm
import statsmodels.stats.diagnostic




# #——————————————————————————————————————————————————————————————




path = 'D:/python_code/NEW_step/try_paper/mycode/new/media.xlsx'
oil_price_path = 'D:/python_code/NEW_step/try_paper/mycode/new/media_oil_price.xlsx'

df = pd.read_excel(path,header=0)
price_df = pd.read_excel(oil_price_path,header=None,sheet_name='Sheet1')


OIL_PRICE = []

trenda = np.array(df.loc[:, 'trenda'].astype('float64'))
trendb = np.array(df.loc[:, 'trendb'].astype('float64'))
trendc = np.array(df.loc[:, 'trendc'].astype('float64'))
trendd = np.array(df.loc[:, 'trendd'].astype('float64'))
trende = np.array(df.loc[:, 'trende'].astype('float64'))
AA = np.array(df["a"])
BB = np.array(df['b'])
XX = []



for i in range(0,len(AA)):
    XX.append(AA[i]+BB[i])


OIL = []
PRE = []
True_price = []
SUM = 0
cnt = 20
for t in range(4,31):
    SUM = 0  #MSE
    SUM2 = 0   #MAE
    SUM3 = 0  #MAPE
    SUM4 = 0   #SSE
    SUM5 = 0  #RMSE
    for i in range(len(XX)-cnt-5032,len(XX)-5032):
        ya = []
        yb = []
        yc = []
        yd = []
        ye = []
        x = []

        if XX[i-1] != XX[i-2] and XX[i-1]<t:

            x.append(XX[i - 1])
            x.append(XX[i - 2])
            x.append(XX[i - 3])
            ya.append(trenda[i - 1])
            ya.append(trenda[i - 2])
            ya.append(trenda[i - 3])
            yb.append(trendb[i - 1])
            yb.append(trendb[i - 2])
            yb.append(trendb[i - 3])
            yc.append(trendc[i - 1])
            yc.append(trendc[i - 2])
            yc.append(trendc[i - 3])
            yd.append(trendd[i - 1])
            yd.append(trendd[i - 2])
            yd.append(trendd[i - 3])
            ye.append(trende[i - 1])
            ye.append(trende[i - 2])
            ye.append(trende[i - 3])
            x = sm.add_constant(x)

            resulta = sm.OLS(ya, x).fit().params
            resultb = sm.OLS(yb, x).fit().params
            resultc = sm.OLS(yc, x).fit().params
            resultd = sm.OLS(yd, x).fit().params
            resulte = sm.OLS(ye, x).fit().params

            ka = resulta[1] * XX[i - 1] + resulta[0]
            kb = resultb[1] * XX[i - 1] + resultb[0]
            kc = resultc[1] * XX[i - 1] + resultc[0]
            kd = resultd[1] * XX[i - 1] + resultd[0]
            ke = resulte[1] * XX[i - 1] + resulte[0]
            oil_price = trenda[i]*pow(XX[i],4)+trendb[i]*pow(XX[i],3)+trendc[i]*pow(XX[i],2)+trendd[i]*pow(XX[i],1)+trende[i]  #拟合
            price = ka*pow(XX[i-1]+1,4) + kb*pow(XX[i-1]+1,3) +kc*pow(XX[i-1]+1,2)+kd*pow(XX[i-1]+1,1)+ke  #预测

        elif XX[i-1] != XX[i-2] and XX[i-1]>=t:
            x.append(XX[i - 1])
            x.append(XX[i - 2])

            ya.append(trenda[i - 1])
            ya.append(trenda[i - 2])

            yb.append(trendb[i - 1])
            yb.append(trendb[i - 2])

            yc.append(trendc[i - 1])
            yc.append(trendc[i - 2])

            yd.append(trendd[i - 1])
            yd.append(trendd[i - 2])

            ye.append(trende[i - 1])
            ye.append(trende[i - 2])

            x = sm.add_constant(x)
            resulta = sm.OLS(ya, x).fit().params
            resultb = sm.OLS(yb, x).fit().params
            resultc = sm.OLS(yc, x).fit().params
            resultd = sm.OLS(yd, x).fit().params
            resulte = sm.OLS(ye, x).fit().params


            ka = resulta[1] * XX[i - 1] + resulta[0]
            kb = resultb[1] * XX[i - 1] + resultb[0]
            kc = resultc[1] * XX[i - 1] + resultc[0]
            kd = resultd[1] * XX[i - 1] + resultd[0]
            ke = resulte[1] * XX[i - 1] + resulte[0]
            oil_price = trenda[i]*pow(XX[i],4)+trendb[i]*pow(XX[i],3)+trendc[i]*pow(XX[i],2)+trendd[i]*pow(XX[i],1)+trende[i]  #拟合
            price = ka*pow(XX[i-1]+1,4) + kb*pow(XX[i-1]+1,3) +kc*pow(XX[i-1]+1,2)+kd*pow(XX[i-1]+1,1)+ke   #预测

        elif XX[i-1]==XX[i-2] and XX[i-1]>=t:
            x.append(XX[i - 1])
            x.append(XX[i - 2])

            ya.append(trenda[i - 1])
            ya.append(trenda[i - 2])

            yb.append(trendb[i - 1])
            yb.append(trendb[i - 2])

            yc.append(trendc[i - 1])
            yc.append(trendc[i - 2])

            yd.append(trendd[i - 1])
            yd.append(trendd[i - 2])

            ye.append(trende[i - 1])
            ye.append(trende[i - 2])
            resulta = sm.OLS(ya, x).fit().params
            resultb = sm.OLS(yb, x).fit().params
            resultc = sm.OLS(yc, x).fit().params
            resultd = sm.OLS(yd, x).fit().params
            resulte = sm.OLS(ye, x).fit().params

            ka = resulta[0] * XX[i - 1]
            kb = resultb[0] * XX[i - 1]
            kc = resultc[0] * XX[i - 1]
            kd = resultd[0] * XX[i - 1]
            ke = resulte[0] * XX[i - 1]
            oil_price = trenda[i]*pow(XX[i],4)+trendb[i]*pow(XX[i],3)+trendc[i]*pow(XX[i],2)+trendd[i]*pow(XX[i],1)+trende[i]
            price = ka*pow(XX[i-1]+1,4) + kb*pow(XX[i-1]+1,3) +kc*pow(XX[i-1]+1,2)+kd*pow(XX[i-1]+1,1)+ke

        elif XX[i-1]==XX[i-2] and XX[i-1]<t:
            x.append(XX[i - 1])
            x.append(XX[i - 2])
            x.append(XX[i - 3])
            ya.append(trenda[i - 1])
            ya.append(trenda[i - 2])
            ya.append(trenda[i - 3])
            yb.append(trendb[i - 1])
            yb.append(trendb[i - 2])
            yb.append(trendb[i - 3])
            yc.append(trendc[i - 1])
            yc.append(trendc[i - 2])
            yc.append(trendc[i - 3])
            yd.append(trendd[i - 1])
            yd.append(trendd[i - 2])
            yd.append(trendd[i - 3])
            ye.append(trende[i - 1])
            ye.append(trende[i - 2])
            ye.append(trende[i - 3])
            x = sm.add_constant(x)

            resulta = sm.OLS(ya, x).fit().params
            resultb = sm.OLS(yb, x).fit().params
            resultc = sm.OLS(yc, x).fit().params
            resultd = sm.OLS(yd, x).fit().params
            resulte = sm.OLS(ye, x).fit().params

            ka = resulta[0] * XX[i - 1]
            kb = resultb[0] * XX[i - 1]
            kc = resultc[0] * XX[i - 1]
            kd = resultd[0] * XX[i - 1]
            ke = resulte[0] * XX[i - 1]
            oil_price = trenda[i]*pow(XX[i],4)+trendb[i]*pow(XX[i],3)+trendc[i]*pow(XX[i],2)+trendd[i]*pow(XX[i],1)+trende[i]  #拟合 趋势向量还原成石油价格的拟合值
            price = ka*pow(XX[i-1]+1,4) + kb*pow(XX[i-1]+1,3) +kc*pow(XX[i-1]+1,2)+kd*pow(XX[i-1]+1,1)+ke  #预测


        error = pow(price_df.loc[i,1]-price,2)
        error2 = abs(price_df.loc[i,1]-price)
        error3 = abs((price_df.loc[i,1]-price)/price_df.loc[i,1])
        error4 = pow(price_df.loc[i,1]-price,2)
        SUM  = SUM + error
        SUM2 = SUM2 + error2
        SUM3 = SUM3 + error3
        SUM4 = SUM4 + error4

        True_price.append(price_df.loc[i,1])
        PRE.append(price)
        OIL.append(oil_price)
        # print('预测第',i+1,'个点的油价',',预测值为',price,'真实值为',price_df.loc[i,1],'拟合值为', oil_price,'预测误差为:',error,'XX[i-1]:',XX[i-1])
    MSE = SUM / cnt
    MAE = SUM2 / cnt
    MAPE = SUM3 /cnt
    SSE = SUM4
    RMSE = math.sqrt(MSE)
    print(str(t)+"时刻的"+"MSE=",MSE,"MAE=",MAE,'MAPE=',MAPE,'SSE=',SSE,'RMSE=',RMSE)

print(len(XX))
plt.plot(True_price)
plt.plot(OIL)
plt.plot(PRE)
plt.title("Brent Oil Price")
plt.legend(["True_price","Fitting_price","Predict_price"])
plt.show()
