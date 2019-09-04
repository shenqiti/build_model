# -*- coding: utf-8 -*-
'''
By:shenqiti
2019/7/9

'''


import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from arch import arch_model
import statsmodels.api as sm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import xlwt
from sklearn import linear_model
from scipy.optimize import curve_fit
from sklearn.svm import SVR

df=pd.read_excel('../data/Data/Brent-11-25.xlsx',sheet_name='Sheet1',header=None)
path2=r'E:\svm结构性断点论文\20171220\月度\趋势向量-月度\result-linar-monthly.xls'
df2=pd.read_excel(path2,sheet_name='Sheet')
trenda = np.array(df2.ix[:, 1].astype('float64'))
trendb = np.array(df2.ix[:, 2].astype('float64'))
xWhole=[]
yWhole=[]
for i in range(1,df.shape[0]):
    mx=[];my=[]
    mx.append(df.loc[i,0])
  #  mx.append(df.ix[i,3])
  #  mx.append(df.ix[i,4])
  #  mx.append(df.ix[i,5])
    my.append(df.loc[i,1])
    xWhole.append(mx)
    yWhole.append(my)
x_train=xWhole[1:df.shape[0]]
y_train=yWhole[1:df.shape[0]]

#print(x_train,'\n',y_train)
data=pd.Series(df.loc[1:,1])
print(data)
data_diff=data.diff(1)
#print(list(data_diff))
# BP神经网络
def getBPData(a,b):
    d1=list(df.iloc[a:b+1,2])
    d2=list(df.iloc[a:b+1,3])
    d3=list(df.iloc[a:b+1,4])
    d4=list(df.iloc[a:b+1,5])
    d5=list(df.iloc[a:b+1,6])
    x=[]
    y=[]
    for i in range(len(d1)):
        x.append([d2[i],d3[i],d4[i],d5[i]])
        y.append([d1[i]])
    return x,y
def getBPResult(x,y,x_test):
    #x=x;y=y
    clf=MLPRegressor(solver='adam',batch_size=128,activation='tanh',hidden_layer_sizes=(5,50),learning_rate='adaptive')
    clf.fit(x,y)
    #print(np.array(xWhole[6793:]),'\n')
    #print(np.array(xWhole[6793:]).reshape(1,-1))
    yr=clf.predict(np.array(x_test))
    pre=list(yr)
    print(pre)
    return pre
#pre=getBPResult(xWhole,yWhole,xWhole)


#for i in range(6793,len(yWhole)):
#    pre=res.forecast(start=i,horizon=1)
#    print(pre.mean[i:i+1],'\n')
#ARIMA
def getARIMAResult(y):
    #mod=sm.tsa.ARMA(list(data_diff)[1:],order=(1,1))
    mod=sm.tsa.ARIMA(y,order=(1,1,1))
    #mod=sm.tsa.statespace.SARIMAX(yWhole[1:],trend='c',order=(1,1,1))
    res=mod.fit(disp=-1)
    #print(res.summary())
    #print(res.fittedvalues,len(res.fittedvalues),type(res.fittedvalues))

    #样本内预测
    pre=res.predict(dynamic=False)
    #样本外预测
    #pre=(res.forecast(20))[0]
    print('len(pre):',len(pre),',len(list):', len(list(df.ix[2:df.shape[0], 2])))
    #result=list(pre)
    result=list(df.ix[2:df.shape[0],2])+pre[0:len(pre)]
    #result=y[len(y)-1]+pre
  #  print(result)
    return result
    #print(pre,'lenPre:',len(pre))
    #pre=res.forecast(len(yWhole[1:]))
#pre=getARIMAResult(yWhole)
# Lasso回归
def getLassoResult(x,y,x_test):
    model_lasso=linear_model.LassoCV(alphas=[1,0.1,0.001,0.005]).fit(x,y)
    y_lasso=model_lasso.predict(np.array(x_test).reshape(-1,1))
 #   print(list(y_lasso),len(y_lasso),type(y_lasso))
    return list(y_lasso)
#pre=getLassoResult(xWhole,yWhole,xWhole)
# NB分类
def getNBResult(x,y,x_test):
    model_NB=MultinomialNB().fit(x,y)
    y_nb=model_NB.predict(x_test)
    print(list(y_nb),type(y_nb),len(y_nb))
    return y_nb
#y_nb=getNBResult(xWhole,yWhole,xWhole)
#GARCH
def getGARCH(y):
    am=arch_model(y,mean='AR',lags=1,vol='garch',p=1,o=0,q=1)
    res=am.fit()
    print(res.summary())
    pred=res.forecast(start=1,horizon=1).mean
    pre=[]
    for i in range(pred.shape[0]):
        pre.append(float(pred.ix[i]))
    return pre
#pre=getGARCH(yWhole)
#print(df.ix[2:df.shape[0]-2,2],2)

#SVR
def getSVRResult(x,y,x_test):
    svr=SVR(gamma=0.01,C=100,kernel='rbf')
    clf=svr.fit(x,y).predict(np.array(x_test).reshape(-1,1))
    return clf
def func(x,a,b,c,d,e):
    #return a * np.exp(-b * x) + c  # 指数函数
    #return a*np.cos(b*x+c)+d # 正弦函数
    return a*np.power(x,4)+b*np.power(x,3)+c*np.power(x,2)+d*np.power(x,1)+e
# 获取多项式拟合的参数
def getPolyParams(x,y):
    x1=[]
    y1=[]
    print(len(x),len(y))
    for i in range(len(x)):
        x1.append(x[i][0])
        y1.append(y[i][0])
    #print('看一眼:',x1,y1)
    trend=np.polyfit(x1,y1,1)
    #trend,pocv=curve_fit(func,x,y)
    trend=list(trend)
    return trend
#多次一步预测取平均
def getMultiPredictResult(a,b):
    avError = 0
    mape = 0
    sse = 0
    for i in range(a,b) :
        y=yWhole[0:i]
        #pre=getARIMAResult(y)
        #pre=getLassoResult(xWhole[0:i],y,xWhole[i])
        x_train,y_train=getBPData(5,i)
        x_test,y_test=getBPData(i,i)
        pre = getBPResult(x_train, y_train, x_test)
        #pre=getSVRResult(xWhole[0:i], y, xWhole[i])
        y.append(pre)
        x=xWhole
        trend=getPolyParams(xWhole[0:i+1],y)
        error = (0.5) * (np.square(trend[0] - trenda[i]) + np.square(trend[1] - trendb[i]))
        mape += np.average([np.abs(( trend[0]- trenda[i]) / trenda[i]), np.abs((trend[1] - trendb[i]) / trendb[i])])
        avError += error
        sse += error * 2
        print('预测第', i + 1, '个点', ',平方损失为:', error, ',mape=', \
              np.average([np.abs((trend[0] - trenda[i]) / trenda[i]), np.abs((trend[1] - trendb[i]) / trendb[i])]) \
                , ',sse=', error * 2)
    print('loss=', avError / 7, 'mape=', mape / 7, ',sse=', sse / 7)
if __name__ == '__main__':
    x_train = xWhole[0:7861]
    y_train = yWhole[0:7861]
    x_test = xWhole[7861:7890]
  #step=22
  #getMultiPredictResult(6903-7,6903)
    pre=getARIMAResult(yWhole)
  #pre=getLassoResult(xWhole,yWhole,xWhole)
    #pre=getSVRResult(x_train,y_train,x_test)
    print(pre)
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Sheet1')
    for i in range(len(pre)-28,len(pre)):
        ws.write(i + 1, 1, pre[i])
    wb.save(r'C:\Users\123\Desktop\result.xls')
  #pre=getBPResult(xWhole,yWhole,xWhole)
'''
  pre=getGARCH(yWhole[0:])
  print('len',len(pre))
'''

#print(len(res.forecast(len(yWhole[1:]))))
