'''
By:shenqiti
2019/9/4


'''

# coding:utf-8
import pywt
import datetime
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from matplotlib import pyplot as plt
import xlwt
start=datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))
path=r'C:\Users\123\Desktop\Data_170820.xlsx'
df=pd.read_excel(path,sheet_name='Brent',header=None)
path2=r'E:\svm结构性断点论文\20171220\月度\趋势向量-月度\result-linar-monthly.xls'
df2=pd.read_excel(path2,sheet_name='Sheet')
trenda = np.array(df2.ix[:, 1].astype('float64'))
trendb = np.array(df2.ix[:, 2].astype('float64'))
def getWaves(a,b):
    price=np.array(df.iloc[a:b+1,2])
    # 小波分解
    #A8,D8,D7,D6,D5,D4,D3,D2,D1=pywt.wavedec(price,'db4',mode='sym',level=4)
    A8, D4, D3, D2, D1 = pywt.wavedec(price, 'db4', mode='sym', level=4)
    #coff=[A2,D2,D1]
    coff=[A8,D4,D3,D2,D1]
    # ARIMA定阶
    order_A8=sm.tsa.arma_order_select_ic(A8,ic='aic')['aic_min_order']
   # order_D8 = sm.tsa.arma_order_select_ic(D8, ic='aic')['aic_min_order']
   # order_D7 = sm.tsa.arma_order_select_ic(D7, ic='aic')['aic_min_order']
   # order_D6 = sm.tsa.arma_order_select_ic(D6, ic='aic')['aic_min_order']
   # order_D5 = sm.tsa.arma_order_select_ic(D5, ic='aic')['aic_min_order']
    order_D4 = sm.tsa.arma_order_select_ic(D4, ic='aic')['aic_min_order']
    order_D3 = sm.tsa.arma_order_select_ic(D3, ic='aic')['aic_min_order']
    order_D2=sm.tsa.arma_order_select_ic(D2,ic='aic')['aic_min_order']
    order_D1=sm.tsa.arma_order_select_ic(D1,ic='aic')['aic_min_order']
    #print(order_A2,order_D1,order_D2)
    #AMRA模型建模
    model_A8=ARMA(A8,order=order_A8)
    #model_D8 = ARMA(D8, order=order_D8)
    #model_D7 = ARMA(D2, order=order_D7)
    #model_D6 = ARMA(D1, order=order_D6)
    #model_D5 = ARMA(D2, order=order_D5)
    model_D4 = ARMA(D1, order=order_D4)
    model_D3 = ARMA(D2, order=order_D3)
    model_D2=ARMA(D2,order=order_D2)
    model_D1=ARMA(D1,order=order_D1)
    # 拟合数据
    result_A8=model_A8.fit()
  #  result_D8 = model_D8.fit()
  #  result_D7=model_D7.fit()
  #  result_D6 = model_D6.fit()
  #  result_D5 = model_D5.fit()
    result_D4 = model_D4.fit()
    result_D3 = model_D3.fit()
    result_D2 = model_D2.fit()
    result_D1 = model_D1.fit()

    pA8=model_A8.predict(params=result_A8.params,start=1,end=len(A8))
   # pD8 = model_D8.predict(params=result_D8.params, start=1, end=len(D8))
   # pD7 = model_D7.predict(params=result_D7.params, start=1, end=len(D7))
   # pD6 = model_D6.predict(params=result_D6.params, start=1, end=len(D6))
   # pD5 = model_D5.predict(params=result_D5.params, start=1, end=len(D5))
    pD4 = model_D4.predict(params=result_D4.params, start=1, end=len(D4))
    pD3 = model_D3.predict(params=result_D3.params, start=1, end=len(D3))
    pD2 = model_D2.predict(params=result_D2.params, start=1, end=len(D2))
    pD1 = model_D1.predict(params=result_D1.params, start=1, end=len(D1))


   # coffnew=[pA2,pD2,pD1]
    coffnew=[pA8,pD4,pD3,pD2,pD1]
    return coffnew
#coffnew=getWaves(1,df.shape[0]-1)
#list=pywt.waverec(coffnew,'db4')
#print(coffnew)
# 小波预测
def getPredictWaveCoff(a,b,level):
    data_train=df.iloc[a:b+1,2]
    data_whole=df.iloc[a:b+29,2]
    adList=pywt.wavedec(data_train, 'db1', level=level)
    adList_all=pywt.wavedec(data_whole,'db1',level=level)
    coffnew=[]
    for i in range(len(adList)):
        delta=len(adList_all[i])-len(adList[i])
        if b==6899:
            print('aaa')
        order = sm.tsa.arma_order_select_ic(adList[i], ic='aic')['aic_min_order']
        model=ARMA(adList[i],order=order)
        try:
            result=model.fit()
        except:
            print(b+1,'except')
            model = ARMA(adList[i], order=(0, 0))
            result = model.fit()
        p=model.predict(params=result.params,start=a,end=len(adList[i])+delta)
        coffnew.append(p)
    data_predict=pywt.waverec(coffnew,'db1')
    data_predict=list(data_predict)
    l=len(data_predict)
 #   print('test',data_predict[-1],data_predict[-2],data_predict[-3],list(data_train)[-1])
    return data_predict[-29:-1]
# 导出拟合数据到excel
def writeToExcel():
    coffnew = getWaves(1, df.shape[0] - 1)
    price = np.array(df.iloc[1:, 2])
    dprice = pywt.waverec(coffnew, 'db4')
    wb=xlwt.Workbook()
    ws=wb.add_sheet('Sheet1')
    for i in range(len(dprice)):
        ws.write(i,0,i+1)
        ws.write(i,1,list(dprice)[i])
    wb.save(r'C:\Users\123\Desktop\result.xls')
# 获取多项式拟合的参数
def getPolyParams(x,y):
    x1=[]
    y1=[]
    for i in range(len(x)):
        x1.append(x[i])
        y1.append(y[i])
    trend=np.polyfit(x1,y1,1)
    trend=list(trend)
    return trend
wb=xlwt.Workbook()
ws=wb.add_sheet('Sheet1')
dprice = getPredictWaveCoff(1,7861,1)
#print(dprice,len(dprice))
for i in range(len(dprice)):
    ws.write(i,0,i+1)
    ws.write(i,1,list(dprice)[i])
wb.save(r'C:\Users\123\Desktop\result.xls')

'''
# 逐步预测
step=22

print(df.shape[0])
print(len(trenda))
mape=0;avError=0;sse=0;
for i in range(df.shape[0]-1-step-7,df.shape[0]-step-1):
#for i in range(6900-5, 6900):
    data_train=list(df.iloc[i-step+1:i+1,2])
    data_predict=getPredictWaveCoff(i-step+1,i,1)
    print('predict',data_predict)
    data_train.append(data_predict)
    x = [k for k in range(i-step+1, i+2)]
    trend=getPolyParams(x,data_train)
    print(i+1, trenda[i])
    error = (0.5) * (np.square(trend[0] - trenda[i]) + np.square(trend[1] - trendb[i]))
    mape += np.average([np.abs(( trend[0]- trenda[i]) / trenda[i]), np.abs((trend[1] - trendb[i]) / trendb[i])])
    avError += error
    sse += error * 2
    print('预测第', i + 1, '个点', ',平方损失为:', error, ',mape=', \
          100*np.average([np.abs((trend[0] - trenda[i]) / trenda[i]),np.abs((trend[1] - trendb[i]) / trendb[i])]),'%' \
            , ',sse=', error * 2)
print('平均下来:loss=', avError / 7, 'mape=', mape / 7, ',sse=', sse / 7)
'''




'''
# 画图
plt.figure(figsize=(10,15))
plt.subplot(3,1,1)
    #plt.hold
plt.plot(list(price)[6462:6709],'blue')
plt.hold
plt.plot(list(dprice)[6462:6709],'red')

    #plt.hold
    #plt.plot(result_D2.fittedvalues,'yellow')
    #plt.hold
    #plt.plot(result_D1.fittedvalues,'black')
plt.show()
'''
