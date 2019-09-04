# -*- coding: utf-8 -*-
"""
获得向量趋势
训练时候的时间数据是用从 1到step
1.通过get_trend/windowed_PolyStep22.py（或者其他文件， 最后个那个22数字代表窗口长度）训练窗口趋势向量。
wsg
变窗口
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit    #非线性最小二乘法拟合
import time
import datetime
start=datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))

path='D:/python_code/NEW_step/try_paper/data/Data/Brent-11-15.xlsx'
resultsPath = 'D:/python_code/NEW_step/try_paper/mycode/new/'
# path='D:/python_code/NEW_step/try_paper/different/data5_pro.csv'
# resultsPath = 'D:/python_code/NEW_step/try_paper/different/'
df = pd.read_excel(path,sheet_name='Sheet1',header=None)

# df = pd.read_csv(path,header=None)
# 选择拟合函数的形式
def Poly2func(x,a,b,c):
    return a*np.power(x,2)+b*np.power(x,1)+c


def Poly3func(x,a,b,c,d):
    return a*np.power(x,3)+b*np.power(x,2)+c*np.power(x,1)+d


def Poly4func(x,a,b,c,d,e):
    return a*np.power(x,4)+b*np.power(x,3)+c*np.power(x,2)+d*np.power(x,1)+e




# 获取多项式拟合的参数
def getPoly2Params(a,b):
    # x=np.array(df.loc[a:b,0].astype('float64'))
    # y=np.array(df.loc[a:b,1].astype('float64'))
    x = np.array(range(0, b-a+1))
    y = np.array(df.loc[a:b,1])
    # trend=np.polyfit(x,y,4)
    trend,pocv=curve_fit(Poly2func,x,y)
    trend=list(trend)
    return trend


def getPoly3Params(a,b):
    # x=np.array(df.loc[a:b,0].astype('float64'))
    # y=np.array(df.loc[a:b,1].astype('float64'))
    x = np.array(range(0, b-a+1))
    y = np.array(df.loc[a:b,1])
    trend,pocv=curve_fit(Poly3func,x,y)    #返回值1.参数的最佳值使得残差的平方和最小化 2.估计的popt协方差。对角线提供参数估计的方差。要计算参数使用的一个标准偏差
    trend=list(trend)
    return trend


def getPoly4Params(a,b):
    # x=np.array(df.loc[a:b,0].astype('float64'))
    # y=np.array(df.loc[a:b,1].astype('float64'))
    x = np.array(range(0, b-a+1))
    y = np.array(df.loc[a:b,1])
    #trend=np.polyfit(x,y,4)
    trend,pocv=curve_fit(Poly4func,x,y)
    trend=list(trend)
    return trend





##############算各个模型的MSE######
def getPoly2MSE(i,j,a,b,c):
    x = np.array(range(0, j-i+1))
    y = np.array(df.loc[i:j,1])
    y2 = list(Poly2func(x,a,b,c))
    temp = np.power(y2-y,2)
    n = np.sum(temp)
    return n/len(temp)


def getPoly3MSE(i,j,a,b,c,d):
    x = np.array(range(0, j-i+1))
    y = np.array(df.loc[i:j,1])
    y2 = list(Poly3func(x,a,b,c,d))
    temp = np.power(y2-y,2)
    n = np.sum(temp)
    return n/len(temp)


def getPoly4MSE(i,j,a,b,c,d,e):
    x = np.array(range(0, j-i+1))
    y = np.array(df.loc[i:j,1])
    y2 = list(Poly4func(x,a,b,c,d,e))
    temp = np.power(y2-y,2)
    n = np.sum(temp)
    return n/len(temp)


#########获取MAPE############


def getPoly2MAPE(i,j,a,b,c):
    x = np.array(range(0, j-i+1))
    y = np.array(df.loc[i:j,1])
    y2 = list(Poly2func(x,a,b,c))
    temp = np.power(y2-y,2)/y
    return np.sum(temp)/len(temp)


def getPoly3MAPE(i,j,a,b,c,d):
    x = np.array(range(0, j-i+1))
    y = np.array(df.loc[i:j,1])
    y2 = list(Poly3func(x,a,b,c,d))
    temp = np.power(y2-y,2)/y
    return np.sum(temp)/len(temp)


def getPoly4MAPE(i,j,a,b,c,d,e):
    x = np.array(range(0, j-i+1))
    y = np.array(df.loc[i:j,1])
    y2 = list(Poly4func(x,a,b,c,d,e))
    temp = np.power(y2-y,2)/y
    return np.sum(temp)/len(temp)


#######计算各个函数的MAE########
def getPoly2MAD(i,j,a,b,c):
    x = np.array(range(0, j-i+1))
    y = np.array(df.loc[i:j,1])
    y2 = list(Poly2func(x,a,b,c))
    temp = np.abs(y-np.abs(y2))
    return np.sum(temp)/len(temp)


def getPoly3MAD(i,j,a,b,c,d):
    x = np.array(range(0, j-i+1))
    y = np.array(df.loc[i:j,1])
    y2 = list(Poly3func(x,a,b,c,d))
    temp = np.abs(y-np.abs(y2))
    return np.sum(temp)/len(temp)


def getPoly4MAD(i,j,a,b,c,d,e):
    x = np.array(range(0, j-i+1))
    y = np.array(df.loc[i:j,1])
    y2 = list(Poly4func(x,a,b,c,d,e))
    temp = np.abs(y-np.abs(y2))
    return np.sum(temp)/len(temp)


######预测函数#######
def predictPoly2(i,j,a,b,c):
    #i,j是数据起始点
    #a,b,c是二次函数参数
    x = np.array(df.loc[i:j,0])
    result = list(Poly2func(x,a,b,c))
    return result


def predictPoly3(i,j,a,b,c,d):
    #i,j是数据起始点
    #a,b,c是二次函数参数
    x = np.array(df.loc[i:j,0])
    result = list(Poly3func(x,a,b,c,d))
    return result


def predictPoly4(i,j,a,b,c,d,e):
    #i,j是数据起始点
    #a,b,c是二次函数参数
    x = np.array(df.loc[i:j,0])
    result = list(Poly4func(x,a,b,c,d,e))
    return result


######获取残差#######
def getResidual(a,b,y):
    #a,b是原始时间序列的起始点
    #y是预测结果
    x = np.array(df.loc[a:b,1])
    return x-y


##########计算各个模型的BIC#########
#如果你的趋势向量没有提取出足够信息的话，每一个残差还是不是相互独立的就是个问题
#还能用极大似然吗？


if __name__ == '__main__':
    AA = [2,5,10,15]
    BB = [2,5,10,15]
    ID = []
    dataList=[]
    FuncDic = {'Poly2':getPoly2Params,'Poly3':getPoly3Params,'Poly4':getPoly4Params}
    MSEfunc = {'Poly2':getPoly2MSE,'Poly3':getPoly3MSE,'Poly4':getPoly4MSE}
    ErrorCount = {'Poly2':0,'Poly3':0,'Poly4':0}
    for a in AA:
        for b in BB:
            for i in range(a, df.shape[0] - b):    #df.shape[0]  5538
                MSE = {'Poly2': 10000, 'Poly3': 10000, 'Poly4':10000}
                Trend = {'Poly2':[],'Poly3':[],'Poly4':[]}
                ID.append(i)
                for key in FuncDic:
                    try:
                        trend = FuncDic[key](i-a, i + b)
                        print('外循环', i, ',内循环', key, '标签已获取！')
                        if key == 'Poly2':
                            MSE[key] = MSEfunc[key](i-a,i+b,trend[0],trend[1],trend[2])
                            trend = [0,0]+trend
                        elif key == 'Poly3':
                            MSE[key] = MSEfunc[key](i-a,i+b,trend[0],trend[1],trend[2],trend[3])
                            trend = [0]+trend
                        elif key == 'Poly4':
                            MSE[key] = MSEfunc[key](i-a,i+b,trend[0],trend[1],trend[2],trend[3],trend[4])



                        Trend[key] = trend
                        # print(MSE[key])
                        if i+b>df.shape[0]:
                            iEnd = df.shape[0]
                        else:
                            iEnd = i+b
                    except RuntimeError as e:
                            print(e)
        # print(MSE)
                data = {
                    'istart': i-a,
                    'iend': i+b,
                    'trend':Trend,
                    'MSE': MSE,
                    'MSE_FinalFunc': min(MSE,key=MSE.get),
                    'MSE_FinalTrend': Trend[min(MSE,key=MSE.get)],    #here
                    'a':a,
                    'b':b
                }
                dataList.append(data)

    istart = []
    iend = []
    MSEfinalFunc = []    #用来存经过选择之后最后的函数类型
    MSEtrenda = []
    MSEtrendb = []
    MSEtrendc = []
    MSEtrendd = []
    MSEtrende = []
    A = []
    B = []

    for i in range(len(dataList)):
        istart.append(dataList[i].get('istart'))
        iend.append(dataList[i].get('iend'))
        MSEfinalFunc.append(dataList[i].get('MSE_FinalFunc'))
        MSEtrenda.append(dataList[i].get('MSE_FinalTrend')[0])
        MSEtrendb.append(dataList[i].get('MSE_FinalTrend')[1])
        MSEtrendc.append(dataList[i].get('MSE_FinalTrend')[2])
        MSEtrendd.append(dataList[i].get('MSE_FinalTrend')[3])
        MSEtrende.append(dataList[i].get('MSE_FinalTrend')[4])
        A.append(dataList[i].get("a"))
        B.append(dataList[i].get("b"))


        print('第', i + 1, '行已成功写入！')

    #定义系数的输出
    dataframe_params = pd.DataFrame({
        'i':ID,
        'istart':istart ,
        'iend':iend,
        'MSE_FinalFunc':MSEfinalFunc,
        'MSEtrenda':MSEtrenda,
        'MSEtrendb':MSEtrendb,
        'MSEtrendc':MSEtrendc,
        'MSEtrendd':MSEtrendd,
        'MSEtrende':MSEtrende,
        'a':A,
        'b':B
    })
    dataframe_params.to_csv(resultsPath+'Poly''-'+str(AA)+str(BB)+'-WINParams'+'-'+str(start)[0:10]+".csv")




end=datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))
print(ErrorCount)
print('本次程序运行时长:',end-start)
