# import pandas as pd
# import numpy as np
#
# data_input = 'D:/python_code/NEW_step/try_paper/revision_data/brentprice.xlsx'
# data_output = 'D:/python_code/NEW_step/try_paper/revision_data/result/brentpricepropro.csv'
# data = np.array(pd.read_excel(data_input))
#
# price = []
# cnt  = []
# lap = 0
# for each in data:
#     lap = lap + 1
#     for i in range(0,361):
#         cnt.append(lap)
#         price.append(each[0])
# print(len(price))
# fn = open(data_output,'w')
# for i in range(0,len(price)):
#     fn.write(str(cnt[i]))
#     fn.write(',')
#     fn.write(str(price[i]))
#     fn.write('\n')
# fn.close()
# print('ok')
########################################################################################################
import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import statsmodels.stats.diagnostic


# data_input = 'D:/python_code/NEW_step/try_paper/revision_data/result/WTI/Poly-[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20][2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]-WINParams-2020-03-03.csv'
# OUT_PUT = 'D:/python_code/NEW_step/try_paper/revision_data/result/WTI/al+~.csv'
# data = pd.read_csv(data_input)
# NO = np.array(data['Unnamed: 0'])
#
# i = np.array(data["i"])
# istart = np.array(data["istart"])
# iend = np.array(data["iend"])
# MSE_FinalFunc = np.array(data["MSE_FinalFunc"])
# MSEtrenda = np.array(data["MSEtrenda"])
# MSEtrendb = np.array(data["MSEtrendb"])
# MSEtrendc = np.array(data["MSEtrendc"])
# MSEtrendd = np.array(data["MSEtrendd"])
# MSEtrende = np.array(data["MSEtrende"])
# a = np.array(data["a"])
# b = np.array(data["b"])
#
# dic = {}
# for k in range(0,len(NO)):
#     if i[k] not in dic:
#         dic[i[k]] = [[i[k],istart[k],iend[k],MSE_FinalFunc[k],MSEtrenda[k],MSEtrendb[k],MSEtrendc[k],MSEtrendd[k],MSEtrende[k],a[k],b[k]]]
#     else:
#         dic[i[k]].append([i[k],istart[k],iend[k],MSE_FinalFunc[k],MSEtrenda[k],MSEtrendb[k],MSEtrendc[k],MSEtrendd[k],MSEtrende[k],a[k],b[k]])
#
#
# fn = open(OUT_PUT,'w')
# for t in range(2,5807):
#     for m in range(0,len(dic[t])):
#         for each in dic[t][m]:
#             fn.write(str(each))
#             fn.write(',')
#         fn.write('\n')
# fn.close()
################################################################################################
# import pandas as pd
# data_input = 'D:/python_code/NEW_step/try_paper/revision_data/result/WTI/al+~.csv'
# data_output = 'D:/python_code/NEW_step/try_paper/revision_data/result/WTI/al+++~.csv'
# data = pd.DataFrame(pd.read_csv(data_input))
# print(data.columns)
# # #21~5023  1805722
# #
# a = data["2"]
# b = data["0"]
# c = data["4"]
# d = data["Poly4"]
# e = data["0.1000000000000006"]
# f = data["-0.8333333333333375"]
# g = data["2.1500000000000075"]
# h = data["-1.4166666666666696"]
# i = data["25.55"]
# j = data["2.1"]
# k = data["2.2"]
# fn = open(data_output,'w')
#
# for t in range(0,len(a)):
#     if int(a[t])>=21 and int(a[t])<=5786:
#         fn.write(str(a[t]))
#         fn.write(',')
#         fn.write(str(b[t]))
#         fn.write(',')
#         fn.write(str(c[t]))
#         fn.write(',')
#         fn.write(str(d[t]))
#         fn.write(',')
#         fn.write(str(e[t]))
#         fn.write(',')
#         fn.write(str(f[t]))
#         fn.write(',')
#         fn.write(str(g[t]))
#         fn.write(',')
#         fn.write(str(h[t]))
#         fn.write(',')
#         fn.write(str(i[t]))
#         fn.write(',')
#         fn.write(str(j[t]))
#         fn.write(',')
#         fn.write(str(k[t]))
#         fn.write('\n')
#         print(len(a)-t)
# print('ok')
# fn.close()

# from collections import Counter
# import pandas as pd
# data_input = 'D:/python_code/NEW_step/try_paper/mycode/PAPER/wti/ab_sta.xlsx'
# data = pd.read_excel(data_input)
# aa = data["a"]
# bb = data["b"]
# print(Counter(aa))
# print(Counter(bb))

#########################################################################################################
# 样本统计特性

import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts

# data_input = 'D:/python_code/NEW_step/try_paper/revision_data/result/oilsta.xlsx'
# data = pd.read_excel(data_input)
# # Brentprice = np.array(data['Brentprice'])
# WTIprice = np.array(data['WTIprice'])
#
#
# Mean = np.mean(WTIprice)
# Std = np.std(WTIprice)
# Range = np.max(WTIprice)-np.min(WTIprice)
# s = pd.Series(WTIprice)
# result = ts.adfuller(WTIprice, 1)
# print(Mean)
# print(Std)
# print(Range)
# print(s.skew())
# print(s.kurt())
# print(result)

# #### 统计ab个数
# from collections import Counter
# import pandas as pd
#
# datainput = 'D:/python_code/NEW_step/try_paper/revision_data/result/absta.xlsx'
# data = pd.read_excel(datainput)
# # Brenta = data['BrentA']
# # Brentb = data['BrentB']
# WTIa = data['a']
# WTIb = data['b']
#
# print(Counter(WTIa))
# print(Counter(WTIb))
# # print(Counter(WTIa))
# # print(Counter(WTIb))

# # #################################################################################
# import datetime
# import warnings
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import time
#
# warnings.filterwarnings("ignore")
#
#
# start = datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))
# #
# #
# path = 'D:/python_code/NEW_step/try_paper/revision_data/result/BRENT/al+++~.csv'
# oil_price_path ='D:/python_code/NEW_step/try_paper/revision_data/result/brentpricepropro.csv'    # 待修改
#
# trend_df = pd.read_csv(path,header=None)
# price_df = pd.read_csv(oil_price_path,header=None)
#
# print(len(trend_df))
# print(len(price_df))
#
# # trenda = np.array(trend_df["MSEtrenda"])
# # trendb = np.array(trend_df["MSEtrendb"])  #88128
# # trendc = np.array(trend_df['MSEtrendc'])
# # trendd = np.array(trend_df["MSEtrendd"])
# # trende = np.array(trend_df["MSEtrende"])
# # AA = np.array(trend_df["a"])
# # BB = np.array(trend_df['b'])
#
# trenda = np.array(trend_df.iloc[:,4])
# trendb = np.array(trend_df.iloc[:,5])  #88128
# trendc = np.array(trend_df.iloc[:,6])
# trendd = np.array(trend_df.iloc[:,7])
# trende = np.array(trend_df.iloc[:,8])
# AA = np.array(trend_df.iloc[:,9])
# BB = np.array(trend_df.iloc[:,10])
#
#
#
#
# def select_trend(t):
#     '''
#     选择第t个的最优趋势
#     :param t: 第t个
#     :return: 最优趋势的a,b，同时返回最优油价
#     '''
#
#     oil_price = []
#
#     for m in range(0,361):
#
#         a = AA[t+m]
#         b = BB[t+m]
#         X = a+b
#         temp_result = trenda[t + m] * np.power(X, 4) + trendb[t + m] * np.power(X, 3) + trendc[t + m] * np.power(X, 2) + \
#                       trendd[t + m] * (X) + trende[t + m]
#         oil_price.append(temp_result)
#
#     true_price = np.array(price_df.loc[t:t+360, 1])
#
#     evaluation = list(abs(true_price-oil_price))
#     best_relative_index = evaluation.index(min(evaluation))
#     best_index = t-best_relative_index+1
#     best_price = oil_price[evaluation.index(min(evaluation))]
#     return best_index, best_relative_index, best_price
#
#
#
# def main():
#     best_trenda = np.zeros(len(trenda)//361)
#     best_trendb = np.zeros(len(trendb)//361)
#     best_trendc = np.zeros(len(trendc)//361)
#     best_trendd = np.zeros(len(trendd)//361)
#     best_trende = np.zeros(len(trende)//361)
#     Best_price = np.zeros(len(trende)//361)
#     best_a = np.zeros(len(AA)//361)
#     best_b = np.zeros(len(BB)//361)
#     relative_index = []
#     cnt = 0
#     ttt = 1
#     for i in range(0,len(trenda),361):
#
#         best_index, best_relative_index, best_price = select_trend(i)
#         best_trenda[cnt] = trenda[best_index]
#         best_trendb[cnt] = trendb[best_index]
#         best_trendc[cnt] = trendc[best_index]
#         best_trendd[cnt] = trendd[best_index]
#         best_trende[cnt] = trende[best_index]
#         Best_price[cnt] = best_price
#         best_a[cnt] = AA[best_index]
#         best_b[cnt] = BB[best_index]
#         relative_index.append(best_relative_index)
#         x = best_relative_index
#         cnt += 1
#         ttt += 1
#         # if ttt == 5003:
#         #     break
#
#     result = pd.DataFrame({
#         'trenda': best_trenda,
#         'trendb': best_trendb,
#         'trendc': best_trendc,
#         'trendd': best_trendd,
#         'trende': best_trende,
#         'best_price':Best_price,
#         'relative_index': relative_index,
#         'a':best_a,
#         'b':best_b
#     })
#     return result
#
#
#
#
# if __name__ == '__main__':
#     test_df = main()
#     selected_path = 'D:/python_code/NEW_step/try_paper/revision_data/result/BRENT'
#     test_df.to_csv(selected_path+'/window_best_trend-'+str(AA)+str(BB)+'-'+str(start)[0:10]+'(3).csv')
#


#####################################################################################
# # KRR
#
# import pandas as pd
# import numpy as np
# from statsmodels.tsa.arima_model import ARIMA
# import math
# import matplotlib.pyplot as plt
#
#
#
# oil_price_path = 'D:/python_code/NEW_step/try_paper/mycode/PAPER/wti/train.xlsx'
# data=pd.read_excel(oil_price_path,sheet_name='Sheet1')
# price = data["Price"]
# cnt = len(price)-50
# error1 = []
# error2 = []
# pre = []
# pre2 = []
#
# for t in range(0,len(price)-50):
#     train = price[t:50+t]
#     test_y = price[50+t]
#     model = ARIMA(train, order=(0,1,1))
#     model_fit = model.fit(disp=0)
#     y_kr = model_fit.forecast()[0]
#     print(y_kr)
#     print(price[50+t])
#     error1.append(abs(test_y-y_kr))
#     error2.append(pow((test_y-y_kr),2))
#     pre.append(y_kr)
#     pre2.append(test_y)
# plt.plot(pre)
# plt.plot(pre2)
# plt.show()
# MSE = sum(error2)/cnt
# MAE = sum(error1)/cnt
# MAPE = sum(error1/test_y)/cnt
# SSE = sum(error2)
# RMSE = math.sqrt(MSE)
# print("训练数据点个数为:",len(train))
# print("ARIMA的" + "MSE=", MSE, "MAE=", MAE, 'MAPE=', MAPE, 'SSE=', SSE, 'RMSE=', RMSE)

# #
# import pandas as pd
# import numpy as np
# from sklearn.svm import SVR
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import learning_curve
# from sklearn.kernel_ridge import KernelRidge
# import time
# import matplotlib.pyplot as plt
# import math
#
# oil_price_path = 'D:/python_code/NEW_step/try_paper/mycode/PAPER/wti/train.xlsx'
# data=pd.read_excel(oil_price_path,sheet_name='Sheet1')
# price = data["Price"]
# cnt = len(price)-50
# error1 = []
# error2 = []
# pre = []
# pre2 = []
#
# for t in range(0,len(price)-50-1):
#     try:
#         train_x = np.array(price[t:50+t]).reshape(-1,1)
#         train_y = np.array(price[t+1:50+t+1]).reshape(-1,1)
#         test_x = np.array(price[50+t]).reshape(-1,1)
#         test_y = price[50+t+1]
#
#         svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
#                                param_grid={"C": [1e0, 1e1, 1e2, 1e3],
#                                            "gamma": np.logspace(-2, 2, 5)})
#
#         svr.fit(train_x,train_y.ravel())
#         y_svr = svr.predict(test_x)
#
#         # kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
#         #                     param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)})
#         # kr.fit(train_x,train_y.ravel())
#         # y_kr = kr.predict(test_x)
#
#         print(test_y)
#         print(y_svr)
#         error1.append(abs(test_y-y_svr))
#         error2.append(pow((test_y-y_svr),2))
#         pre.append(y_svr)
#         pre2.append(test_y)
#     except Exception as e:
#         print(e)
# plt.plot(pre)
# plt.plot(pre2)
# plt.show()
# MSE = sum(error2)/cnt
# MAE = sum(error1)/cnt
# MAPE = sum(error1/test_y)/cnt
# SSE = sum(error2)
# RMSE = math.sqrt(MSE)
# print("训练数据点个数为:",len(train_x))
# print("KRR的" + "MSE=", MSE, "MAE=", MAE, 'MAPE=', MAPE, 'SSE=', SSE, 'RMSE=', RMSE)



#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import preprocessing
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from sklearn.metrics import mean_squared_error
# import math
#
# oil_price_path = 'D:/python_code/NEW_step/try_paper/mycode/PAPER/wti/train.xlsx'
# data=pd.read_excel(oil_price_path,sheet_name='Sheet1')
# price = data["Price"]
# cnt = len(price)-50
# error1 = []
# error2 = []
# pre = []
# pre2 = []
#
# dim = 1
# # 建立网络(adam)
# hiddennum = 12
# batch_size = 20
# model = Sequential()
# model.add(LSTM(hiddennum, return_sequences=True, input_dim=dim))
# model.add(LSTM(hiddennum))
# model.add(Dense(1, activation='linear'))
# model.compile(optimizer='adam', loss='mse')
# for t in range(0,len(price)-50-1):
#     try:
#         train_x = np.array(price[t:50+t]).reshape(50,1,1)
#         train_y = np.array(price[t+1:50+t+1]).reshape(-1,1)
#         test_x = np.array(price[50+t]).reshape(1,1,1)
#         test_y = price[50+t+1]
#
#
#
#         model.fit(train_x,train_y,batch_size=batch_size,epochs=500,verbose=0)
#         predict =model.predict(test_x,batch_size=20)
#
#         print(test_y)
#         print(predict)
#         error1.append(abs(test_y-predict))
#         error2.append(pow((test_y-predict),2))
#         pre.append(predict)
#         pre2.append(test_y)
#     except Exception as e:
#         print(e)
#
# MSE = sum(error2)/cnt
# MAE = sum(error1)/cnt
# MAPE = sum(error1/test_y)/cnt
# SSE = sum(error2)
# RMSE = math.sqrt(MSE)
# print("训练数据点个数为:",len(train_x))
# print("LSTM的" + "MSE=", MSE, "MAE=", MAE, 'MAPE=', MAPE, 'SSE=', SSE, 'RMSE=', RMSE)
################################################################################################

# import pandas as pd
# import numpy as np
#
# #如果预测值比前一天值要大于10% 那么n减少
# INPUT_PATH = 'D:/python_code/NEW_step/try_paper/mycode/PAPER/chaotiao.xlsx'
# OUT_PUT  = 'D:/python_code/NEW_step/try_paper/mycode/PAPER/chaotiao_NNN.csv'
# data = np.array(pd.read_excel(INPUT_PATH))
#
# n_best = []
# price_best = []
# for i in range(1,len(data)):
#     price = data[i-1][2]
#     index = int(data[i][0])
#
#     if index > data[i][1]:
#         while abs(data[i][2]+data[i][index+1]-price)>0.1*price :
#             index = index - 1
#             if index <=0 :
#                 index = 2
#                 break
#         price_best.append(data[i][index+1])
#         n_best.append(index)
#     else:
#         price_best.append(data[i][index+1])
#         n_best.append(index)
#
# fn = open(OUT_PUT,'w')
# for i in range(0,len(n_best)):
#     fn.write(str(n_best[i]))
#     fn.write(',')
#     fn.write(str(price_best[i]))
#     fn.write('\n')
#
# fn.close()
# print('ok')





# ################################################################
# #2~50阶 存储一个不加入abs的和一个加入abs的 后者筛选突跃点
#
# import pandas as pd
# import numpy as np
# import math
# import statsmodels.api as sm
#
# path = 'D:/python_code/NEW_step/try_paper/revision_data/result/BRENT/Brent_trainH.xlsx'
# out = 'D:/python_code/NEW_step/try_paper/revision_data/result/BRENT/Brent_mse.csv'
# df = pd.read_excel(path,header=0)
#
#
# price_df = df["price"]
# OIL_PRICE = []
#
#
#
# trenda = np.array(df.loc[:, 'trenda'].astype('float64'))
# trendb = np.array(df.loc[:, 'trendb'].astype('float64'))
# trendc = np.array(df.loc[:, 'trendc'].astype('float64'))
# trendd = np.array(df.loc[:, 'trendd'].astype('float64'))
# trende = np.array(df.loc[:, 'trende'].astype('float64'))
# AA = np.array(df["a"])
# BB = np.array(df['b'])
# XX = []
#
#
#
# for i in range(0,len(AA)):
#     XX.append(AA[i]+BB[i])
#
#
# OIL = []
# PRE = []
# True_price = []
# SUM = 0
# RM = []
#
# for t in range(15,16):    #这里面的t就是趋势阈值H
#
#     for i in range(50,len(XX)):
#
#         SUM = 0  # MSE
#         SUM2 = 0  # MAE
#         SUM3 = 0  # MAPE
#         SUM4 = 0  # SSE
#         SUM5 = 0  # RMS
#         temp = []
#         for t in range(2,52):   #滞后阶
#             ya = []
#             yb = []
#             yc = []
#             yd = []
#             ye = []
#             x = []
#
#             for y in range(1, t):
#                 x.append(XX[i - y])
#                 ya.append(trenda[i - y])
#                 yb.append(trendb[i - y])
#                 yc.append(trendc[i - y])
#                 yd.append(trendd[i - y])
#                 ye.append(trende[i - y])
#
#             print("第"+str(i)+"个数据点："+"滞后"+str(y)+"阶的结果如下：")
#             x = sm.add_constant(x)
#             resulta = sm.OLS(ya, x).fit().params
#             resultb = sm.OLS(yb, x).fit().params
#             resultc = sm.OLS(yc, x).fit().params
#             resultd = sm.OLS(yd, x).fit().params
#             resulte = sm.OLS(ye, x).fit().params
#
#             try:
#                 ka = resulta[1] * XX[i - 1] + resulta[0]
#                 kb = resultb[1] * XX[i - 1] + resultb[0]
#                 kc = resultc[1] * XX[i - 1] + resultc[0]
#                 kd = resultd[1] * XX[i - 1] + resultd[0]
#                 ke = resulte[1] * XX[i - 1] + resulte[0]
#             except:
#                 ka = resulta[0] * XX[i - 1]
#                 kb = resultb[0] * XX[i - 1]
#                 kc = resultc[0] * XX[i - 1]
#                 kd = resultd[0] * XX[i - 1]
#                 ke = resulte[0] * XX[i - 1]
#
#             price = ka * pow(XX[i - 1] + 1, 4) + kb * pow(XX[i - 1] + 1, 3) + kc * pow(XX[i - 1] + 1, 2) + \
#                         kd * pow(XX[i - 1] + 1, 1) + ke  # 预测
#
#             oil_price = trenda[i]*pow(XX[i],4)+trendb[i]*pow(XX[i],3)+trendc[i]*pow(XX[i],2)+trendd[i]*pow(XX[i],1)+trende[i]  #拟合 趋势向量还原成石油价格的拟合值
#             price = ka*pow(XX[i-1]+1,4) + kb*pow(XX[i-1]+1,3) +kc*pow(XX[i-1]+1,2)+kd*pow(XX[i-1]+1,1)+ke  #预测
#             Err = abs(price-price_df[i])   #预测值-真实值
#             error = pow(price_df[i]-price,2)
#             error2 = abs(price_df[i]-price)
#             error3 = abs((price_df[i]-price)/price_df[i])
#             error4 = pow(price_df[i]-price,2)
#             SUM  = SUM + error
#             SUM2 = SUM2 + error2
#             SUM3 = SUM3 + error3
#             SUM4 = SUM4 + error4
#
#             True_price.append(price_df[i])
#             PRE.append(price)
#             OIL.append(oil_price)
#
#             MSE = SUM
#             MAE = SUM2
#             MAPE = SUM3
#             SSE = SUM4
#             RMSE = math.sqrt(MSE)
#
#             temp.append(Err)
#             # print("MSE=",MSE,"MAE=",MAE,'MAPE=',MAPE,'SSE=',SSE,'RMSE=',RMSE,'真实值:',price_df[i],'预测值：',price,'误差：',Err)
#             print('真实值:',price_df[i],'预测值：',price,'误差：',Err)
#
#         RM.append(temp)
# print(RM)
# fn = open(out,'w')
#
# for each in RM:
#     fn.write(str(each))
#     fn.write('\n')
#
# fn.close()

####################################################################################v
# #筛选出最小的n
#
# import pandas as pd
# import numpy as np
# from collections import Counter
#
#
#
# input = "D:/python_code/NEW_step/try_paper/revision_data/result/WTI/WTI_mse.csv"
# out = 'D:/python_code/NEW_step/try_paper/revision_data/result/WTI/MIN_N.csv'
# data = np.array(pd.read_csv(input))
#
# data1 = []
# for each in data:
#     data1.append(abs(each))
# MIN_n = []
# MIN = []
# for i in range(0,len(data)):
#     for j in range(0,len(data[i])-1):
#         if (abs(data[i][j+1]) - abs(data[i][j]) ) >= abs(data[i][j]):
#             MIN_n.append(j+2)
#             MIN.append(data[i][j])
#             break
#         if j == len(data[i])-2:
#             n = list(data1[i]).index(min(data1[i]))
#             MIN_n.append(n)
#             MIN.append(data[i][n])
# print(len(data))
# print(len(MIN_n))
# print(MIN_n)
# print(Counter(MIN_n))
# print(len(Counter(MIN_n)))
# #
# fn = open(out,'w')
# for i in range(0,len(MIN_n)):
#    fn.write(str(MIN[i]))
#    fn.write(',')
#    fn.write(str(MIN_n[i]))
#    fn.write('\n')
# fn.close()

####################################################
#预测H

# import pandas as pd
# import numpy as np
# from statsmodels.tsa.arima_model import ARMA
# from sklearn.svm import SVR
# from sklearn.model_selection import GridSearchCV
# import warnings
# from sklearn.kernel_ridge import KernelRidge
# import math
#
# warnings.filterwarnings('ignore')
#
# INPUT_PATH = 'D:/python_code/NEW_step/try_paper/mycode/PAPER/train_n.xlsx'
# OUT_PUT = 'D:/python_code/NEW_step/try_paper/mycode/PAPER/predict_n.csv'
#
# data = np.array(pd.read_excel(INPUT_PATH))
#
# W1 = []
# W2 = []
#
# for length in range(50, 51):
#     SUM = []
#     SUM1 = []
#     SUM2 = []
#     Y_test = []
#     for i in range(0, len(data) - length - 1):
#         x_train = np.array(data[i:length + i]).reshape(-1, 1)  # 滚动预测
#         y_train = data[1 + i:length + i + 1]
#         x_test = np.array(data[length + i]).reshape(-1, 1)
#         y_test = data[length + i + 1]
#
#         svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=2,
#                            param_grid={"C": [1e0, 1e1, 1e2, 1e3],
#                                        "gamma": np.logspace(-2, 2, 5)})
#         svr.fit(x_train, y_train)
#
#         y_predict = [round(svr.predict(x_test)[0])]
#         W1.append(y_predict)  # 预测
#         W2.append(y_test)  # 实际
#         print(y_predict)
#         print(y_test)
#
# fn = open(OUT_PUT, 'w')
# for i in range(0, len(W1)):
#     fn.write(str(W1[i][0]))
#     fn.write(',')
#     fn.write(str(W2[i][0]))
#     fn.write('\n')
# print('ok')

#########################33
## MSPE mean squared prediction error

####
# AR模型
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
