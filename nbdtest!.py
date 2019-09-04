
'''
By:shenqiti
2019/9/4

曾经论文走过的坑 是我最宝贵的经历.

'''

import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
from collections import Counter
import statsmodels.api as sm
import statsmodels.stats.diagnostic
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import time
import matplotlib.pyplot as plt
import math
warnings.filterwarnings("ignore")




# #——————————————————————————————————————————————————————————————



# #
# # # # 分段预测
# # #
# # #
# path = 'D:/python_code/NEW_step/try_paper/mycode/new/new2/window_best_trend-[ 2  2  2 ... 20 20 20][ 2  3  4 ... 18 19 20]-2019-08-31(3).csv'
# # oil_price_path = 'D:/python_code/NEW_step/try_paper/mycode/new/media_oil_price.xlsx'
#
# df = pd.read_csv(path,header=0)
# # price_df = pd.read_excel(oil_price_path,header=None,sheet_name='Sheet1')
#
# price_df = df["price"]
#
# OIL_PRICE = []
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
# cnt = 1
# for t in range(1,30):    #这里面的t就是趋势阈值H
#     SUM = 0  #MSE
#     SUM2 = 0   #MAE
#     SUM3 = 0  #MAPE
#     SUM4 = 0   #SSE
#     SUM5 = 0  #RMS
#     for i in range(len(XX)-cnt-532,len(XX)-532):
#         ya = []
#         yb = []
#         yc = []
#         yd = []
#         ye = []
#         a = []
#         b = []
#         x = []
#         if AA[i-1]<t:
#             x.append(XX[i - 1])
#
#
#             ya.append(trenda[i - 1])
#
#
#             yb.append(trendb[i - 1])
#
#             yc.append(trendc[i - 1])
#
#
#             yd.append(trendd[i - 1])
#
#
#             ye.append(trende[i - 1])
#             x = sm.add_constant(x)
#
#             resulta = sm.OLS(ya, x).fit().params
#             resultb = sm.OLS(yb, x).fit().params
#             resultc = sm.OLS(yc, x).fit().params
#             resultd = sm.OLS(yd, x).fit().params
#             resulte = sm.OLS(ye, x).fit().params
#             try:
#                 ka = resulta[1] * XX[i - 1] + resulta[0]
#                 kb = resultb[1] * XX[i - 1] + resultb[0]
#                 kc = resultc[1] * XX[i - 1] + resultc[0]
#                 kd = resultd[1] * XX[i - 1] + resultd[0]
#                 ke = resulte[1] * XX[i - 1] + resulte[0]
#             except:
#
#                 ka = resulta[0] * XX[i - 1]
#                 kb = resultb[0] * XX[i - 1]
#                 kc = resultc[0] * XX[i - 1]
#                 kd = resultd[0] * XX[i - 1]
#                 ke = resulte[0] * XX[i - 1]
#
#             price = ka * pow(XX[i - 1] + 1, 4) + kb * pow(XX[i - 1] + 1, 3) + kc * pow(XX[i - 1] + 1, 2) + kd * pow(
#                 XX[i - 1] + 1, 1) + ke  # 预测
#
#         elif AA[i-1]>=t:
#             x.append(XX[i - 1])
#             x.append(XX[i - 2])
#
#             ya.append(trenda[i - 1])
#             ya.append(trenda[i - 2])
#
#             yb.append(trendb[i - 1])
#             yb.append(trendb[i - 2])
#
#             yc.append(trendc[i - 1])
#             yc.append(trendc[i - 2])
#
#             yd.append(trendd[i - 1])
#             yd.append(trendd[i - 2])
#
#             ye.append(trende[i - 1])
#             ye.append(trende[i - 2])
#
#             x = sm.add_constant(x)
#             resulta = sm.OLS(ya, x).fit().params
#             resultb = sm.OLS(yb, x).fit().params
#             resultc = sm.OLS(yc, x).fit().params
#             resultd = sm.OLS(yd, x).fit().params
#             resulte = sm.OLS(ye, x).fit().params
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
#             price = ka*pow(XX[i-1]+1,4) + kb*pow(XX[i-1]+1,3) +kc*pow(XX[i-1]+1,2)+kd*pow(XX[i-1]+1,1)+ke   #预测
#
#
#
#
#         error = pow(price_df[i]-price,2)
#         error2 = abs(price_df[i]-price)
#         error3 = abs((price_df[i]-price)/price_df[i])
#         error4 = pow(price_df[i]-price,2)
#         SUM  = SUM + error
#         SUM2 = SUM2 + error2
#         SUM3 = SUM3 + error3
#         SUM4 = SUM4 + error4
#
#         True_price.append(price_df[i])
#         PRE.append(price)
#
#         # print('预测第',i+1,'个点的油价',',预测值为',price,'真实值为',price_df.loc[i,1],'拟合值为', oil_price,'预测误差为:',error,'XX[i-1]:',XX[i-1])
#     MSE = SUM / cnt
#     MAE = SUM2 / cnt
#     MAPE = SUM3 /cnt
#     SSE = SUM4
#     RMSE = math.sqrt(MSE)
#     print("H="+str(t)+"MSE=",MSE,"MAE=",MAE,'MAPE=',MAPE,'SSE=',SSE,'RMSE=',RMSE)

# print(len(XX))
# plt.plot(True_price)
# plt.plot(OIL)
# plt.plot(PRE)
# plt.title("Europe Brent Spot Price FOB")
# plt.legend(["True_price","Fitting_price","Predict_price"])
# plt.xlabel("Date")
# plt.ylabel("Price(Dollars per Barrel)")
# plt.show()
# #
#


#___________________________________________
#不分段预测
#
#
# #

# path = 'D:/python_code/NEW_step/try_paper/mycode/new/new2/window_best_trend-[ 2  2  2 ... 20 20 20][ 2  3  4 ... 18 19 20]-2019-08-31(3).csv'
# out = 'D:/python_code/NEW_step/try_paper/mycode/PAPER/RMSE.csv'
# df = pd.read_csv(path,header=0)
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
# cnt =1000   #样本点
# RM = []
#
# for t in range(15,16):    #这里面的t就是趋势阈值H
#
#     for i in range(len(XX)-cnt-32,len(XX)-32):
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
#             Err = abs(price_df[i]-price)
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
#





#————————————————绘制拟合图

# INPUT_PATH = 'D:/python_code/NEW_step/try_paper/mycode/new/media.xlsx'
# data = pd.read_excel(INPUT_PATH)
# true_price = np.array(data["True Price"])
# pre_price = np.array(data["Predict Price"])
# plt.plot(true_price)
# plt.plot(pre_price)
# plt.title("Fitting stage diagram")
# plt.legend(["true_price","predict_price"])
# plt.show()

#————————————————————






#
# start = datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))
#
#
# path = 'D:/python_code/NEW_step/try_paper/mycode/new/new2/all_finall3.csv'
# oil_price_path ='D:/python_code/NEW_step/try_paper/mycode/new/new2/Br_pro.csv'    # 待修改
#
# trend_df = pd.read_csv(path,header=None)
# price_df = pd.read_csv(oil_price_path,header=None)
#
# print(len(trend_df))
# print(len(price_df))

# trenda = np.array(trend_df["MSEtrenda"])
# trendb = np.array(trend_df["MSEtrendb"])  #88128
# trendc = np.array(trend_df['MSEtrendc'])
# trendd = np.array(trend_df["MSEtrendd"])
# trende = np.array(trend_df["MSEtrende"])
# AA = np.array(trend_df["a"])
# BB = np.array(trend_df['b'])
#
# trenda = np.array(trend_df.iloc[:,4])
# trendb = np.array(trend_df.iloc[:,5])  #88128
# trendc = np.array(trend_df.iloc[:,6])
# trendd = np.array(trend_df.iloc[:,7])
# trende = np.array(trend_df.iloc[:,8])
# AA = np.array(trend_df.iloc[:,9])
# BB = np.array(trend_df.iloc[:,10])




# def select_trend(t):
#     '''
#     选择第t个的最优趋势
#     :param t: 第t个
#     :return: 最优趋势的a,b，同时返回最优油价
#     '''
#
#     oil_price = []
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
#     for i in range(0,len(trenda),361):
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
#
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
#     selected_path = 'D:/python_code/NEW_step/try_paper/mycode/new/new2'
#     test_df.to_csv(selected_path+'/window_best_trend-'+str(AA)+str(BB)+'-'+str(start)[0:10]+'(3).csv')
# #
#
#
#
















# INPUT_PATH = 'D:/python_code/NEW_step/try_paper/mycode/new/Brent_pro1.xlsx'
# OUT_PUT = 'D:/python_code/NEW_step/try_paper/mycode/new/Brent_pro11.csv'
#
#
# data = pd.read_excel(INPUT_PATH)
# ID = np.array(data["id"])
# price = np.array(data["price"])
# P = []
# id = []
#
# for k in range(len(price)):
#     for i in range(0,64):
#         P.append(price[k])
#         id.append(ID[k])
#
# fn = open(OUT_PUT,'w')
#
# for i in range(0,len(P)):
#     fn.write(str(id[i]))
#     fn.write(',')
#     fn.write(str(P[i]))
#     fn.write('\n')
#
# fn.close()



#整理数据


# INPUT_PATH = 'D:/python_code/NEW_step/try_paper/mycode/new/Poly-[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20][2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]-WINParams-2019-08-31.csv'
# OUT_PUT = 'D:/python_code/NEW_step/try_paper/mycode/new/new2/all_finall3.csv'
# data = pd.read_csv(INPUT_PATH)
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
# for k in range(0,len(i)):
#     if i[k] not in dic:
#         dic[i[k]] = [[i[k],istart[k],iend[k],MSE_FinalFunc[k],MSEtrenda[k],MSEtrendb[k],MSEtrendc[k],MSEtrendd[k],MSEtrende[k],a[k],b[k]]]
#     else:
#         dic[i[k]].append([i[k],istart[k],iend[k],MSE_FinalFunc[k],MSEtrenda[k],MSEtrendb[k],MSEtrendc[k],MSEtrendd[k],MSEtrende[k],a[k],b[k]])
#
#
# fn = open(OUT_PUT,'w')
# for t in range(20,5518):    #（2，5536）
#     for m in range(0,len(dic[t])):
#         for each in dic[t][m]:
#             fn.write(str(each))
#             fn.write(',')
#         fn.write('\n')
# fn.close()




# INPUT_PATH = 'D:/python_code/NEW_step/try_paper/mycode/new/new2/(5, 10, 15, 20)-2019-08-02(3).csv'
# out_put = 'D:/python_code/NEW_step/try_paper/mycode/new/new2/window.csv'
#
# INPUT_PATH2 = 'D:/python_code/NEW_step/try_paper/mycode/new/new2/window_best_trend-[5, 10, 15, 20]-2019-08-02(3).csv'
#
#
# data = pd.read_csv(INPUT_PATH)
# wv = pd.read_csv(INPUT_PATH2)
# ta = np.array(wv["trenda"])
# tb = np.array(wv["trendb"])
# tc = np.array(wv["trendc"])
# td = np.array(wv["trendd"])
# te = np.array(wv["trende"])
#
# tab = []
# tbb = []
# tcb = []
# tdb = []
# teb = []
#
# error = data["err"]
# error = np.array(error)
# trend5 = 0
# trend10 = 0
# trend15 = 0
# trend20 = 0
# b = []
# pre_price = np.array(data["Predict Price"])
# rel_price = np.array(data["Ture Price"])
# price = []
# r_price = []
# ERROR= []
# for i in range(0,len(error),4):
#     trend5 = error[i]
#     trend10 = error[i+1]
#     trend15 = error[i+2]
#     trend20 = error[i+3]
#     temp = min(trend5,trend10,trend15,trend20)
#     ERROR.append(temp)
#     if temp == error[i]:
#         b.append("5")
#         price.append(pre_price[i])
#         r_price.append(rel_price[i])
#         tab.append(ta[i])
#         tbb.append(tb[i])
#         tcb.append(tc[i])
#         tdb.append(td[i])
#         teb.append(te[i])
#     elif temp == error[i+1]:
#         b.append("10")
#         price.append(pre_price[i+1])
#         r_price.append(rel_price[i+1])
#         tab.append(ta[i+1])
#         tbb.append(tb[i+1])
#         tcb.append(tc[i+1])
#         tdb.append(td[i+1])
#         teb.append(te[i+1])
#     elif temp == error[i+2]:
#         b.append("15")
#         price.append(pre_price[i+2])
#         r_price.append(rel_price[i+2])
#         tab.append(ta[i+2])
#         tbb.append(tb[i+2])
#         tcb.append(tc[i+2])
#         tdb.append(td[i+2])
#         teb.append(te[i+2])
#     elif temp == error[i+3]:
#         b.append("20")
#         price.append(pre_price[i+3])
#         r_price.append(rel_price[i+3])
#         tab.append(ta[i+3])
#         tbb.append(tb[i+3])
#         tcb.append(tc[i+3])
#         tdb.append(td[i+3])
#         teb.append(te[i+3])
# print(b)
# print(price)
#
# fn = open(out_put,'w')
# cnt = 0
# for i in range(len(b)):
#     fn.write(str(cnt))
#     fn.write(',')
#     fn.write(str(r_price[i]))
#     fn.write(',')
#     fn.write(str(price[i]))
#     fn.write(',')
#     fn.write(str(b[i]))
#     fn.write(',')
#     fn.write(str(tab[i]))
#     fn.write(',')
#     fn.write(str(tbb[i]))
#     fn.write(',')
#     fn.write(str(tcb[i]))
#     fn.write(',')
#     fn.write(str(tdb[i]))
#     fn.write(',')
#     fn.write(str(teb[i]))
#     fn.write('\n')
#     cnt += 1
# fn.close()








# oil_price_path ='D:/python_code/NEW_step/try_paper/mycode/new/all.csv'
# out_path = 'D:/python_code/NEW_step/try_paper/mycode/new/train.csv'
# trend_df = pd.read_excel(oil_price_path)
#
# print(trend_df)
#
# P = []
#
# a5 = np.array(trend_df["MSEtrenda5"])
# a10 = np.array(trend_df["MSEtrenda10"])
# a15 = np.array(trend_df["MSEtrenda15"])
# a20 = np.array(trend_df["MSEtrenda20"])
#
# b5 = np.array(trend_df["MSEtrendb5"])
# b10 = np.array(trend_df["MSEtrendb10"])
# b15 = np.array(trend_df["MSEtrendb15"])
# b20 = np.array(trend_df["MSEtrendb20"])
#
# c5 = np.array(trend_df["MSEtrendc5"])
# c10 = np.array(trend_df["MSEtrendc10"])
# c15 = np.array(trend_df["MSEtrendc15"])
# c20 = np.array(trend_df["MSEtrendc20"])
#
# d5 = np.array(trend_df["MSEtrendd5"])
# d10 = np.array(trend_df["MSEtrendd10"])
# d15 = np.array(trend_df["MSEtrendd15"])
# d20 = np.array(trend_df["MSEtrendd20"])
#
# e5 = np.array(trend_df["MSEtrende5"])
# e10 = np.array(trend_df["MSEtrende10"])
# e15 = np.array(trend_df["MSEtrende15"])
# e20 = np.array(trend_df["MSEtrende20"])
#
#
#
# fn = open(out_path,'w')
# for i in range(len(e20)):
#
#     fn.write(str(a5[i]))
#     fn.write(',')
#     fn.write(str(b5[i]))
#     fn.write(',')
#     fn.write(str(c5[i]))
#     fn.write(',')
#     fn.write(str(d5[i]))
#     fn.write(',')
#     fn.write(str(e5[i]))
#     fn.write('\n')
#
#     fn.write(str(a10[i]))
#     fn.write(',')
#     fn.write(str(b10[i]))
#     fn.write(',')
#     fn.write(str(c10[i]))
#     fn.write(',')
#     fn.write(str(d10[i]))
#     fn.write(',')
#     fn.write(str(e10[i]))
#     fn.write('\n')
#
#     fn.write(str(a15[i]))
#     fn.write(',')
#     fn.write(str(b15[i]))
#     fn.write(',')
#     fn.write(str(c15[i]))
#     fn.write(',')
#     fn.write(str(d15[i]))
#     fn.write(',')
#     fn.write(str(e15[i]))
#     fn.write('\n')
#
#     fn.write(str(a20[i]))
#     fn.write(',')
#     fn.write(str(b20[i]))
#     fn.write(',')
#     fn.write(str(c20[i]))
#     fn.write(',')
#     fn.write(str(d20[i]))
#     fn.write(',')
#     fn.write(str(e20[i]))
#     fn.write('\n')
#
#
# fn.close()




#
# INPUT = 'D:/python_code/NEW_step/try_paper/mycode/new/all.xlsx'
#
# data = pd.read_excel(INPUT)
# MSEtrende5 = np.array(data["MSEtrende5"])
# MSEtrende10 = np.array(data["MSEtrende10"])
# MSEtrende15 = np.array(data["MSEtrende15"])
# MSEtrende20 = np.array(data["MSEtrende20"])
# MSEtrende22 = np.array(data["MSEtrende22"])
#
# plt.plot(MSEtrende5)
# plt.plot(MSEtrende10)
# plt.plot(MSEtrende15)
# plt.plot(MSEtrende20)
#
# plt.legend(['MSEtrende5','MSEtrende10','MSEtrende15','MSEtrende20'])
#
# plt.show()


# start = datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))
# path = 'D:/python_code/NEW_step/try_paper/mycode/results/Poly-22-Params-2019-04-23.csv'
# oil_price_path ='D:/python_code/NEW_step/try_paper/data/Data/Brent-11-25.xlsm'

# path = 'D:/python_code/NEW_step/try_paper/mycode/new/all.xlsx'
# oil_price_path ='D:/python_code/NEW_step/try_paper/data/Data/Brent-11-25.xlsm'
#
# trend_df = pd.read_excel(path,header=0)
# price_df = pd.read_excel(oil_price_path,header=None)

# trenda = np.array(trend_df.loc[:, 'MSEtrenda'].astype('float64'))
# trendb = np.array(trend_df.loc[:, 'MSEtrendb'].astype('float64'))
# trendc = np.array(trend_df.loc[:, 'MSEtrendc'].astype('float64'))
# trendd = np.array(trend_df.loc[:, 'MSEtrendd'].astype('float64'))
# trende = np.array(trend_df.loc[:, 'MSEtrende'].astype('float64'))

# length = len(trend_df["MSEtrenda5"])   #其他的就减去差即可
# a5 = np.array(trend_df["MSEtrenda5"])
# a10 = np.array(trend_df["MSEtrenda10"])
# a15 = np.array(trend_df["MSEtrenda15"])
# a20 = np.array(trend_df["MSEtrenda20"])
# a22 = np.array(trend_df["MSEtrenda22"])
# trenda = np.hstack((a5,a10,a15,a20,a22))

# ########################3
# INPUT_DATA = 'D:/python_code/NEW_step/try_paper/mycode/new/new2/Br.xlsx'
# out_put = 'D:/python_code/NEW_step/try_paper/mycode/new/new2/Br_pro.csv'
#
# data= pd.read_excel(INPUT_DATA)
# id = data["id"]
# price = data["price"]
#
# fn = open(out_put,'w')
# for i in range(0,len(price)):
#     for j in range(0,361):
#         fn.write(str(id[i]))
#         fn.write(',')
#         fn.write(str(price[i]))
#         fn.write('\n')
#
# fn.close()
# print('ok')



# LR.classfy _ H
#

# INPUT_PATH = 'D:/python_code/NEW_step/try_paper/mycode/PAPER/classfyH.xlsx'
#
# data = pd.read_excel(INPUT_PATH)
# data = np.array(data)
#
#
# x, y = np.split(data, (7,), axis=1)    #根据csv文件选择x，y列
# x = x[:, :7]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)
#
# svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
#                        param_grid={"C": [1e0, 1e1, 1e2, 1e3],
#                                    "gamma": np.logspace(-2, 2, 5)})
#
#     # 训练
# svr.fit(x_train,y_train.ravel())
#
# y_svr = svr.predict(x_test)
# error1 = abs(y_test-y_svr)
# error2 = [each*each for each in error1]
# MSE = sum(error2)/len(y_test)
# MAE = sum(error1)/len(y_test)
# MAPE = sum(error1/y_test)/len(y_test)
# SSE = sum(error2)
# # RMSE = math.sqrt(MSE)
# print("训练数据点个数为:",len(x_train))
# print("SVR的" + "MSE=", MSE, "MAE=", MAE, 'MAPE=', MAPE, 'SSE=', SSE, 'RMSE=')

# #min mae
# input = "D:/python_code/NEW_step/try_paper/mycode/PAPER/H.csv"
# out = 'D:/python_code/NEW_step/try_paper/mycode/PAPER/MIN_N.csv'
# data = np.array(pd.read_csv(input))
#
#
#
# MIN_n = []
# MIN = []
# for i in range(0,len(data)):
#     for j in range(0,len(data[i])-1):
#         if (data[i][j+1] - data[i][j] ) >= data[i][j]:
#             MIN_n.append(j+2)
#             MIN.append(data[i][j])
#             break
#         if j == len(data[i])-2:
#             MIN_n.append(list(data[i]).index(min(data[i])))
#             MIN.append(min(data[i]))
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
# #
#
#


