'''
By:shenqiti
2019/9/4

最烂的代码  最糟糕的状态  最好的经历.
警戒自己，要有理论支撑再去实验。

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



# path = 'D:/python_code/NEW_step/try_paper/mycode/new/media.xlsx'
# oil_price_path = 'D:/python_code/NEW_step/try_paper/mycode/new/media_oil_price.xlsx'
#
# df = pd.read_excel(path,header=0)
# price_df = pd.read_excel(oil_price_path,header=None,sheet_name='Sheet1')
#
#
# AA = np.array(df["a"])
# BB = np.array(df['b'])

# cnt = 1
# #确定H    趋势阈值  a=H-b
# for H in range(1,2):
#     SUM = 0  #MSE
#     SUM2 = 0   #MAE
#     SUM3 = 0  #MAPE
#     SUM4 = 0   #SSE
#     SUM5 = 0  #RMSE
#
#     for i in range(len(AA)-cnt-32,len(AA)-32):
#         X = [0,1]
#         Ya = []
#
#         Ya.append(AA[i-2])
#         Ya.append(AA[i -1])
#
#
#         resultA = sm.OLS(Ya,X ).fit().params
#         KA = resultA[0]*1
#         print(KA)
    #     error = pow(H-KA,2)
    #     error2 = abs(H-KA)
    #     error3 = abs((H-KA)/H)
    #     error4 = pow(H-KA,2)
    #     SUM  = SUM + error
    #     SUM2 = SUM2 + error2
    #     SUM3 = SUM3 + error3
    #     SUM4 = SUM4 + error4
    #
    #
    # MSE = SUM / cnt
    # MAE = SUM2 / cnt
    # MAPE = SUM3 /cnt
    # SSE = SUM4
    # RMSE = math.sqrt(MSE)
    #
    # print("H="+str(H)+"MSE=",MSE,"MAE=",MAE,'MAPE=',MAPE,'SSE=',SSE,'RMSE=',RMSE)
    #


INPUT_PATH = 'D:/python_code/NEW_step/try_paper/mycode/new/new2/window_best_trend-[ 2  2  2 ... 20 20 20][ 2  3  4 ... 18 19 20]-2019-08-31(3).csv'

data = pd.read_csv(INPUT_PATH)
trenda = np.array(data["trenda"])
trendb = np.array(data["trendb"])
trendc = np.array(data["trendc"])
trendd = np.array(data["trendd"])
trende = np.array(data["trende"])
A = np.array(data["a"])
B = np.array(data["b"])


plt.subplot(5,1,1)
plt.plot(trenda)
plt.title("Trend_A")

plt.subplot(5,1,2)
plt.plot(trendb)
plt.title("Trend_B")

plt.subplot(5,1,3)
plt.plot(trendc)
plt.title("Trend_C")

plt.subplot(5,1,4)
plt.plot(trendd)
plt.title("Trend_D")
plt.subplot(5,1,5)
plt.plot(trende)
plt.title("Trend_E")

plt.show()

# # plt.show()
# plt.plot(trendb)
# # plt.title("Trend_B")
# # plt.show()
# plt.plot(trendc)
# # plt.title("Trend_C")
# # plt.show()
# plt.plot(trendd)
# # plt.title("Trend_D")
# # plt.show()
# plt.plot(trende)
# # plt.title("Trend_E")
# plt.show()


a = Counter(A)
b = Counter(B)


xx = list(a.keys())
yy = list(a.values())
xxx = list(b.keys())
yyy = list(b.values())
print(xx)
print(yy)
print(xxx)
print(yyy)
print(b)


total_width, n = 0.8, 2
width = total_width / n
plt.bar(xx, yy, width=width, label='a')
for i in range(len(xx)):
    xxx[i] = xx[i] + width
plt.bar(xxx, yyy, width=width, label='b')
plt.legend()
plt.title("A AND B")
plt.xlabel("length")
plt.ylabel("Count")
plt.show()

# INPUT_PATH = 'D:/python_code/NEW_step/try_paper/mycode/PAPER/all_data.xlsx'
# data = pd.read_excel(INPUT_PATH,sheet_name='Time_trend')
# price = np.array(data["True Price"])
# AA = np.array(data["a"])
# BB = np.array(data['b'])
#
#
# draw_data = price[2000:2100]
# draw_A = [i for i in range(40,50)]
# draw_B = [i for i in range(40,50)]
# A_x = [50-AA[2050]]*10
# B_x = [50+BB[2050]]*10
# plt.plot(draw_data)
# plt.scatter(50,price[2050],color ='orange')
# plt.plot(A_x,draw_A,color='orange')
# plt.plot(B_x,draw_B,color='orange')
#
#
#
# draw_data = price[2000:2100]
# draw_A = [i for i in range(50,60)]
# draw_B = [i for i in range(50,60)]
# A_x = [70-AA[2070]]*10
# B_x = [70+BB[2070]]*10
# plt.scatter(70,price[2070],color ='green')
# plt.plot(A_x,draw_A,color='green')
# plt.plot(B_x,draw_B,color='green')
#
#
# draw_A = [i for i in range(45,55)]
# draw_B = [i for i in range(45,55)]
# A_x = [60-AA[2060]]*10
# B_x = [60+BB[2060]]*10
# plt.scatter(60,price[2060],color ='yellow')
# plt.plot(A_x,draw_A,color='yellow')
# plt.plot(B_x,draw_B,color='yellow')
#
#
#
# draw_A = [i for i in range(40,50)]
# draw_B = [i for i in range(40,50)]
# A_x = [30-AA[2030]]*10
# B_x = [30+BB[2030]]*10
# plt.scatter(30,price[2030],color ='red')
# plt.plot(A_x,draw_A,color='red')
# plt.plot(B_x,draw_B,color='red')
#
#
# draw_A = [i for i in range(50,60)]
# draw_B = [i for i in range(50,60)]
# A_x = [82-AA[2082]]*10
# B_x = [82+BB[2082]]*10
# plt.scatter(82,price[2082],color ='purple')
# plt.plot(A_x,draw_A,color='purple')
# plt.plot(B_x,draw_B,color='purple')
#
# plt.title("Time-varying window of sample points")
# plt.ylabel("Price")
# plt.xlabel("Sample")
# plt.show()

