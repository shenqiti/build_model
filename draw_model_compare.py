'''
By:shenqiti
2019/9/4

傻过吧...正规格式用excel画
'''



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


INPUT_PATH = 'D:/python_code/NEW_step/try_paper/mycode/PAPER/new_conpare.xlsx'
data = pd.read_excel(INPUT_PATH)

# n=1 所以MSE=SSE


MSE_we = np.array(data['MSE_we'])
MAE_we = np.array(data['MAE_we'])
MAPE_we = np.array(data['MAPE_we'])
xxxxxx = xxxxx = xxxx = xxx = xx = x = [i for i in range(len(MAE_we))]


MSE_SVR = np.array(data['MSE_SVR'])
MAE_SVR = np.array(data['MAE_SVR'])
MAPE_SVR = np.array(data['MAPE_SVR'])

MSE_BPNN = np.array(data['MSE_BPNN'])
MAE_BPNN = np.array(data['MAE_BPNN'])
MAPE_BPNN = np.array(data['MAPE_BPNN'])

MSE_LSTM = np.array(data['MSE_LSTM'])
MAE_LSTM = np.array(data['MAE_LSTM'])
MAPE_LSTM = np.array(data['MAPE_LSTM'])

MSE_ELM = np.array(data['MSE_ELM'])
MAE_ELM = np.array(data['MAE_ELM'])
MAPE_ELM = np.array(data['MAPE_ELM'])

MSE_KRR = np.array(data['MSE_KRR'])
MAE_KRR = np.array(data['MAE_KRR'])
MAPE_KRR = np.array(data['MAPE_KRR'])



#求11个样本值的综合指标 MSE MAE MAPE SSE MAPE

MSE1 = sum(MSE_we)/len(MSE_we)
MSE2 = sum(MSE_SVR)/len(MSE_SVR)
MSE3 = sum(MSE_BPNN)/len(MSE_BPNN)
MSE4 = sum(MSE_LSTM)/len(MSE_LSTM)
MSE5 = sum(MSE_ELM)/len(MSE_ELM)
MSE6 = sum(MSE_KRR)/len(MSE_KRR)
print(MSE1,MSE2,MSE3,MSE4,MSE5,MSE6)

MAE1 = sum(MAE_we)/len(MAE_we)
MAE2 = sum(MAE_SVR)/len(MAE_SVR)
MAE3 = sum(MAE_BPNN)/len(MAE_BPNN)
MAE4 = sum(MAE_LSTM)/len(MAE_LSTM)
MAE5 = sum(MAE_ELM)/len(MAE_ELM)
MAE6 = sum(MAE_KRR)/len(MAE_KRR)

print(MAE1,MAE2,MAE3,MAE4,MAE5,MAE6)

MAPE1 = sum(MAPE_we)/len(MAPE_we)
MAPE2 = sum(MAPE_SVR)/len(MAPE_we)
MAPE3 = sum(MAPE_BPNN)/len(MAPE_we)
MAPE4 = sum(MAPE_LSTM)/len(MAPE_we)
MAPE5 = sum(MAPE_ELM)/len(MAPE_we)
MAPE6 = sum(MAPE_KRR)/len(MAPE_we)

print(MAPE1,MAPE2,MAPE3,MAPE4,MAPE5,MAPE6)

RMSE1 = math.sqrt(sum(MSE_we)/len(MSE_we))
RMSE2 = math.sqrt(sum(MSE_SVR)/len(MSE_SVR))
RMSE3 = math.sqrt(sum(MSE_BPNN)/len(MSE_BPNN))
RMSE4 = math.sqrt(sum(MSE_LSTM)/len(MSE_LSTM))
RMSE5 = math.sqrt(sum(MSE_ELM)/len(MSE_ELM))
RMSE6 = math.sqrt(sum(MSE_KRR)/len(MSE_KRR))

print(RMSE1,RMSE2,RMSE3,RMSE4,RMSE5,RMSE6)
#_________________________________________________
# plt.subplot(2,2,1)
# total_width, n = 6, 6
# width = total_width / n
# x = xx = xxx = xxxx = xxxxx = xxxxxx = [0]
# plt.bar(x, MSE1, width=width, label='TD')
# for i in range(len(x)):
#     xx[i] = x[i] + width
# plt.bar(xx, MSE2, width=width, label='SVR')
#
# for i in range(len(xx)):
#     xxx[i] = xx[i] + width
# plt.bar(xxx, MSE3, width=width, label='BPNN')
#
# for i in range(len(xxx)):
#     xxxx[i] = xxx[i] + width
# plt.bar(xxxx, MSE4, width=width, label='LSTM')
#
# for i in range(len(xxxx)):
#     xxxxx[i] = xxxx[i] + width
# plt.bar(xxxxx, MSE5, width=width, label='ELM')
#
# for i in range(len(xxxxx)):
#     xxxxxx[i] = xxxxx[i] + width
# plt.bar(xxxxxx, MSE6, width=width, label='KRR')
#
# plt.legend()
# plt.title("MSE_11")
# plt.xlabel("Model")
# plt.ylabel("MSE")
#
#
#
# plt.subplot(2,2,2)
# total_width, n = 6, 6
# width = total_width / n
# x = xx = xxx = xxxx = xxxxx = xxxxxx = [0]
# plt.bar(x, MAE1, width=width, label='TD')
# for i in range(len(x)):
#     xx[i] = x[i] + width
# plt.bar(xx, MAE2, width=width, label='SVR')
#
# for i in range(len(xx)):
#     xxx[i] = xx[i] + width
# plt.bar(xxx, MAE3, width=width, label='BPNN')
#
# for i in range(len(xxx)):
#     xxxx[i] = xxx[i] + width
# plt.bar(xxxx, MAE4, width=width, label='LSTM')
#
# for i in range(len(xxxx)):
#     xxxxx[i] = xxxx[i] + width
# plt.bar(xxxxx, MAE5, width=width, label='ELM')
#
# for i in range(len(xxxxx)):
#     xxxxxx[i] = xxxxx[i] + width
# plt.bar(xxxxxx, MAE6, width=width, label='KRR')
#
# plt.legend()
# plt.title("MAE_11")
# plt.xlabel("Model")
# plt.ylabel("MAE")
#
# plt.subplot(2,2,3)
# total_width, n = 6, 6
# width = total_width / n
# x = xx = xxx = xxxx = xxxxx = xxxxxx = [0]
# plt.bar(x, MAPE1, width=width, label='TD')
# for i in range(len(x)):
#     xx[i] = x[i] + width
# plt.bar(xx, MAPE2, width=width, label='SVR')
#
# for i in range(len(xx)):
#     xxx[i] = xx[i] + width
# plt.bar(xxx, MAPE3, width=width, label='BPNN')
#
# for i in range(len(xxx)):
#     xxxx[i] = xxx[i] + width
# plt.bar(xxxx, MAPE4, width=width, label='LSTM')
#
# for i in range(len(xxxx)):
#     xxxxx[i] = xxxx[i] + width
# plt.bar(xxxxx, MAPE5, width=width, label='ELM')
#
# for i in range(len(xxxxx)):
#     xxxxxx[i] = xxxxx[i] + width
# plt.bar(xxxxxx, MAPE6, width=width, label='KRR')
#
# plt.legend()
# plt.title("MAPE_11")
# plt.xlabel("Model")
# plt.ylabel("MAPE(%)")
#
#
# plt.subplot(2,2,4)
# total_width, n = 6, 6
# width = total_width / n
# x = xx = xxx = xxxx = xxxxx = xxxxxx = [0]
# plt.bar(x, RMSE1, width=width, label='TD')
# for i in range(len(x)):
#     xx[i] = x[i] + width
# plt.bar(xx, RMSE2, width=width, label='SVR')
#
# for i in range(len(xx)):
#     xxx[i] = xx[i] + width
# plt.bar(xxx, RMSE3, width=width, label='BPNN')
#
# for i in range(len(xxx)):
#     xxxx[i] = xxx[i] + width
# plt.bar(xxxx, RMSE4, width=width, label='LSTM')
#
# for i in range(len(xxxx)):
#     xxxxx[i] = xxxx[i] + width
# plt.bar(xxxxx, RMSE5, width=width, label='ELM')
#
# for i in range(len(xxxxx)):
#     xxxxxx[i] = xxxxx[i] + width
# plt.bar(xxxxxx, RMSE6, width=width, label='KRR')
#
# plt.legend()
# plt.title("RMSE_11")
# plt.xlabel("Model")
# plt.ylabel("RMSE")
#
# plt.show()
#_________________________________________________
plt.subplot(2,2,1)
name_list = [1,2,3,4,5,6,7,8,9,10,11]
total_width, n = 0.6, 6
width = total_width / n
plt.bar(x, MSE_we, width=width, label='TD')
for i in range(len(x)):
    xx[i] = x[i] + width
plt.bar(xx, MSE_SVR, width=width, label='SVR')

for i in range(len(xx)):
    xxx[i] = xx[i] + width
plt.bar(xxx, MSE_BPNN, width=width, label='BPNN')

for i in range(len(xxx)):
    xxxx[i] = xxx[i] + width
plt.bar(xxxx, MSE_LSTM, width=width, label='LSTM')

for i in range(len(xxxx)):
    xxxxx[i] = xxxx[i] + width
plt.bar(xxxxx, MSE_ELM, width=width, label='ELM')

for i in range(len(xxxxx)):
    xxxxxx[i] = xxxxx[i] + width
plt.bar(xxxxxx, MSE_KRR, width=width, label='KRR',tick_label = name_list)



plt.legend()
plt.title("SSE_1")
plt.xlabel("point")
plt.ylabel("SSE")


plt.subplot(2,2,2)
name_list = [1,2,3,4,5,6,7,8,9,10,11]
total_width, n = 0.6, 6
width = total_width / n
plt.bar(x, MAE_we, width=width, label='TD')
for i in range(len(x)):
    xx[i] = x[i] + width
plt.bar(xx, MAE_SVR, width=width, label='SVR')

for i in range(len(xx)):
    xxx[i] = xx[i] + width
plt.bar(xxx, MAE_BPNN, width=width, label='BPNN')

for i in range(len(xxx)):
    xxxx[i] = xxx[i] + width
plt.bar(xxxx, MAE_LSTM, width=width, label='LSTM')

for i in range(len(xxxx)):
    xxxxx[i] = xxxx[i] + width
plt.bar(xxxxx, MAE_ELM, width=width, label='ELM')

for i in range(len(xxxxx)):
    xxxxxx[i] = xxxxx[i] + width
plt.bar(xxxxxx, MAE_KRR, width=width, label='KRR',tick_label = name_list)



plt.legend()
plt.title("MAE_1")
plt.xlabel("point")
plt.ylabel("MAE")


plt.subplot(2,1,2)
name_list = [1,2,3,4,5,6,7,8,9,10,11]
total_width, n = 0.6, 6
width = total_width / n
plt.bar(x, MAPE_we, width=width, label='TD       ')
for i in range(len(x)):
    xx[i] = x[i] + width
plt.bar(xx, MAPE_SVR, width=width, label='SVR')

for i in range(len(xx)):
    xxx[i] = xx[i] + width
plt.bar(xxx, MAPE_BPNN, width=width, label='BPNN')

for i in range(len(xxx)):
    xxxx[i] = xxx[i] + width
plt.bar(xxxx, MAPE_LSTM, width=width, label='LSTM')

for i in range(len(xxxx)):
    xxxxx[i] = xxxx[i] + width
plt.bar(xxxxx, MAPE_ELM, width=width, label='ELM')

for i in range(len(xxxxx)):
    xxxxxx[i] = xxxxx[i] + width
plt.bar(xxxxxx, MAPE_KRR, width=width, label='KRR',tick_label = name_list)



plt.legend()
plt.title("MAPE_1")
plt.xlabel("point")
plt.ylabel("MAPE(%)")
plt.show()
