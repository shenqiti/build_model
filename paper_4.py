'''
By:shenqiti
2019/9/4
改写时变趋势格式
'''


import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import statsmodels.stats.diagnostic








INPUT_PATH = 'D:/python_code/NEW_step/try_paper/mycode/new/Poly-[2, 4, 6, 8, 10, 12, 15, 20][2, 4, 6, 8, 10, 12, 15, 20]-WINParams-2019-08-03.csv'
OUT_PUT = 'D:/python_code/NEW_step/try_paper/mycode/new/new2/alll~.csv'
data = pd.read_csv(INPUT_PATH)
NO = np.array(data["No"])
i = np.array(data["i"])
istart = np.array(data["istart"])
iend = np.array(data["iend"])
MSE_FinalFunc = np.array(data["MSE_FinalFunc"])
MSEtrenda = np.array(data["MSEtrenda"])
MSEtrendb = np.array(data["MSEtrendb"])
MSEtrendc = np.array(data["MSEtrendc"])
MSEtrendd = np.array(data["MSEtrendd"])
MSEtrende = np.array(data["MSEtrende"])
a = np.array(data["a"])
b = np.array(data["b"])

dic = {}
for k in range(0,len(NO)):
    if i[k] not in dic:
        dic[i[k]] = [[i[k],istart[k],iend[k],MSE_FinalFunc[k],MSEtrenda[k],MSEtrendb[k],MSEtrendc[k],MSEtrendd[k],MSEtrende[k],a[k],b[k]]]
    else:
        dic[i[k]].append([i[k],istart[k],iend[k],MSE_FinalFunc[k],MSEtrenda[k],MSEtrendb[k],MSEtrendc[k],MSEtrendd[k],MSEtrende[k],a[k],b[k]])


fn = open(OUT_PUT,'w')
for t in range(2,5536):
    for m in range(0,len(dic[t])):
        for each in dic[t][m]:
            fn.write(str(each))
            fn.write(',')
        fn.write('\n')
fn.close()
