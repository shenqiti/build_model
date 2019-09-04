'''
By:shenqiti
2019/9/4
改写数据

'''

import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import statsmodels.stats.diagnostic




INPUT_PATH = 'D:/python_code/NEW_step/try_paper/mycode/new/Brent_pro1.xlsx'
OUT_PUT = 'D:/python_code/NEW_step/try_paper/mycode/new/Brent_pro11.csv'


data = pd.read_excel(INPUT_PATH)
ID = np.array(data["id"])
price = np.array(data["price"])
P = []
id = []

for k in range(len(price)):
    for i in range(0,64):
        P.append(price[k])
        id.append(ID[k])

fn = open(OUT_PUT,'w')

for i in range(0,len(P)):
    fn.write(str(id[i]))
    fn.write(',')
    fn.write(str(P[i]))
    fn.write('\n')

fn.close()
